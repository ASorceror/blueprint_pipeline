"""
Pipeline Orchestration Module

Coordinates the full extraction workflow from PDF to output files.
"""

import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

import pymupdf
from shapely.geometry import Polygon

from .constants import (
    PageType,
    ProcessingPath,
    Confidence,
    DEFAULT_CEILING_HEIGHT_FT,
    DEFAULT_RENDER_DPI,
    DEFAULT_GAP_TOLERANCE_INCHES,
    DEFAULT_COMMERCIAL_SCALE,
    DEFAULT_COMMERCIAL_SCALE_FACTOR,
    POINTS_PER_INCH,
)
from .pdf.reader import open_pdf, get_page_count, get_page_dimensions, render_page_to_image
from .pdf.classifier import classify_page_type, classify_page
from .vector.extractor import extract_wall_segments
from .vector.wall_merger import detect_and_merge_double_walls
from .vector.polygonizer import bridge_gaps, segments_to_polygons, filter_room_polygons
from .calibration.scale_detector import (
    extract_scale_from_page,
    parse_manual_scale,
    check_scale_conflict,
)
from .calibration.unit_converter import (
    pdf_points_to_real,
    area_points_to_real,
    calculate_scale_factor_from_calibration,
    parse_calibration_string,
)
# dimension_associator not used directly in pipeline
from .text.ocr_engine import extract_all_text
from .text.label_matcher import filter_room_labels, match_labels_to_polygons
from .text.height_parser import (
    extract_heights_from_page,
    match_heights_to_rooms,
    normalize_room_name,
)
from .raster.preprocessor import preprocess_image
from .raster.room_detector import detect_rooms_from_raster
from .geometry.room import Room
from .geometry.calculator import (
    calculate_floor_area,
    calculate_perimeter,
    calculate_wall_area,
    calculate_ceiling_area,
    validate_room_measurements,
    create_room_from_polygon,
)
from .output.csv_writer import write_rooms_to_csv, generate_csv_filename
from .output.json_writer import write_rooms_to_json, generate_json_filename
from .output.pdf_annotator import create_annotated_pdf, generate_annotated_pdf_filename


logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""
    input_pdf: str
    output_dir: str
    pages: str = "all"
    units: str = "imperial"
    manual_scale: Optional[str] = None
    calibration: Optional[str] = None
    default_height: float = DEFAULT_CEILING_HEIGHT_FT
    gap_tolerance_inches: float = DEFAULT_GAP_TOLERANCE_INCHES
    dpi: int = DEFAULT_RENDER_DPI
    no_ocr: bool = False
    no_annotate: bool = False
    verbose: bool = False
    debug: bool = False


@dataclass
class PageResult:
    """Result from processing a single page."""
    page_num: int
    page_type: str
    processing_path: str
    rooms: List[Room]
    scale_factor: float
    warnings: List[str]
    processing_time: float


@dataclass
class PipelineResult:
    """Result from full pipeline execution."""
    input_file: str
    output_dir: str
    total_pages: int
    pages_processed: int
    total_rooms: int
    rooms: List[Room]
    scale_used: str
    units: str
    warnings: List[str]
    csv_path: Optional[str]
    json_path: Optional[str]
    annotated_pdf_path: Optional[str]
    processing_time: float


def get_pages_to_process(page_spec: str, total_pages: int) -> List[int]:
    """
    Parse page specification to list of 0-indexed page numbers.

    Args:
        page_spec: Page specification (e.g., "all", "1-5", "1,3,5")
        total_pages: Total pages in PDF

    Returns:
        List of 0-indexed page numbers
    """
    if page_spec.lower() == "all":
        return list(range(total_pages))

    pages = set()
    for part in page_spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            start = int(start) - 1
            end = int(end) - 1
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part) - 1)

    return sorted([p for p in pages if 0 <= p < total_pages])


def detect_scale(
    doc: pymupdf.Document,
    page_nums: List[int],
    manual_scale: Optional[str] = None,
    calibration: Optional[str] = None,
    verbose: bool = False
) -> Tuple[float, str, List[str]]:
    """
    Detect or apply scale factor.

    Args:
        doc: PDF document
        page_nums: Pages to check
        manual_scale: Manual scale override
        calibration: Two-point calibration string
        verbose: Enable verbose logging

    Returns:
        Tuple of (scale_factor, scale_description, warnings)
    """
    warnings = []

    # Manual scale override
    if manual_scale:
        scale_result = parse_manual_scale(manual_scale)
        if scale_result:
            logger.info(f"Using manual scale: {scale_result.detected_notation}")
            return scale_result.scale_factor, scale_result.detected_notation, warnings
        else:
            warnings.append(f"Could not parse manual scale: {manual_scale}")

    # Two-point calibration
    if calibration:
        try:
            (x1, y1), (x2, y2), real_dist = parse_calibration_string(calibration)
            scale_factor = calculate_scale_factor_from_calibration(
                (x1, y1), (x2, y2), real_dist
            )
            logger.info(f"Using calibration: {calibration} -> {scale_factor:.2f}")
            return scale_factor, f"calibration: {calibration}", warnings
        except Exception as e:
            warnings.append(f"Calibration failed: {e}")

    # Auto-detect from pages
    detected_scales = []
    for page_num in page_nums[:5]:  # Check first 5 pages
        page = doc[page_num]
        scale_result = extract_scale_from_page(page)
        if scale_result and scale_result.scale_source != "default":
            detected_scales.append((page_num, scale_result))
            if verbose:
                logger.debug(f"Page {page_num + 1}: detected scale {scale_result.detected_notation}")

    if detected_scales:
        # Use first detected scale
        page_num, first_scale = detected_scales[0]
        scale_factor = first_scale.scale_factor
        scale_text = first_scale.detected_notation

        # Check for conflicts
        for other_page, other_scale in detected_scales[1:]:
            if check_scale_conflict(scale_factor, other_scale.scale_factor):
                warnings.append(
                    f"Scale conflict: page {page_num + 1} ({scale_factor:.2f}) "
                    f"vs page {other_page + 1} ({other_scale.scale_factor:.2f})"
                )

        logger.info(f"Auto-detected scale: {scale_text} ({scale_factor:.2f})")
        return scale_factor, scale_text, warnings

    # Fall back to default
    warnings.append(f"No scale detected, using default: {DEFAULT_COMMERCIAL_SCALE}")
    logger.warning(f"Using default scale: {DEFAULT_COMMERCIAL_SCALE}")
    return DEFAULT_COMMERCIAL_SCALE_FACTOR, DEFAULT_COMMERCIAL_SCALE, warnings


def process_vector_page(
    page: pymupdf.Page,
    page_num: int,
    scale_factor: float,
    config: PipelineConfig
) -> Tuple[List[Room], List[str]]:
    """
    Process a vector PDF page.

    Args:
        page: PyMuPDF page
        page_num: Page number (0-indexed)
        scale_factor: Scale factor for conversion
        config: Pipeline configuration

    Returns:
        Tuple of (rooms, warnings)
    """
    warnings = []
    rooms = []

    page_width, page_height = get_page_dimensions(page)
    page_name = f"Page {page_num + 1}"

    # Extract text (for labels)
    text_blocks = []
    if not config.no_ocr:
        try:
            text_blocks = extract_all_text(page, config.dpi, use_ocr=True)
        except Exception as e:
            warnings.append(f"Text extraction failed: {e}")
            text_blocks = extract_all_text(page, config.dpi, use_ocr=False)
    else:
        text_blocks = extract_all_text(page, config.dpi, use_ocr=False)

    # Filter for room labels
    room_labels = filter_room_labels(text_blocks)

    # Extract vector segments
    segments = extract_wall_segments(page)
    if not segments:
        warnings.append(f"No vector segments found on page {page_num + 1}")
        return rooms, warnings

    # Merge double walls
    merged_segments, _ = detect_and_merge_double_walls(segments)

    # Bridge gaps
    gap_tolerance_pts = config.gap_tolerance_inches * POINTS_PER_INCH
    bridged_segments, bridge_count = bridge_gaps(merged_segments, gap_tolerance_pts)

    # Polygonize (includes filtering)
    room_polygons, debug_info = segments_to_polygons(bridged_segments, page_width, page_height)

    if not room_polygons:
        warnings.append(f"No valid room polygons on page {page_num + 1}")
        return rooms, warnings

    # Extract Shapely polygons for label matching
    shapely_polygons = [rp.shapely_polygon for rp in room_polygons]

    # Match labels to polygons
    label_matches = match_labels_to_polygons(room_labels, shapely_polygons)

    # Create room objects
    for i, room_polygon in enumerate(room_polygons):
        room_id = f"room_{page_num:03d}_{i:03d}"

        # Get matched label
        room_name = f"ROOM {i + 1}"
        label_confidence = Confidence.NONE
        for label in label_matches:
            if label.polygon_index == i:
                room_name = label.text
                label_confidence = label.confidence
                break

        # Create room
        vertices = list(room_polygon.shapely_polygon.exterior.coords)[:-1]
        room = create_room_from_polygon(
            polygon_id=room_id,
            polygon=room_polygon.shapely_polygon,
            vertices=vertices,
            room_name=room_name,
            sheet_number=page_num,
            scale_factor=scale_factor,
            ceiling_height_ft=config.default_height,
            source="vector",
            confidence=str(label_confidence) if label_confidence else Confidence.MEDIUM
        )
        room.sheet_name = page_name
        room.name_confidence = label_confidence

        # Validate
        validation_warnings = validate_room_measurements(room)
        if validation_warnings:
            room.warnings = validation_warnings
            warnings.extend([f"{room_name}: {w}" for w in validation_warnings])

        rooms.append(room)

    return rooms, warnings


def process_raster_page(
    page: pymupdf.Page,
    page_num: int,
    scale_factor: float,
    config: PipelineConfig
) -> Tuple[List[Room], List[str]]:
    """
    Process a raster/scanned PDF page.

    Args:
        page: PyMuPDF page
        page_num: Page number (0-indexed)
        scale_factor: Scale factor for conversion
        config: Pipeline configuration

    Returns:
        Tuple of (rooms, warnings)
    """
    warnings = []
    rooms = []

    page_width, page_height = get_page_dimensions(page)
    page_name = f"Page {page_num + 1}"

    # Render page to image
    try:
        image = render_page_to_image(page, config.dpi)
    except Exception as e:
        warnings.append(f"Failed to render page: {e}")
        return rooms, warnings

    # Extract text
    text_blocks = []
    if not config.no_ocr:
        try:
            text_blocks = extract_all_text(page, config.dpi, use_ocr=True)
        except Exception as e:
            warnings.append(f"OCR failed: {e}")

    room_labels = filter_room_labels(text_blocks)

    # Preprocess image
    preprocessing_result = preprocess_image(image, config.dpi)

    # Detect rooms
    raster_rooms, quality = detect_rooms_from_raster(
        preprocessing_result.binary,
        page_width, page_height,
        config.dpi,
        room_labels
    )

    if quality == "POOR":
        warnings.append(f"Low quality room detection on page {page_num + 1}")

    # Convert to Room objects
    for i, (polygon, label) in enumerate(raster_rooms):
        room_id = f"room_{page_num:03d}_{i:03d}"
        room_name = label if label else f"ROOM {i + 1}"

        vertices = list(polygon.exterior.coords)[:-1] if hasattr(polygon, 'exterior') else []
        room = create_room_from_polygon(
            polygon_id=room_id,
            polygon=polygon,
            vertices=vertices,
            room_name=room_name,
            sheet_number=page_num,
            scale_factor=scale_factor,
            ceiling_height_ft=config.default_height,
            source="raster",
            confidence=Confidence.LOW if label else Confidence.NONE
        )
        room.sheet_name = page_name
        room.name_confidence = Confidence.LOW if label else Confidence.NONE

        validation_warnings = validate_room_measurements(room)
        if validation_warnings:
            room.warnings = validation_warnings

        rooms.append(room)

    return rooms, warnings


def extract_ceiling_heights(
    doc: pymupdf.Document,
    page_nums: List[int],
    rooms: List[Room],
    config: PipelineConfig
) -> List[str]:
    """
    Extract ceiling heights from RCP pages and match to rooms.

    Args:
        doc: PDF document
        page_nums: Pages to check for RCP
        rooms: Rooms to update
        config: Pipeline configuration

    Returns:
        List of warnings
    """
    warnings = []
    height_annotations = []

    # Find RCP pages and extract heights
    for page_num in page_nums:
        page = doc[page_num]
        page_type = classify_page_type(page)

        if page_type == PageType.RCP:
            # Extract height annotations from RCP page
            annotations = extract_heights_from_page(page, force_ocr=not config.no_ocr)
            height_annotations.extend(annotations)

            if config.verbose:
                logger.debug(f"Page {page_num + 1} (RCP): {len(annotations)} heights")

    # Match heights to rooms
    if height_annotations:
        match_heights_to_rooms(rooms, height_annotations, config.default_height)
        logger.info(f"Matched {len(height_annotations)} height annotations to rooms")
    else:
        # Apply default height to all rooms
        for room in rooms:
            if room.ceiling_height_ft == 0 or room.ceiling_height_ft == config.default_height:
                room.ceiling_height_ft = config.default_height
                room.height_source = "default"

        if rooms:
            warnings.append(f"No RCP heights found, using default: {config.default_height} ft")

    # Recalculate wall areas with actual heights
    for room in rooms:
        room.wall_area_sqft = room.perimeter_ft * room.ceiling_height_ft
        room.ceiling_area_sqft = room.floor_area_sqft  # Flat ceiling assumption

    return warnings


def run_pipeline(args) -> PipelineResult:
    """
    Run the full extraction pipeline.

    Args:
        args: Parsed command-line arguments

    Returns:
        PipelineResult with all outputs
    """
    start_time = time.time()

    # Create config from args
    config = PipelineConfig(
        input_pdf=args.input,
        output_dir=args.output,
        pages=args.pages,
        units=args.units,
        manual_scale=getattr(args, 'scale', None),
        calibration=getattr(args, 'calib', None),
        default_height=args.default_height,
        gap_tolerance_inches=args.door_gap,
        dpi=args.dpi,
        no_ocr=args.no_ocr,
        no_annotate=args.no_annotate,
        verbose=args.verbose,
        debug=args.debug,
    )

    # Setup logging
    log_level = logging.DEBUG if config.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(message)s')

    logger.info(f"Processing: {config.input_pdf}")

    all_warnings = []
    all_rooms = []

    # Open PDF
    doc = open_pdf(config.input_pdf)
    total_pages = get_page_count(doc)

    # Determine pages to process
    page_nums = get_pages_to_process(config.pages, total_pages)
    logger.info(f"Processing {len(page_nums)} of {total_pages} pages")

    # Detect scale
    scale_factor, scale_used, scale_warnings = detect_scale(
        doc, page_nums,
        config.manual_scale,
        config.calibration,
        config.verbose
    )
    all_warnings.extend(scale_warnings)

    # Process each page
    floor_plan_pages = []
    for page_num in page_nums:
        page = doc[page_num]
        page_classification = classify_page(page, page_num)
        page_type = page_classification.page_type
        processing_path = page_classification.processing_path

        if config.verbose:
            logger.debug(f"Page {page_num + 1}: {page_type}, path: {processing_path}")

        # Skip non-floor-plan pages (they're processed for heights later)
        if page_type != PageType.FLOOR_PLAN:
            continue

        floor_plan_pages.append(page_num)

        # Process based on path
        if processing_path == ProcessingPath.VECTOR:
            rooms, warnings = process_vector_page(page, page_num, scale_factor, config)
        elif processing_path == ProcessingPath.RASTER:
            rooms, warnings = process_raster_page(page, page_num, scale_factor, config)
        elif processing_path == ProcessingPath.HYBRID:
            # Try vector first, fall back to raster
            rooms, warnings = process_vector_page(page, page_num, scale_factor, config)
            if not rooms:
                rooms, raster_warnings = process_raster_page(page, page_num, scale_factor, config)
                warnings.extend(raster_warnings)
        else:
            warnings = [f"Skipping page {page_num + 1}"]
            rooms = []

        all_rooms.extend(rooms)
        all_warnings.extend(warnings)

        if config.verbose:
            logger.debug(f"Page {page_num + 1}: {len(rooms)} rooms")

    # Extract ceiling heights from RCP pages
    height_warnings = extract_ceiling_heights(doc, page_nums, all_rooms, config)
    all_warnings.extend(height_warnings)

    # Generate outputs
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = None
    json_path = None
    annotated_path = None

    if all_rooms:
        # CSV output
        csv_path = generate_csv_filename(config.input_pdf, config.output_dir)
        write_rooms_to_csv(all_rooms, csv_path)
        logger.info(f"CSV written: {csv_path}")

        # JSON output
        json_path = generate_json_filename(config.input_pdf, config.output_dir)
        write_rooms_to_json(
            all_rooms, json_path,
            input_file=config.input_pdf,
            scale_used=scale_used,
            units=config.units,
            include_geometry=True
        )
        logger.info(f"JSON written: {json_path}")

        # Annotated PDF
        if not config.no_annotate:
            annotated_path = generate_annotated_pdf_filename(config.input_pdf, config.output_dir)
            create_annotated_pdf(config.input_pdf, annotated_path, all_rooms)
            logger.info(f"Annotated PDF written: {annotated_path}")
    else:
        all_warnings.append("No rooms extracted from any page")
        logger.warning("No rooms extracted")

    doc.close()

    processing_time = time.time() - start_time

    # Summary
    total_area = sum(r.floor_area_sqft for r in all_rooms)
    logger.info(f"\nSummary:")
    logger.info(f"  Pages processed: {len(floor_plan_pages)}")
    logger.info(f"  Rooms extracted: {len(all_rooms)}")
    logger.info(f"  Total floor area: {total_area:,.0f} SF")
    logger.info(f"  Processing time: {processing_time:.1f}s")

    if all_warnings and config.verbose:
        logger.info(f"\nWarnings ({len(all_warnings)}):")
        for w in all_warnings[:10]:
            logger.info(f"  - {w}")
        if len(all_warnings) > 10:
            logger.info(f"  ... and {len(all_warnings) - 10} more")

    return PipelineResult(
        input_file=config.input_pdf,
        output_dir=config.output_dir,
        total_pages=total_pages,
        pages_processed=len(floor_plan_pages),
        total_rooms=len(all_rooms),
        rooms=all_rooms,
        scale_used=scale_used,
        units=config.units,
        warnings=all_warnings,
        csv_path=csv_path,
        json_path=json_path,
        annotated_pdf_path=annotated_path,
        processing_time=processing_time,
    )
