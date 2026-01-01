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
from PIL import Image
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
    MAX_VECTOR_SEGMENTS,
    TITLE_BLOCK_SEARCH_REGION_START,
    TITLE_BLOCK_SEARCH_REGION_END,
    TITLE_BLOCK_DEFAULT_X1,
)
from .pdf.reader import open_pdf, get_page_count, get_page_dimensions, render_page_to_image
from .pdf.classifier import classify_page_type, classify_page
from .vector.extractor import extract_wall_segments, extract_wall_segments_simple
from .vector.wall_merger import detect_and_merge_double_walls
from .vector.polygonizer import bridge_gaps, segments_to_polygons, filter_room_polygons
from .vector.filters import SegmentFilterPipeline, FilterConfig
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
from .detection.hybrid_detector import (
    HybridRoomDetector,
    create_rooms_from_detection,
)

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
    # Segment filtering options
    enable_filters: bool = True
    filter_title_block: bool = True
    filter_hatching: bool = True
    filter_dimension: bool = True
    filter_annotation: bool = True
    # Label-driven detection options
    label_driven: bool = True  # Use label-driven detection by default
    label_min_confidence: float = 0.5
    boundary_min_completeness: float = 0.5


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
    config: PipelineConfig,
    filter_pipeline: Optional[SegmentFilterPipeline] = None
) -> Tuple[List[Room], List[str]]:
    """
    Process a vector PDF page.

    Args:
        page: PyMuPDF page
        page_num: Page number (0-indexed)
        scale_factor: Scale factor for conversion
        config: Pipeline configuration
        filter_pipeline: Optional segment filter pipeline

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
            # Render page to image for OCR
            page_image = render_page_to_image(page, config.dpi)
            text_blocks = extract_all_text(page, page_image=page_image, force_ocr=True)
        except Exception as e:
            warnings.append(f"Text extraction failed: {e}")
            text_blocks = extract_all_text(page, page_image=None, force_ocr=False)
    else:
        text_blocks = extract_all_text(page, page_image=None, force_ocr=False)

    # Filter for room labels
    room_labels = filter_room_labels(text_blocks)

    # Extract vector segments
    segments, extraction_info = extract_wall_segments(page)
    if not segments:
        warnings.append(f"No vector segments found on page {page_num + 1}")
        return rooms, warnings

    original_segment_count = len(segments)

    # Check for overly complex CAD files
    if len(segments) > MAX_VECTOR_SEGMENTS:
        logger.warning(
            f"Page {page_num + 1} has {len(segments)} segments (>{MAX_VECTOR_SEGMENTS}). "
            f"Complex CAD file - processing may be slow."
        )
        warnings.append(f"Complex CAD: {len(segments)} segments detected")

    # Apply segment filters (title block, hatching, etc.)
    if filter_pipeline is not None and config.enable_filters:
        segments, filter_stats = filter_pipeline.filter(
            segments, page_width, page_height, text_blocks
        )
        if filter_stats.total_removed > 0:
            logger.info(
                f"Page {page_num + 1}: Filtered {filter_stats.total_removed} "
                f"of {original_segment_count} segments "
                f"({filter_stats.total_removal_rate:.1%})"
            )

    if not segments:
        warnings.append(f"No segments remaining after filtering on page {page_num + 1}")
        return rooms, warnings

    # Merge double walls
    merged_segments, _ = detect_and_merge_double_walls(segments)

    # Bridge gaps
    gap_tolerance_pts = config.gap_tolerance_inches * POINTS_PER_INCH
    bridged_segments, bridge_count = bridge_gaps(merged_segments, gap_tolerance_pts)

    # Polygonize (includes filtering)
    room_polygons, debug_info = segments_to_polygons(bridged_segments, page_width, page_height)

    # Filter polygons to drawing area (exclude title block rooms)
    if filter_pipeline is not None and config.enable_filters:
        room_polygons, polygon_removed = filter_pipeline.filter_polygons(
            room_polygons, page_width, page_height
        )
        if polygon_removed > 0:
            logger.debug(f"Page {page_num + 1}: Filtered {polygon_removed} polygons outside drawing area")

    if not room_polygons:
        warnings.append(f"No valid room polygons on page {page_num + 1}")
        return rooms, warnings

    # Match labels to polygons (pass RoomPolygon list, not Shapely polygons)
    label_matches = match_labels_to_polygons(room_labels, room_polygons)

    # Create room objects
    for i, room_polygon in enumerate(room_polygons):
        room_id = f"room_{page_num:03d}_{i:03d}"

        # Get matched label by polygon_id
        room_name = f"ROOM {i + 1}"
        label_confidence = Confidence.NONE
        for match in label_matches:
            if match.polygon_id == room_polygon.polygon_id:
                room_name = match.room_name
                label_confidence = match.name_confidence
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

    # Extract text using rendered image for OCR
    text_blocks = []
    if not config.no_ocr:
        try:
            text_blocks = extract_all_text(page, page_image=image, force_ocr=True)
        except Exception as e:
            warnings.append(f"OCR failed: {e}")

    room_labels = filter_room_labels(text_blocks)

    # Preprocess image
    preprocessing_result = preprocess_image(image, config.dpi)

    # Transform label centroids from PDF coordinates to image coordinates
    # PDF: origin bottom-left, Y increases upward
    # Image: origin top-left, Y increases downward
    label_centroids_image = None
    if room_labels:
        img_h, img_w = preprocessing_result.binary_image.shape[:2]
        label_centroids_image = []
        for label in room_labels:
            pdf_x, pdf_y = label.centroid
            # Transform PDF coords to image coords
            img_x = (pdf_x / page_width) * img_w
            img_y = ((page_height - pdf_y) / page_height) * img_h  # Flip Y
            label_centroids_image.append((img_x, img_y))

    detection_result = detect_rooms_from_raster(
        preprocessing_result.binary_image,
        page_width, page_height,
        config.dpi,
        label_centroids_image
    )

    if detection_result.quality == "POOR":
        warnings.append(f"Low quality room detection on page {page_num + 1}")

    # Match labels to detected room polygons
    label_matches = match_labels_to_polygons(room_labels, detection_result.rooms)

    # Convert to Room objects
    for i, room_polygon in enumerate(detection_result.rooms):
        room_id = f"room_{page_num:03d}_{i:03d}"

        # Get matched label by polygon_id
        room_name = f"ROOM {i + 1}"
        label_confidence = Confidence.NONE
        for match in label_matches:
            if match.polygon_id == room_polygon.polygon_id:
                room_name = match.room_name
                label_confidence = match.name_confidence
                break

        room = create_room_from_polygon(
            polygon_id=room_id,
            polygon=room_polygon.shapely_polygon,
            vertices=room_polygon.vertices,
            room_name=room_name,
            sheet_number=page_num,
            scale_factor=scale_factor,
            ceiling_height_ft=config.default_height,
            source="raster",
            confidence=room_polygon.confidence
        )
        room.sheet_name = page_name
        room.name_confidence = label_confidence

        validation_warnings = validate_room_measurements(room)
        if validation_warnings:
            room.warnings = validation_warnings

        rooms.append(room)

    return rooms, warnings


def process_label_driven_page(
    page: pymupdf.Page,
    page_num: int,
    scale_factor: float,
    config: PipelineConfig,
    filter_pipeline: Optional[SegmentFilterPipeline] = None
) -> Tuple[List[Room], List[str]]:
    """
    Process a page using label-driven detection.

    This is the NEW approach where room labels are the primary source of truth.
    Boundaries are built around labels using nearby wall segments.

    Args:
        page: PyMuPDF page
        page_num: Page number (0-indexed)
        scale_factor: Scale factor for conversion
        config: Pipeline configuration
        filter_pipeline: Optional segment filter pipeline

    Returns:
        Tuple of (rooms, warnings)
    """
    warnings = []
    page_width, page_height = get_page_dimensions(page)
    page_name = f"Page {page_num + 1}"

    # Step 1: Extract text for labels
    text_blocks = []
    if not config.no_ocr:
        try:
            page_image = render_page_to_image(page, config.dpi)
            text_blocks = extract_all_text(page, page_image=page_image, force_ocr=True)
        except Exception as e:
            warnings.append(f"Text extraction failed: {e}")
            text_blocks = extract_all_text(page, page_image=None, force_ocr=False)
    else:
        text_blocks = extract_all_text(page, page_image=None, force_ocr=False)

    if not text_blocks:
        warnings.append(f"No text found on page {page_num + 1}")
        return [], warnings

    # Step 2: Extract wall segments
    segments_raw, extraction_info = extract_wall_segments(page)
    if not segments_raw:
        warnings.append(f"No wall segments found on page {page_num + 1}")

    # Apply segment filters if available
    if filter_pipeline is not None and config.enable_filters and segments_raw:
        segments_raw, filter_stats = filter_pipeline.filter(
            segments_raw, page_width, page_height, text_blocks
        )
        if filter_stats.total_removed > 0:
            logger.debug(
                f"Page {page_num + 1}: Filtered {filter_stats.total_removed} segments"
            )

    # Convert segments to tuples for hybrid detector
    wall_segments = []
    for seg in segments_raw:
        wall_segments.append((
            seg.start[0], seg.start[1], seg.end[0], seg.end[1],
            seg.width if hasattr(seg, 'width') else 1.0
        ))

    # Step 3: Determine drawing bounds (exclude title block)
    drawing_bounds = None
    if filter_pipeline is not None:
        bounds = filter_pipeline.get_drawing_bounds()
        if bounds and bounds.title_block_detected:
            drawing_bounds = (
                0, 0,
                bounds.x_max * page_width,
                page_height
            )

    if drawing_bounds is None:
        # Default: exclude right 15% for title block
        drawing_bounds = (0, 0, page_width * 0.85, page_height)

    # Step 4: Run hybrid detection
    detector = HybridRoomDetector(
        drawing_bounds=drawing_bounds,
        min_label_confidence=config.label_min_confidence,
        min_boundary_completeness=config.boundary_min_completeness,
        scale_factor=scale_factor
    )

    detection_result = detector.detect_rooms(
        text_blocks=text_blocks,
        wall_segments=wall_segments,
        page_width=page_width,
        page_height=page_height
    )

    # Step 5: Convert to Room objects
    rooms = create_rooms_from_detection(
        detection_result=detection_result,
        page_num=page_num,
        scale_factor=scale_factor,
        ceiling_height_ft=config.default_height
    )

    # Set page name
    for room in rooms:
        room.sheet_name = page_name

    # Add detection warnings
    warnings.extend(detection_result.warnings)

    logger.info(
        f"Page {page_num + 1} (label-driven): {len(rooms)} rooms from "
        f"{detection_result.labels_found} labels"
    )

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
    # Determine if label-driven detection is enabled
    label_driven = not getattr(args, 'geometry_only', False)

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
        # Filter options
        enable_filters=not getattr(args, 'no_filters', False),
        filter_title_block=not getattr(args, 'no_title_block_filter', False),
        filter_hatching=not getattr(args, 'no_hatching_filter', False),
        filter_dimension=not getattr(args, 'no_dimension_filter', False),
        filter_annotation=not getattr(args, 'no_annotation_filter', False),
        # Label-driven detection options
        label_driven=label_driven,
        label_min_confidence=getattr(args, 'label_confidence', 0.5),
        boundary_min_completeness=getattr(args, 'boundary_completeness', 0.5),
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

    # Initialize segment filter pipeline
    filter_pipeline = None
    if config.enable_filters:
        # Create filter configuration
        debug_output_dir = Path(config.output_dir) / "debug" if config.debug else None
        filter_config = FilterConfig(
            enable_drawing_area=config.filter_title_block,
            enable_hatching=config.filter_hatching,
            enable_dimension=config.filter_dimension,
            enable_annotation=config.filter_annotation,
            title_block_search_region=(TITLE_BLOCK_SEARCH_REGION_START, TITLE_BLOCK_SEARCH_REGION_END),
            title_block_default_x1=TITLE_BLOCK_DEFAULT_X1,
            debug_output_dir=debug_output_dir,
        )
        filter_pipeline = SegmentFilterPipeline(filter_config)

        # Initialize filter pipeline with sample page images
        sample_page_nums = page_nums[:5] if len(page_nums) >= 5 else page_nums[:3]
        if not sample_page_nums:
            sample_page_nums = page_nums[:1] if page_nums else []

        sample_images = []
        for page_num in sample_page_nums:
            page = doc[page_num]
            img_array = render_page_to_image(page, dpi=100)  # Low DPI for detection
            # Convert BGR numpy array to RGB PIL Image
            img_rgb = img_array[:, :, ::-1]  # BGR to RGB
            pil_img = Image.fromarray(img_rgb)
            sample_images.append(pil_img)

        if sample_images:
            filter_pipeline.initialize(sample_images)
            bounds = filter_pipeline.get_drawing_bounds()
            if bounds and bounds.title_block_detected:
                logger.info(
                    f"Title block detected at x={bounds.title_block_x1:.1%} "
                    f"(drawing area: 0-{bounds.x_max:.1%})"
                )

    # Process each page
    floor_plan_pages = []
    for page_num in page_nums:
        page = doc[page_num]
        page_classification = classify_page(page, page_num)
        page_type = page_classification.page_type
        processing_path = page_classification.processing_path

        if config.verbose:
            logger.debug(f"Page {page_num + 1}: {page_type}, path: {processing_path}")

        # Skip non-floor-plan pages for VECTOR path (text classification is reliable)
        # But process RASTER pages even if classified as OTHER (scanned images have no text)
        if page_type != PageType.FLOOR_PLAN:
            if processing_path == ProcessingPath.VECTOR:
                continue
            elif processing_path == ProcessingPath.SKIP:
                continue
            # For RASTER/HYBRID with OTHER type, still try to process (may be scanned floor plan)

        floor_plan_pages.append(page_num)

        # Process based on mode and path
        if config.label_driven:
            # NEW: Label-driven detection (primary)
            rooms, warnings = process_label_driven_page(
                page, page_num, scale_factor, config, filter_pipeline
            )
            # Fall back to geometry if label-driven finds no rooms
            if not rooms:
                logger.info(
                    f"Page {page_num + 1}: Label-driven found no rooms, "
                    f"falling back to geometry"
                )
                if processing_path == ProcessingPath.VECTOR:
                    rooms, warnings = process_vector_page(
                        page, page_num, scale_factor, config, filter_pipeline
                    )
                elif processing_path in (ProcessingPath.RASTER, ProcessingPath.HYBRID):
                    rooms, warnings = process_raster_page(
                        page, page_num, scale_factor, config
                    )
        else:
            # OLD: Geometry-first detection
            if processing_path == ProcessingPath.VECTOR:
                rooms, warnings = process_vector_page(page, page_num, scale_factor, config, filter_pipeline)
                # Fall back to raster if vector yields no rooms
                if not rooms:
                    logger.info(f"Page {page_num + 1}: Vector extraction found no rooms, falling back to raster")
                    rooms, raster_warnings = process_raster_page(page, page_num, scale_factor, config)
                    warnings.extend(raster_warnings)
            elif processing_path == ProcessingPath.RASTER:
                rooms, warnings = process_raster_page(page, page_num, scale_factor, config)
            elif processing_path == ProcessingPath.HYBRID:
                # Try vector first, fall back to raster
                rooms, warnings = process_vector_page(page, page_num, scale_factor, config, filter_pipeline)
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

    # Filter out artifact rooms (too small to be real rooms)
    from .constants import MIN_AREA_REAL_SQFT
    original_count = len(all_rooms)
    all_rooms = [r for r in all_rooms if r.floor_area_sqft >= MIN_AREA_REAL_SQFT]
    filtered_count = original_count - len(all_rooms)
    if filtered_count > 0:
        logger.info(f"Filtered out {filtered_count} rooms below {MIN_AREA_REAL_SQFT} SF minimum")

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
