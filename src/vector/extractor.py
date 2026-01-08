"""
Vector Extractor Module

Functions for extracting line segments from vector PDF pages.
"""

import logging
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import pymupdf

from ..constants import (
    MIN_SEGMENT_LENGTH_POINTS,
    MAX_SEGMENT_LENGTH_RATIO,
    ANNOTATION_LINE_WIDTH_MAX,
    WALL_LINE_WIDTH_MIN,
    BORDER_LINE_WIDTH_MIN,
    DEFINITE_WALL_WIDTH_MIN,
    MEDIUM_LINE_WIDTH_MIN,
    DRAWING_AREA_MARGIN_POINTS,
)

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """
    A line segment with metadata.

    Represents a single line segment extracted from a PDF vector path,
    including visual properties (width, color, dashes) and optional
    construction phase classification.

    Attributes:
        start: Start point (x, y) in PDF points
        end: End point (x, y) in PDF points
        width: Line width in PDF points
        color: Stroke color as RGB tuple (0-1 range), None for black
        dashes: Dash pattern [dash, gap, ...], None for solid
        fill: Fill color as RGB tuple (0-1 range), None for no fill
        fill_type: PyMuPDF draw type ('s'=stroke, 'f'=fill, 'fs'=both)
        construction_phase: Classified phase (NEW, EXISTING, etc.)
        phase_confidence: Confidence in phase classification (0-1)
        phase_method: Method used for classification
    """
    start: Tuple[float, float]
    end: Tuple[float, float]
    width: float
    color: Optional[Tuple[float, ...]] = None
    dashes: Optional[List[float]] = None  # Dash pattern from PDF [dash, gap, ...]

    # Fill information (for construction phase detection)
    fill: Optional[Tuple[float, ...]] = None  # Fill color from PyMuPDF
    fill_type: Optional[str] = None  # 's'=stroke, 'f'=fill, 'fs'=both

    # Construction phase classification
    construction_phase: Optional[str] = None  # ConstructionPhase value
    phase_confidence: float = 0.0
    phase_method: Optional[str] = None  # ClassificationMethod value

    @property
    def is_dashed(self) -> bool:
        """Check if this segment has a dash pattern (non-solid line)."""
        return self.dashes is not None and len(self.dashes) > 0

    @property
    def length(self) -> float:
        """Calculate segment length."""
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return math.sqrt(dx * dx + dy * dy)

    @property
    def angle(self) -> float:
        """Calculate segment angle in degrees (0-180)."""
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        angle = math.degrees(math.atan2(dy, dx))
        # Normalize to 0-180 (direction doesn't matter)
        if angle < 0:
            angle += 180
        if angle >= 180:
            angle -= 180
        return angle

    @property
    def midpoint(self) -> Tuple[float, float]:
        """Get the midpoint of the segment."""
        return (
            (self.start[0] + self.end[0]) / 2,
            (self.start[1] + self.end[1]) / 2
        )

    def distance_to_point(self, point: Tuple[float, float]) -> float:
        """Calculate perpendicular distance from point to line."""
        x0, y0 = point
        x1, y1 = self.start
        x2, y2 = self.end

        # Line length
        line_len = self.length
        if line_len == 0:
            return math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

        # Perpendicular distance formula
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        return numerator / line_len

    @property
    def has_fill(self) -> bool:
        """Check if this segment has a fill color."""
        return self.fill is not None

    @property
    def is_stroke_only(self) -> bool:
        """Check if this segment is stroke-only (no fill)."""
        return self.fill is None or self.fill_type == 's'

    @property
    def is_gray_fill(self) -> bool:
        """
        Check if this segment has a gray fill (typically EXISTING construction).

        Gray fills in the 0.25-0.75 range indicate existing construction
        per industry convention.
        """
        if self.fill is None:
            return False

        if len(self.fill) < 3:
            gray = self.fill[0]
        else:
            r, g, b = self.fill[0], self.fill[1], self.fill[2]
            # Check if grayscale
            if max(abs(r - g), abs(g - b), abs(r - b)) > 0.05:
                return False
            gray = (r + g + b) / 3

        return 0.25 <= gray <= 0.75

    def set_phase(self, phase: str, confidence: float, method: str) -> None:
        """
        Set the construction phase classification.

        Args:
            phase: ConstructionPhase value as string
            confidence: Classification confidence (0-1)
            method: ClassificationMethod value as string
        """
        self.construction_phase = phase
        self.phase_confidence = confidence
        self.phase_method = method


def extract_all_paths(page: pymupdf.Page) -> List[dict]:
    """
    Extract all drawing paths from a PDF page.

    Args:
        page: pymupdf.Page object

    Returns:
        List of path dictionaries containing points, width, color, fill
    """
    try:
        drawings = page.get_drawings()
        logger.debug(f"Extracted {len(drawings)} drawing paths from page")
        return drawings
    except Exception as e:
        logger.warning(f"Error extracting paths: {e}")
        return []


def _parse_dash_pattern(dashes) -> Optional[List[float]]:
    """
    Parse dash pattern from PyMuPDF format.
    
    PyMuPDF returns dashes as:
    - None or [] for solid lines
    - [dash, gap] for simple dashed
    - [dash, gap, dash, gap, ...] for complex patterns
    
    Args:
        dashes: Dash pattern from PyMuPDF
        
    Returns:
        Parsed dash pattern or None for solid lines
    """
    if dashes is None:
        return None
    if isinstance(dashes, (list, tuple)) and len(dashes) > 0:
        # Filter out zero values and convert to list
        pattern = [float(d) for d in dashes if d > 0]
        return pattern if pattern else None
    return None


def path_to_segments(path: dict) -> List[Segment]:
    """
    Convert a drawing path to line segments.

    Extracts line segments from PyMuPDF path dictionaries, preserving
    visual properties including fill information for construction phase detection.

    Args:
        path: Dictionary containing path data from pymupdf with keys:
            - items: List of draw commands
            - width: Line width
            - color: Stroke color (RGB tuple)
            - fill: Fill color (RGB tuple or None)
            - type: Draw type ('s'=stroke, 'f'=fill, 'fs'=both)
            - dashes: Dash pattern

    Returns:
        List of Segment objects with fill info preserved
    """
    segments = []

    # Get line width, color, and dash pattern
    width = path.get("width", 0)
    color = path.get("color", None)
    dashes = _parse_dash_pattern(path.get("dashes", None))

    # Get fill information for construction phase detection
    fill = path.get("fill", None)
    fill_type = path.get("type", None)  # 's'=stroke, 'f'=fill, 'fs'=both

    # Get the items in the path
    items = path.get("items", [])

    current_point = None

    for item in items:
        item_type = item[0]

        if item_type == "l":  # Line
            # item = ("l", start_point, end_point)
            start = (item[1].x, item[1].y)
            end = (item[2].x, item[2].y)
            segments.append(Segment(
                start=start, end=end, width=width, color=color, dashes=dashes,
                fill=fill, fill_type=fill_type
            ))
            current_point = end

        elif item_type == "m":  # Move
            # item = ("m", point)
            current_point = (item[1].x, item[1].y)

        elif item_type == "c":  # Curve (Bezier)
            # item = ("c", p1, p2, p3, p4) - cubic Bezier
            # Approximate curve with line segments
            if current_point is not None:
                # For simplicity, just connect start to end
                # More sophisticated: subdivide curve
                end = (item[4].x, item[4].y)
                segments.append(Segment(
                    start=current_point, end=end, width=width, color=color, dashes=dashes,
                    fill=fill, fill_type=fill_type
                ))
                current_point = end

        elif item_type == "re":  # Rectangle
            # item = ("re", rect)
            rect = item[1]
            # Rectangle has 4 corners
            p1 = (rect.x0, rect.y0)
            p2 = (rect.x1, rect.y0)
            p3 = (rect.x1, rect.y1)
            p4 = (rect.x0, rect.y1)
            segments.append(Segment(
                start=p1, end=p2, width=width, color=color, dashes=dashes,
                fill=fill, fill_type=fill_type
            ))
            segments.append(Segment(
                start=p2, end=p3, width=width, color=color, dashes=dashes,
                fill=fill, fill_type=fill_type
            ))
            segments.append(Segment(
                start=p3, end=p4, width=width, color=color, dashes=dashes,
                fill=fill, fill_type=fill_type
            ))
            segments.append(Segment(
                start=p4, end=p1, width=width, color=color, dashes=dashes,
                fill=fill, fill_type=fill_type
            ))

        elif item_type == "qu":  # Quad
            # item = ("qu", quad)
            quad = item[1]
            # Quad has 4 points
            points = [(quad.ul.x, quad.ul.y), (quad.ur.x, quad.ur.y),
                      (quad.lr.x, quad.lr.y), (quad.ll.x, quad.ll.y)]
            for i in range(4):
                segments.append(Segment(
                    start=points[i],
                    end=points[(i + 1) % 4],
                    width=width,
                    color=color,
                    dashes=dashes,
                    fill=fill,
                    fill_type=fill_type
                ))

    return segments


def paths_to_segments(paths: List[dict]) -> List[Segment]:
    """
    Convert all drawing paths to line segments.

    Args:
        paths: List of path dictionaries

    Returns:
        List of all Segment objects
    """
    all_segments = []
    for path in paths:
        segments = path_to_segments(path)
        all_segments.extend(segments)

    logger.debug(f"Converted {len(paths)} paths to {len(all_segments)} segments")
    return all_segments




def analyze_line_width_distribution(segments: List[Segment]) -> dict:
    """
    Analyze the distribution of line widths in the segments.
    
    This helps understand the drawing's structure:
    - Very thin lines (< 0.25): Annotations, dimensions, hatching
    - Medium lines (0.25 - 1.0): May be walls or other elements
    - Thick lines (>= 1.0): Likely walls
    - Very thick lines (>= 2.0): Border/frame lines
    
    Args:
        segments: List of all segments
        
    Returns:
        Dictionary with width statistics
    """
    from collections import Counter
    
    if not segments:
        return {"total": 0, "distribution": {}}
    
    widths = [round(seg.width, 1) if seg.width else 0 for seg in segments]
    width_counts = Counter(widths)
    
    # Categorize
    thin = sum(1 for w in widths if w < ANNOTATION_LINE_WIDTH_MAX)
    medium = sum(1 for w in widths if ANNOTATION_LINE_WIDTH_MAX <= w < DEFINITE_WALL_WIDTH_MIN)
    thick = sum(1 for w in widths if DEFINITE_WALL_WIDTH_MIN <= w < BORDER_LINE_WIDTH_MIN)
    border = sum(1 for w in widths if w >= BORDER_LINE_WIDTH_MIN)
    hairline = sum(1 for w in widths if w == 0)
    
    total = len(segments)
    
    stats = {
        "total": total,
        "hairline_count": hairline,
        "thin_count": thin,
        "medium_count": medium,
        "thick_count": thick,
        "border_count": border,
        "distribution": dict(sorted(width_counts.items())),
        "hairline_pct": round(hairline / total * 100, 1) if total else 0,
        "thin_pct": round(thin / total * 100, 1) if total else 0,
        "medium_pct": round(medium / total * 100, 1) if total else 0,
        "thick_pct": round(thick / total * 100, 1) if total else 0,
        "border_pct": round(border / total * 100, 1) if total else 0,
    }
    
    logger.debug(f"Line width analysis: {stats['total']} segments - "
                f"hairline:{stats['hairline_pct']}%, thin:{stats['thin_pct']}%, "
                f"medium:{stats['medium_pct']}%, thick:{stats['thick_pct']}%, "
                f"border:{stats['border_pct']}%")
    
    return stats


def detect_drawing_area_bounds(
    segments: List[Segment],
    page_width: float,
    page_height: float
) -> Optional[Tuple[float, float, float, float]]:
    """
    Detect the main drawing area bounds by finding thick border lines.
    
    Architectural drawings typically have a border/frame around the drawing
    area, with title block in the bottom-right corner. The thick lines
    (width >= BORDER_LINE_WIDTH_MIN) often define this boundary.
    
    Args:
        segments: List of all segments
        page_width: Page width in points
        page_height: Page height in points
        
    Returns:
        Tuple of (x0, y0, x1, y1) for drawing area bounds, or None if not detected
    """
    # Find thick segments that could be border lines
    border_segments = [
        seg for seg in segments 
        if seg.width is not None and seg.width >= BORDER_LINE_WIDTH_MIN
    ]
    
    if len(border_segments) < 4:
        logger.debug(f"Only {len(border_segments)} thick segments found, not enough for border detection")
        return None
    
    # Get bounding box of thick lines
    min_x = min(min(seg.start[0], seg.end[0]) for seg in border_segments)
    max_x = max(max(seg.start[0], seg.end[0]) for seg in border_segments)
    min_y = min(min(seg.start[1], seg.end[1]) for seg in border_segments)
    max_y = max(max(seg.start[1], seg.end[1]) for seg in border_segments)
    
    # Check if this is a reasonable drawing area (not too small, not entire page)
    width = max_x - min_x
    height = max_y - min_y
    
    # Drawing area should be at least 50% of page in each dimension
    if width < page_width * 0.5 or height < page_height * 0.5:
        logger.debug(f"Detected area too small: {width:.0f}x{height:.0f} vs page {page_width:.0f}x{page_height:.0f}")
        return None
    
    # Should leave some margin for title block
    if width > page_width * 0.98 and height > page_height * 0.98:
        logger.debug("Detected area covers entire page, likely not a proper border")
        return None
    
    logger.info(f"Detected drawing area bounds: ({min_x:.0f}, {min_y:.0f}) to ({max_x:.0f}, {max_y:.0f})")
    return (min_x, min_y, max_x, max_y)


def filter_segments_by_drawing_area(
    segments: List[Segment],
    drawing_bounds: Tuple[float, float, float, float],
    margin: float = DRAWING_AREA_MARGIN_POINTS
) -> List[Segment]:
    """
    Filter segments to keep only those within the drawing area.
    
    This excludes elements in title blocks, legends, and borders.
    
    Args:
        segments: List of segments to filter
        drawing_bounds: (x0, y0, x1, y1) bounds of drawing area
        margin: Extra margin inside the bounds
        
    Returns:
        Filtered list of segments within drawing area
    """
    x0, y0, x1, y1 = drawing_bounds
    
    # Apply margin
    x0 += margin
    y0 += margin
    x1 -= margin
    y1 -= margin
    
    filtered = []
    for seg in segments:
        # Check if segment midpoint is within bounds
        mx, my = seg.midpoint
        if x0 <= mx <= x1 and y0 <= my <= y1:
            filtered.append(seg)
    
    excluded = len(segments) - len(filtered)
    logger.info(f"Drawing area filter: {len(segments)} -> {len(filtered)} segments ({excluded} excluded)")
    return filtered


def filter_wall_segments(
    segments: List[Segment],
    page_width: float,
    page_height: float,
    strict_width_filter: bool = False
) -> List[Segment]:
    """
    Filter segments to keep only likely wall segments.

    Filter criteria (improved based on PDF analysis):
    1. Length >= MIN_SEGMENT_LENGTH_POINTS
    2. Length <= page_diagonal * MAX_SEGMENT_LENGTH_RATIO
    3. Width-based classification:
       - Hairlines (width=0): KEEP (CAD exports often use this)
       - Thin (< 0.25): SKIP (annotations, dimensions)
       - Medium (0.25-1.0): KEEP if strict_width_filter=False, else depends
       - Thick (>= 1.0): KEEP (definite walls)
       - Border (>= 2.0): SKIP (frame/border lines)
    4. Skip border lines (very thick, very long)

    Args:
        segments: List of all segments
        page_width: Page width in points
        page_height: Page height in points
        strict_width_filter: If True, only keep hairlines and thick lines

    Returns:
        Filtered list of wall segments
    """
    page_diagonal = math.sqrt(page_width ** 2 + page_height ** 2)
    max_length = page_diagonal * MAX_SEGMENT_LENGTH_RATIO

    wall_segments = []
    skipped_thin = 0
    skipped_border = 0
    skipped_length = 0

    for seg in segments:
        # Filter 1: Minimum length
        if seg.length < MIN_SEGMENT_LENGTH_POINTS:
            skipped_length += 1
            continue

        # Filter 2: Maximum length (border lines)
        if seg.length > max_length:
            skipped_length += 1
            continue

        # Filter 3: Line width classification
        width = seg.width if seg.width is not None else 0
        
        # Hairlines (width=0) - often used in CAD exports for walls
        if width == 0:
            wall_segments.append(seg)
            continue
            
        # Very thin lines - likely annotations/dimensions
        if width < ANNOTATION_LINE_WIDTH_MAX:
            skipped_thin += 1
            continue
        
        # Border/frame lines - too thick, skip
        if width >= BORDER_LINE_WIDTH_MIN:
            skipped_border += 1
            continue
        
        # Definite wall lines (1.0 - 2.0)
        if width >= DEFINITE_WALL_WIDTH_MIN:
            wall_segments.append(seg)
            continue
        
        # Medium lines (0.25 - 1.0)
        if strict_width_filter:
            # In strict mode, skip medium lines (they're likely annotations)
            skipped_thin += 1
            continue
        else:
            # In normal mode, keep medium lines as potential walls
            wall_segments.append(seg)

    logger.info(f"Wall filter: {len(segments)} -> {len(wall_segments)} segments "
                f"(skipped: {skipped_thin} thin, {skipped_border} border, {skipped_length} length)")
    return wall_segments


def extract_wall_segments(
    page: pymupdf.Page,
    use_drawing_area_filter: bool = False,  # Disabled by default for v4 compatibility
    strict_width_filter: bool = False
) -> Tuple[List[Segment], dict]:
    """
    Extract wall segments from a PDF page.

    This is the main entry point for segment extraction.
    
    Enhanced to:
    1. Analyze line width distribution
    2. Detect drawing area bounds from thick border lines
    3. Filter segments to exclude title block area
    4. Apply improved width-based filtering

    Args:
        page: pymupdf.Page object
        use_drawing_area_filter: If True, exclude segments outside drawing area
        strict_width_filter: If True, only keep hairlines and thick lines

    Returns:
        Tuple of (List of wall segments, extraction info dict)
    """
    # Get page dimensions
    rect = page.rect
    page_width = rect.width
    page_height = rect.height

    # Extract all paths
    paths = extract_all_paths(page)

    # Convert to segments
    segments = paths_to_segments(paths)
    
    extraction_info = {
        "total_segments": len(segments),
        "page_width": page_width,
        "page_height": page_height,
        "drawing_bounds": None,
    }
    
    if not segments:
        return [], extraction_info
    
    # Analyze line width distribution
    width_stats = analyze_line_width_distribution(segments)
    extraction_info["width_stats"] = width_stats
    
    # Detect drawing area bounds from thick border lines
    drawing_bounds = None
    if use_drawing_area_filter:
        drawing_bounds = detect_drawing_area_bounds(segments, page_width, page_height)
        extraction_info["drawing_bounds"] = drawing_bounds
    
    # Filter by drawing area if bounds detected
    if drawing_bounds:
        segments = filter_segments_by_drawing_area(segments, drawing_bounds)
        extraction_info["segments_in_drawing_area"] = len(segments)

    # Filter to wall segments
    wall_segments = filter_wall_segments(
        segments, page_width, page_height, 
        strict_width_filter=strict_width_filter
    )
    extraction_info["wall_segments"] = len(wall_segments)

    return wall_segments, extraction_info


def extract_wall_segments_simple(page: pymupdf.Page) -> List[Segment]:
    """
    Simple extraction without enhanced features (for backwards compatibility).
    
    Args:
        page: pymupdf.Page object
        
    Returns:
        List of wall segments
    """
    segments, _ = extract_wall_segments(page, use_drawing_area_filter=False)
    return segments
