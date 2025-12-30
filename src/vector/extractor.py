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
)

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """A line segment with metadata."""
    start: Tuple[float, float]
    end: Tuple[float, float]
    width: float
    color: Optional[Tuple[float, ...]] = None

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


def path_to_segments(path: dict) -> List[Segment]:
    """
    Convert a drawing path to line segments.

    Args:
        path: Dictionary containing path data from pymupdf

    Returns:
        List of Segment objects
    """
    segments = []

    # Get line width and color
    width = path.get("width", 0)
    color = path.get("color", None)

    # Get the items in the path
    items = path.get("items", [])

    current_point = None

    for item in items:
        item_type = item[0]

        if item_type == "l":  # Line
            # item = ("l", start_point, end_point)
            start = (item[1].x, item[1].y)
            end = (item[2].x, item[2].y)
            segments.append(Segment(start=start, end=end, width=width, color=color))
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
                segments.append(Segment(start=current_point, end=end, width=width, color=color))
                current_point = end

        elif item_type == "re":  # Rectangle
            # item = ("re", rect)
            rect = item[1]
            # Rectangle has 4 corners
            p1 = (rect.x0, rect.y0)
            p2 = (rect.x1, rect.y0)
            p3 = (rect.x1, rect.y1)
            p4 = (rect.x0, rect.y1)
            segments.append(Segment(start=p1, end=p2, width=width, color=color))
            segments.append(Segment(start=p2, end=p3, width=width, color=color))
            segments.append(Segment(start=p3, end=p4, width=width, color=color))
            segments.append(Segment(start=p4, end=p1, width=width, color=color))

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
                    color=color
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


def filter_wall_segments(
    segments: List[Segment],
    page_width: float,
    page_height: float
) -> List[Segment]:
    """
    Filter segments to keep only likely wall segments.

    Filter criteria (from spec 2.2):
    1. Length >= MIN_SEGMENT_LENGTH_POINTS
    2. Length <= page_diagonal * MAX_SEGMENT_LENGTH_RATIO
    3. Width >= WALL_LINE_WIDTH_MIN OR width is 0 (hairline)
    4. Not a dashed line (if detectable)

    Args:
        segments: List of all segments
        page_width: Page width in points
        page_height: Page height in points

    Returns:
        Filtered list of wall segments
    """
    page_diagonal = math.sqrt(page_width ** 2 + page_height ** 2)
    max_length = page_diagonal * MAX_SEGMENT_LENGTH_RATIO

    wall_segments = []

    for seg in segments:
        # Filter 1: Minimum length
        if seg.length < MIN_SEGMENT_LENGTH_POINTS:
            continue

        # Filter 2: Maximum length (border lines)
        if seg.length > max_length:
            continue

        # Filter 3: Line width
        # Width 0 = hairline, often used for walls
        # Width >= WALL_LINE_WIDTH_MIN = likely wall
        # Width < ANNOTATION_LINE_WIDTH_MAX = annotation (skip unless hairline)
        if seg.width > 0 and seg.width < WALL_LINE_WIDTH_MIN:
            # This might be an annotation line
            # But also keep medium-width lines
            if seg.width < ANNOTATION_LINE_WIDTH_MAX:
                continue

        wall_segments.append(seg)

    logger.info(f"Filtered {len(segments)} segments to {len(wall_segments)} wall segments")
    return wall_segments


def extract_wall_segments(page: pymupdf.Page) -> List[Segment]:
    """
    Extract wall segments from a PDF page.

    This is the main entry point for segment extraction.

    Args:
        page: pymupdf.Page object

    Returns:
        List of wall segments
    """
    # Get page dimensions
    rect = page.rect
    page_width = rect.width
    page_height = rect.height

    # Extract all paths
    paths = extract_all_paths(page)

    # Convert to segments
    segments = paths_to_segments(paths)

    # Filter to wall segments
    wall_segments = filter_wall_segments(segments, page_width, page_height)

    return wall_segments
