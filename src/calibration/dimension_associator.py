"""
Dimension Associator Module

Links dimension text to the line segments they measure.
"""

import logging
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import pymupdf

from ..constants import (
    DIMENSION_TICK_SEARCH_RADIUS,
    DIMENSION_LINE_SEARCH_RADIUS,
    Confidence,
)
from ..vector.extractor import Segment

logger = logging.getLogger(__name__)


@dataclass
class DimensionAssociation:
    """A dimension text associated with a line segment."""
    text: str
    value_inches: float
    text_bbox: Tuple[float, float, float, float]
    segment: Optional[Segment]
    pdf_length: float  # Length of associated segment in PDF points
    confidence: str
    scale_factor: Optional[float] = None  # Calculated if segment found


def get_text_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Get the center point of a bounding box."""
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2, (y0 + y1) / 2)


def get_text_orientation(bbox: Tuple[float, float, float, float]) -> str:
    """
    Determine if text is horizontal or vertical based on bbox aspect ratio.

    Args:
        bbox: (x0, y0, x1, y1)

    Returns:
        "horizontal" or "vertical"
    """
    x0, y0, x1, y1 = bbox
    width = abs(x1 - x0)
    height = abs(y1 - y0)

    return "horizontal" if width > height else "vertical"


def point_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate distance between two points."""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def is_segment_parallel_to_orientation(segment: Segment, orientation: str) -> bool:
    """Check if segment is roughly parallel to given orientation."""
    angle = segment.angle

    if orientation == "horizontal":
        # Horizontal: angle near 0 or 180 degrees
        return angle < 15 or angle > 165
    else:
        # Vertical: angle near 90 degrees
        return 75 < angle < 105


def find_tick_marks(
    page: pymupdf.Page,
    center: Tuple[float, float],
    orientation: str,
    radius: float = DIMENSION_TICK_SEARCH_RADIUS
) -> List[Tuple[float, float]]:
    """
    Find tick marks near a dimension text.

    Tick marks are short line segments perpendicular to the dimension orientation.

    Args:
        page: PDF page
        center: Center point to search around
        orientation: "horizontal" or "vertical"
        radius: Search radius in points

    Returns:
        List of tick mark positions
    """
    tick_positions = []

    try:
        drawings = page.get_drawings()

        for drawing in drawings:
            for item in drawing.get("items", []):
                if item[0] == "l":  # Line
                    start = (item[1].x, item[1].y)
                    end = (item[2].x, item[2].y)

                    # Check if it's a short segment (tick mark)
                    length = point_distance(start, end)
                    if length > 20:  # Too long for tick mark
                        continue

                    # Check if it's near the dimension text
                    mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
                    if point_distance(mid, center) > radius * 10:
                        continue

                    # Check if it's perpendicular to dimension orientation
                    seg_angle = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))
                    if seg_angle < 0:
                        seg_angle += 180

                    if orientation == "horizontal":
                        # Tick should be vertical (angle ~90)
                        if 75 < seg_angle < 105:
                            tick_positions.append(mid)
                    else:
                        # Tick should be horizontal (angle ~0 or ~180)
                        if seg_angle < 15 or seg_angle > 165:
                            tick_positions.append(mid)

    except Exception as e:
        logger.debug(f"Error finding tick marks: {e}")

    return tick_positions


def find_associated_segment(
    dim_center: Tuple[float, float],
    dim_orientation: str,
    segments: List[Segment],
    tick_positions: List[Tuple[float, float]] = None
) -> Tuple[Optional[Segment], str]:
    """
    Find the line segment that a dimension text is measuring.

    Algorithm from spec 3.4:
    1. If tick marks found, find segment whose endpoints match tick positions
    2. Otherwise, find nearby parallel segment

    Args:
        dim_center: Center of dimension text
        dim_orientation: "horizontal" or "vertical"
        segments: List of segments to search
        tick_positions: Optional list of tick mark positions

    Returns:
        Tuple of (segment or None, confidence level)
    """
    # Method 1: Match tick marks to segment endpoints
    if tick_positions and len(tick_positions) >= 2:
        for seg in segments:
            # Check if segment endpoints are near tick marks
            start_near_tick = any(
                point_distance(seg.start, tick) < DIMENSION_TICK_SEARCH_RADIUS
                for tick in tick_positions
            )
            end_near_tick = any(
                point_distance(seg.end, tick) < DIMENSION_TICK_SEARCH_RADIUS
                for tick in tick_positions
            )

            if start_near_tick and end_near_tick:
                logger.debug(f"Found segment via tick marks: {seg.length:.1f} pts")
                return seg, Confidence.HIGH

    # Method 2: Find nearby parallel segment
    candidates = []
    for seg in segments:
        # Check if parallel to dimension orientation
        if not is_segment_parallel_to_orientation(seg, dim_orientation):
            continue

        # Check distance to dimension text
        dist = seg.distance_to_point(dim_center)
        if dist < DIMENSION_LINE_SEARCH_RADIUS:
            candidates.append((seg, dist))

    if len(candidates) == 1:
        seg, _ = candidates[0]
        logger.debug(f"Found single candidate segment: {seg.length:.1f} pts")
        return seg, Confidence.MEDIUM

    elif len(candidates) > 1:
        # Return closest
        candidates.sort(key=lambda x: x[1])
        seg, _ = candidates[0]
        logger.debug(f"Found closest segment from {len(candidates)} candidates: {seg.length:.1f} pts")
        return seg, Confidence.LOW

    return None, Confidence.NONE


def associate_dimensions_to_segments(
    page: pymupdf.Page,
    dimension_matches: List[Tuple[str, float, Tuple[float, float, float, float]]],
    segments: List[Segment]
) -> List[DimensionAssociation]:
    """
    Associate dimension text matches with line segments.

    Args:
        page: PDF page (for finding tick marks)
        dimension_matches: List of (text, value_inches, bbox)
        segments: List of segments

    Returns:
        List of DimensionAssociation objects with calculated scale factors
    """
    associations = []

    for text, value_inches, bbox in dimension_matches:
        center = get_text_center(bbox)
        orientation = get_text_orientation(bbox)

        # Find tick marks
        tick_positions = find_tick_marks(page, center, orientation)

        # Find associated segment
        segment, confidence = find_associated_segment(
            center, orientation, segments, tick_positions
        )

        pdf_length = segment.length if segment else 0

        # Calculate scale factor if we have both values
        scale_factor = None
        if segment and value_inches > 0:
            value_feet = value_inches / 12
            if value_feet > 0:
                scale_factor = pdf_length / value_feet

        assoc = DimensionAssociation(
            text=text,
            value_inches=value_inches,
            text_bbox=bbox,
            segment=segment,
            pdf_length=pdf_length,
            confidence=confidence,
            scale_factor=scale_factor
        )
        associations.append(assoc)

        if segment:
            logger.info(
                f"Dimension '{text}' ({value_inches:.1f} in) -> "
                f"segment {pdf_length:.1f} pts, scale={scale_factor:.2f} pts/ft"
                if scale_factor else
                f"Dimension '{text}' ({value_inches:.1f} in) -> segment {pdf_length:.1f} pts"
            )

    return associations
