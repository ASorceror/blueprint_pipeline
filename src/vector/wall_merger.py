"""
Wall Merger Module

Detects and merges double-line walls into centerlines.
Commercial blueprints often draw walls as two parallel lines.
"""

import logging
import math
from dataclasses import dataclass
from typing import List, Tuple, Set, Optional

from ..constants import (
    PARALLEL_ANGLE_TOLERANCE_DEG,
    DOUBLE_LINE_DISTANCE_MAX,
)
from .extractor import Segment

logger = logging.getLogger(__name__)


@dataclass
class MergedWall:
    """A wall segment with optional thickness information."""
    segment: Segment
    thickness: Optional[float] = None
    is_merged: bool = False


def angle_difference(angle1: float, angle2: float) -> float:
    """
    Calculate the difference between two angles (0-180 range).

    Args:
        angle1: First angle in degrees
        angle2: Second angle in degrees

    Returns:
        Absolute difference in degrees (0-90)
    """
    diff = abs(angle1 - angle2)
    # Handle wrap-around at 180 degrees
    if diff > 90:
        diff = 180 - diff
    return diff


def segments_are_parallel(seg_a: Segment, seg_b: Segment) -> bool:
    """
    Check if two segments are approximately parallel.

    Args:
        seg_a: First segment
        seg_b: Second segment

    Returns:
        True if segments are parallel within tolerance
    """
    angle_diff = angle_difference(seg_a.angle, seg_b.angle)
    return angle_diff < PARALLEL_ANGLE_TOLERANCE_DEG


def perpendicular_distance(seg_a: Segment, seg_b: Segment) -> float:
    """
    Calculate the perpendicular distance between two parallel line segments.

    Uses the midpoint of seg_b and measures distance to line of seg_a.

    Args:
        seg_a: First segment (reference line)
        seg_b: Second segment

    Returns:
        Perpendicular distance in points
    """
    return seg_a.distance_to_point(seg_b.midpoint)


def segments_overlap_lengthwise(seg_a: Segment, seg_b: Segment) -> bool:
    """
    Check if two parallel segments overlap when projected onto their common direction.

    Args:
        seg_a: First segment
        seg_b: Second segment

    Returns:
        True if segments have overlapping extent
    """
    # Project onto common axis (use seg_a's direction)
    angle_rad = math.radians(seg_a.angle)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

    # Project all endpoints onto the line direction
    def project(point):
        return point[0] * cos_a + point[1] * sin_a

    a_start_proj = project(seg_a.start)
    a_end_proj = project(seg_a.end)
    b_start_proj = project(seg_b.start)
    b_end_proj = project(seg_b.end)

    # Get ranges
    a_min, a_max = min(a_start_proj, a_end_proj), max(a_start_proj, a_end_proj)
    b_min, b_max = min(b_start_proj, b_end_proj), max(b_start_proj, b_end_proj)

    # Check overlap
    return a_min < b_max and b_min < a_max


def lengths_are_similar(seg_a: Segment, seg_b: Segment, tolerance: float = 0.2) -> bool:
    """
    Check if two segments have similar lengths (within 20% by default).

    Args:
        seg_a: First segment
        seg_b: Second segment
        tolerance: Maximum relative difference (default 0.2 = 20%)

    Returns:
        True if lengths are similar
    """
    len_a = seg_a.length
    len_b = seg_b.length

    if len_a == 0 or len_b == 0:
        return False

    ratio = max(len_a, len_b) / min(len_a, len_b)
    return ratio <= (1 + tolerance)


def calculate_centerline(seg_a: Segment, seg_b: Segment) -> Segment:
    """
    Calculate the centerline between two parallel segments.

    Args:
        seg_a: First segment
        seg_b: Second segment

    Returns:
        New segment representing the centerline
    """
    # Average the endpoints
    # First, we need to align the segments (ensure same direction)
    # Use the midpoint of each segment and the average length

    mid_a = seg_a.midpoint
    mid_b = seg_b.midpoint

    # Centerline midpoint
    center_mid = ((mid_a[0] + mid_b[0]) / 2, (mid_a[1] + mid_b[1]) / 2)

    # Use average angle (should be very close anyway)
    avg_angle_rad = math.radians((seg_a.angle + seg_b.angle) / 2)

    # Use the longer segment's length
    half_length = max(seg_a.length, seg_b.length) / 2

    # Calculate endpoints
    dx = math.cos(avg_angle_rad) * half_length
    dy = math.sin(avg_angle_rad) * half_length

    start = (center_mid[0] - dx, center_mid[1] - dy)
    end = (center_mid[0] + dx, center_mid[1] + dy)

    # Use average width (default to 1.0 if None)
    width_a = seg_a.width if seg_a.width is not None else 1.0
    width_b = seg_b.width if seg_b.width is not None else 1.0
    avg_width = (width_a + width_b) / 2

    return Segment(start=start, end=end, width=avg_width)


def detect_and_merge_double_walls(segments: List[Segment]) -> Tuple[List[Segment], int]:
    """
    Detect and merge double-line walls to centerlines.

    Algorithm from spec 2.3:
    1. Build spatial index of all segments
    2. For each segment A, find nearby parallel segments
    3. If parallel and close, mark as double-line pair
    4. Calculate centerline for pairs
    5. Return merged segments

    Args:
        segments: List of wall segments

    Returns:
        Tuple of (merged segment list, number of pairs merged)
    """
    if not segments:
        return [], 0

    n = len(segments)
    paired: Set[int] = set()
    pairs: List[Tuple[int, int, float]] = []  # (idx_a, idx_b, distance)

    # Step 2-3: Find double-line pairs
    # Note: O(n²) but typically n is manageable for blueprints
    for i in range(n):
        if i in paired:
            continue

        seg_a = segments[i]

        for j in range(i + 1, n):
            if j in paired:
                continue

            seg_b = segments[j]

            # Check if parallel
            if not segments_are_parallel(seg_a, seg_b):
                continue

            # Check perpendicular distance
            perp_dist = perpendicular_distance(seg_a, seg_b)
            if perp_dist > DOUBLE_LINE_DISTANCE_MAX:
                continue

            # Check if lengths are similar (within 20%)
            if not lengths_are_similar(seg_a, seg_b):
                continue

            # Check if they overlap lengthwise
            if not segments_overlap_lengthwise(seg_a, seg_b):
                continue

            # Found a double-line pair
            pairs.append((i, j, perp_dist))
            paired.add(i)
            paired.add(j)
            break  # Move to next segment

    # Step 4-5: Create centerlines for pairs and collect unpaired segments
    result_segments = []

    for idx_a, idx_b, thickness in pairs:
        centerline = calculate_centerline(segments[idx_a], segments[idx_b])
        result_segments.append(centerline)

    # Add unpaired segments unchanged
    for i, seg in enumerate(segments):
        if i not in paired:
            result_segments.append(seg)

    logger.info(
        f"Double-line wall merger: {len(pairs)} pairs merged, "
        f"{n} -> {len(result_segments)} segments"
    )

    return result_segments, len(pairs)
