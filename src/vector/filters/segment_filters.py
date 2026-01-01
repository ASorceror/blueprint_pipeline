"""
Segment Filters

Filters for removing non-wall segments from vector extraction:
- HatchingFilter: Removes cross-hatch and fill patterns
- DimensionLineFilter: Removes dimension annotation lines
- AnnotationLineFilter: Removes thin annotation/markup lines
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# FILTER CONSTANTS (can be moved to constants.py)
# =============================================================================

# Hatching detection
HATCHING_ANGLE_TOLERANCE_DEG = 2.0  # Lines within this angle are considered parallel
HATCHING_MIN_GROUP_SIZE = 5  # Minimum parallel lines to consider hatching
HATCHING_MAX_SPACING_RATIO = 0.05  # Max spacing as ratio of line length
HATCHING_LENGTH_VARIANCE_MAX = 0.3  # Max length variance within a group

# Dimension line detection
DIMENSION_LINE_WIDTH_MAX = 0.5  # Thin lines are likely dimensions
DIMENSION_ARROW_ANGLE_MIN = 30  # Min angle for arrow head detection
DIMENSION_ARROW_ANGLE_MAX = 60  # Max angle for arrow head detection
DIMENSION_TEXT_PROXIMITY_PX = 20  # Search radius for dimension text

# Annotation line detection
ANNOTATION_LINE_WIDTH_MAX = 0.3  # Very thin lines are annotations
ANNOTATION_MIN_SEGMENTS = 3  # Min connected segments for annotation path
# Grid line detection
GRID_LINE_MAX_WIDTH = 1.5  # Grid lines are thin (0.5-1.5pt)
GRID_LINE_MIN_SPAN_RATIO = 0.7  # Must span 70% of page dimension
GRID_LINE_ANGLE_TOLERANCE_DEG = 2.0  # Must be horizontal or vertical
GRID_LINE_EDGE_THRESHOLD_POINTS = 100.0  # Endpoints within 100pt of page edge




@dataclass
class FilterStats:
    """Statistics from a filter operation."""
    total_input: int = 0
    total_output: int = 0
    removed: int = 0
    reason: str = ""

    @property
    def removal_rate(self) -> float:
        if self.total_input == 0:
            return 0.0
        return self.removed / self.total_input



# =============================================================================
# GRID LINE FILTER
# =============================================================================

class GridLineFilter:
    """
    Detects and removes architectural grid lines.

    Grid lines are structural reference lines characterized by:
    - Thin line width (0.5-1.5pt)
    - Dashed line pattern (long-dash-short-dash center line style)
    - Full page span (70-100% of page width/height)
    - Strictly horizontal or vertical orientation
    - Endpoints near page edges
    - Letters (A, B, C) or numbers (1, 2, 3) in circles at page margins

    These MUST be filtered BEFORE room boundary detection to prevent
    false wall detection.
    """

    def __init__(
        self,
        max_width: float = GRID_LINE_MAX_WIDTH,
        min_span_ratio: float = GRID_LINE_MIN_SPAN_RATIO,
        angle_tolerance: float = GRID_LINE_ANGLE_TOLERANCE_DEG,
        edge_threshold: float = GRID_LINE_EDGE_THRESHOLD_POINTS,
        require_dashed: bool = False
    ):
        """
        Initialize the grid line filter.

        Args:
            max_width: Maximum line width for grid lines (pts)
            min_span_ratio: Minimum span as ratio of page dimension
            angle_tolerance: Angle tolerance for H/V classification (degrees)
            edge_threshold: Max distance from page edge for endpoints
            require_dashed: If True, only filter dashed lines as grid lines
        """
        self.max_width = max_width
        self.min_span_ratio = min_span_ratio
        self.angle_tolerance = angle_tolerance
        self.edge_threshold = edge_threshold
        self.require_dashed = require_dashed

    def filter(
        self,
        segments: List[Any],
        page_width: float,
        page_height: float
    ) -> Tuple[List[Any], FilterStats]:
        """
        Remove grid lines from segments.

        Args:
            segments: List of Segment objects
            page_width: Page width in PDF points
            page_height: Page height in PDF points

        Returns:
            Tuple of (filtered_segments, stats)
        """
        if not segments:
            return [], FilterStats(reason="grid_line")

        grid_line_indices: Set[int] = set()

        for i, seg in enumerate(segments):
            if self._is_grid_line(seg, page_width, page_height):
                grid_line_indices.add(i)

        # Filter out grid line segments
        filtered = [
            seg for i, seg in enumerate(segments)
            if i not in grid_line_indices
        ]

        stats = FilterStats(
            total_input=len(segments),
            total_output=len(filtered),
            removed=len(grid_line_indices),
            reason="grid_line"
        )

        if stats.removed > 0:
            logger.info(
                f"GridLineFilter: Removed {stats.removed} segments "
                f"({stats.removal_rate:.1%} of total)"
            )

        return filtered, stats

    def _is_grid_line(
        self,
        seg: Any,
        page_width: float,
        page_height: float
    ) -> bool:
        """
        Check if a segment is a grid line based on multiple criteria.

        Returns True if segment matches grid line characteristics:
        1. Width <= max_width
        2. Is dashed (if require_dashed is True)
        3. Spans >= min_span_ratio of page dimension
        4. Is horizontal or vertical (within angle tolerance)
        5. Has endpoints near page edges
        """
        # Criterion 1: Line width
        width = seg.width if hasattr(seg, 'width') and seg.width is not None else 0

        # Skip hairlines (width=0) - these are likely walls in CAD exports
        if width == 0:
            return False

        if width > self.max_width:
            return False

        # Criterion 2: Dash pattern (if required)
        if self.require_dashed:
            is_dashed = hasattr(seg, 'is_dashed') and seg.is_dashed
            if not is_dashed:
                return False

        # Criterion 3: Line length / span ratio
        length = self._get_segment_length(seg)

        # Check horizontal span (compare to page width)
        is_horizontal = self._is_horizontal(seg)
        is_vertical = self._is_vertical(seg)

        if not (is_horizontal or is_vertical):
            return False

        if is_horizontal:
            span_ratio = length / page_width if page_width > 0 else 0
        else:  # vertical
            span_ratio = length / page_height if page_height > 0 else 0

        if span_ratio < self.min_span_ratio:
            return False

        # Criterion 4: Edge proximity
        if not self._endpoints_near_edges(seg, page_width, page_height, is_horizontal):
            return False

        # All criteria met - this is a grid line
        return True

    def _is_horizontal(self, seg: Any) -> bool:
        """Check if segment is horizontal (angle ~0 degrees or ~180 degrees)."""
        angle = self._get_segment_angle(seg)
        return angle <= self.angle_tolerance or angle >= (180 - self.angle_tolerance)

    def _is_vertical(self, seg: Any) -> bool:
        """Check if segment is vertical (angle ~90 degrees)."""
        angle = self._get_segment_angle(seg)
        return abs(angle - 90) <= self.angle_tolerance

    def _endpoints_near_edges(
        self,
        seg: Any,
        page_width: float,
        page_height: float,
        is_horizontal: bool
    ) -> bool:
        """Check if segment endpoints are near page edges."""
        x1, y1 = seg.start
        x2, y2 = seg.end

        if is_horizontal:
            # For horizontal lines, check if X endpoints are near left/right edges
            left_near = min(x1, x2) <= self.edge_threshold
            right_near = max(x1, x2) >= (page_width - self.edge_threshold)
            return left_near and right_near
        else:
            # For vertical lines, check if Y endpoints are near top/bottom edges
            top_near = min(y1, y2) <= self.edge_threshold
            bottom_near = max(y1, y2) >= (page_height - self.edge_threshold)
            return top_near and bottom_near

    def _get_segment_angle(self, seg: Any) -> float:
        """Get segment angle in degrees (0-180)."""
        if hasattr(seg, 'angle'):
            return seg.angle
        dx = seg.end[0] - seg.start[0]
        dy = seg.end[1] - seg.start[1]
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0:
            angle += 180
        if angle >= 180:
            angle -= 180
        return angle

    def _get_segment_length(self, seg: Any) -> float:
        """Get segment length."""
        if hasattr(seg, 'length'):
            return seg.length
        dx = seg.end[0] - seg.start[0]
        dy = seg.end[1] - seg.start[1]
        return math.sqrt(dx * dx + dy * dy)


# =============================================================================
# HATCHING FILTER
# =============================================================================

class HatchingFilter:
    """
    Detects and removes cross-hatch and fill patterns.

    Hatching patterns are characterized by:
    - Groups of parallel lines at similar angles
    - Evenly spaced lines
    - Similar line lengths within a group

    Cross-hatching has two or more intersecting groups.
    """

    def __init__(
        self,
        angle_tolerance: float = HATCHING_ANGLE_TOLERANCE_DEG,
        min_group_size: int = HATCHING_MIN_GROUP_SIZE,
        max_spacing_ratio: float = HATCHING_MAX_SPACING_RATIO,
        length_variance_max: float = HATCHING_LENGTH_VARIANCE_MAX
    ):
        """
        Initialize the hatching filter.

        Args:
            angle_tolerance: Angle tolerance for parallel line grouping (degrees)
            min_group_size: Minimum lines to form a hatching group
            max_spacing_ratio: Maximum spacing between lines as ratio of length
            length_variance_max: Maximum coefficient of variation for lengths
        """
        self.angle_tolerance = angle_tolerance
        self.min_group_size = min_group_size
        self.max_spacing_ratio = max_spacing_ratio
        self.length_variance_max = length_variance_max

    def filter(
        self,
        segments: List[Any],
        page_width: float,
        page_height: float
    ) -> Tuple[List[Any], FilterStats]:
        """
        Remove hatching patterns from segments.

        Args:
            segments: List of Segment objects
            page_width: Page width in PDF points
            page_height: Page height in PDF points

        Returns:
            Tuple of (filtered_segments, stats)
        """
        if not segments:
            return [], FilterStats(reason="hatching")

        # Group segments by angle
        angle_groups = self._group_by_angle(segments)

        # Identify hatching groups
        hatching_indices: Set[int] = set()

        for angle, group_indices in angle_groups.items():
            if len(group_indices) < self.min_group_size:
                continue

            group_segments = [segments[i] for i in group_indices]

            # Check if this group has hatching characteristics
            if self._is_hatching_group(group_segments, page_width, page_height):
                hatching_indices.update(group_indices)
                logger.debug(
                    f"Hatching detected: {len(group_indices)} lines at ~{angle}°"
                )

        # Filter out hatching segments
        filtered = [
            seg for i, seg in enumerate(segments)
            if i not in hatching_indices
        ]

        stats = FilterStats(
            total_input=len(segments),
            total_output=len(filtered),
            removed=len(hatching_indices),
            reason="hatching"
        )

        if stats.removed > 0:
            logger.info(
                f"HatchingFilter: Removed {stats.removed} segments "
                f"({stats.removal_rate:.1%} of total)"
            )

        return filtered, stats

    def _group_by_angle(self, segments: List[Any]) -> Dict[int, List[int]]:
        """
        Group segment indices by their angle (quantized to tolerance).
        """
        groups = defaultdict(list)

        for i, seg in enumerate(segments):
            angle = self._get_segment_angle(seg)
            # Quantize angle to tolerance buckets
            bucket = int(angle / self.angle_tolerance) * int(self.angle_tolerance)
            groups[bucket].append(i)

        return groups

    def _get_segment_angle(self, seg: Any) -> float:
        """Get segment angle in degrees (0-180)."""
        if hasattr(seg, 'angle'):
            return seg.angle

        dx = seg.end[0] - seg.start[0]
        dy = seg.end[1] - seg.start[1]
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0:
            angle += 180
        if angle >= 180:
            angle -= 180
        return angle

    def _is_hatching_group(
        self,
        segments: List[Any],
        page_width: float,
        page_height: float
    ) -> bool:
        """
        Check if a group of parallel segments forms a hatching pattern.
        """
        if len(segments) < self.min_group_size:
            return False

        # Check length variance
        lengths = [self._get_segment_length(seg) for seg in segments]
        if len(set(lengths)) > 1:  # Not all same length
            mean_length = np.mean(lengths)
            std_length = np.std(lengths)
            cv = std_length / mean_length if mean_length > 0 else float('inf')

            # Hatching has consistent line lengths
            if cv > self.length_variance_max:
                return False

        # Check spacing consistency
        # Project segments onto perpendicular axis and check spacing
        avg_angle = np.mean([self._get_segment_angle(seg) for seg in segments])
        perp_angle = avg_angle + 90
        perp_rad = math.radians(perp_angle)

        # Project midpoints onto perpendicular axis
        projections = []
        for seg in segments:
            mid = self._get_midpoint(seg)
            proj = mid[0] * math.cos(perp_rad) + mid[1] * math.sin(perp_rad)
            projections.append(proj)

        projections.sort()

        # Calculate spacings
        spacings = [projections[i+1] - projections[i] for i in range(len(projections)-1)]

        if not spacings:
            return False

        # Check if spacings are consistent
        mean_spacing = np.mean(spacings)
        std_spacing = np.std(spacings)
        cv_spacing = std_spacing / mean_spacing if mean_spacing > 0 else float('inf')

        # Hatching has consistent spacing
        if cv_spacing > 0.5:  # Allow 50% variation in spacing
            return False

        # Check if spacing is small relative to line length
        mean_length = np.mean(lengths)
        if mean_length > 0 and mean_spacing / mean_length > self.max_spacing_ratio * 10:
            return False

        return True

    def _get_segment_length(self, seg: Any) -> float:
        """Get segment length."""
        if hasattr(seg, 'length'):
            return seg.length
        dx = seg.end[0] - seg.start[0]
        dy = seg.end[1] - seg.start[1]
        return math.sqrt(dx * dx + dy * dy)

    def _get_midpoint(self, seg: Any) -> Tuple[float, float]:
        """Get segment midpoint."""
        if hasattr(seg, 'midpoint'):
            return seg.midpoint
        return (
            (seg.start[0] + seg.end[0]) / 2,
            (seg.start[1] + seg.end[1]) / 2
        )


# =============================================================================
# DIMENSION LINE FILTER
# =============================================================================

class DimensionLineFilter:
    """
    Detects and removes dimension annotation lines.

    Dimension lines are characterized by:
    - Thin line width
    - Arrow heads at endpoints
    - Nearby dimension text
    - Typically horizontal or vertical
    """

    def __init__(
        self,
        max_line_width: float = DIMENSION_LINE_WIDTH_MAX,
        arrow_angle_range: Tuple[float, float] = (DIMENSION_ARROW_ANGLE_MIN, DIMENSION_ARROW_ANGLE_MAX),
        text_proximity: float = DIMENSION_TEXT_PROXIMITY_PX
    ):
        """
        Initialize the dimension line filter.

        Args:
            max_line_width: Maximum line width for dimension lines
            arrow_angle_range: (min, max) angle for arrow head detection
            text_proximity: Search radius for associated dimension text
        """
        self.max_line_width = max_line_width
        self.arrow_angle_range = arrow_angle_range
        self.text_proximity = text_proximity

    def filter(
        self,
        segments: List[Any],
        page_width: float,
        page_height: float,
        text_blocks: Optional[List[Any]] = None
    ) -> Tuple[List[Any], FilterStats]:
        """
        Remove dimension lines from segments.

        Args:
            segments: List of Segment objects
            page_width: Page width in PDF points
            page_height: Page height in PDF points
            text_blocks: Optional list of text blocks for dimension text matching

        Returns:
            Tuple of (filtered_segments, stats)
        """
        if not segments:
            return [], FilterStats(reason="dimension")

        # Build spatial index of segment endpoints for arrow detection
        endpoint_map = self._build_endpoint_map(segments)

        dimension_indices: Set[int] = set()

        for i, seg in enumerate(segments):
            # Skip if already marked
            if i in dimension_indices:
                continue

            # Check line width
            width = seg.width if hasattr(seg, 'width') and seg.width is not None else 1.0
            if width > self.max_line_width:
                continue

            # Check for arrow patterns at endpoints
            has_start_arrow = self._check_arrow_at_point(
                seg.start, seg, endpoint_map, segments
            )
            has_end_arrow = self._check_arrow_at_point(
                seg.end, seg, endpoint_map, segments
            )

            if has_start_arrow or has_end_arrow:
                dimension_indices.add(i)
                # Also mark the arrow segments
                for arrow_idx in self._get_arrow_segments(seg.start, endpoint_map, segments):
                    dimension_indices.add(arrow_idx)
                for arrow_idx in self._get_arrow_segments(seg.end, endpoint_map, segments):
                    dimension_indices.add(arrow_idx)

        # Filter out dimension segments
        filtered = [
            seg for i, seg in enumerate(segments)
            if i not in dimension_indices
        ]

        stats = FilterStats(
            total_input=len(segments),
            total_output=len(filtered),
            removed=len(dimension_indices),
            reason="dimension"
        )

        if stats.removed > 0:
            logger.info(
                f"DimensionLineFilter: Removed {stats.removed} segments "
                f"({stats.removal_rate:.1%} of total)"
            )

        return filtered, stats

    def _build_endpoint_map(
        self,
        segments: List[Any]
    ) -> Dict[Tuple[int, int], List[int]]:
        """
        Build a spatial index of segment endpoints.

        Returns:
            Dict mapping quantized (x, y) to list of segment indices
        """
        endpoint_map = defaultdict(list)
        quantize = 5  # Quantize to 5-point grid

        for i, seg in enumerate(segments):
            start_key = (int(seg.start[0] / quantize), int(seg.start[1] / quantize))
            end_key = (int(seg.end[0] / quantize), int(seg.end[1] / quantize))
            endpoint_map[start_key].append((i, 'start'))
            endpoint_map[end_key].append((i, 'end'))

        return endpoint_map

    def _check_arrow_at_point(
        self,
        point: Tuple[float, float],
        main_seg: Any,
        endpoint_map: Dict,
        segments: List[Any]
    ) -> bool:
        """
        Check if there's an arrow pattern at the given point.

        Arrow patterns have two short lines at angles 30-60° from the main line.
        """
        quantize = 5
        key = (int(point[0] / quantize), int(point[1] / quantize))

        # Get segments at this point
        nearby_segs = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                neighbor_key = (key[0] + dx, key[1] + dy)
                if neighbor_key in endpoint_map:
                    for idx, endpoint_type in endpoint_map[neighbor_key]:
                        if segments[idx] is not main_seg:
                            nearby_segs.append((idx, segments[idx]))

        if len(nearby_segs) < 2:
            return False

        # Get main segment angle
        main_angle = self._get_segment_angle(main_seg)

        # Check for arrow pattern (two lines at symmetric angles)
        arrow_angles = []
        for idx, seg in nearby_segs:
            seg_angle = self._get_segment_angle(seg)
            angle_diff = abs(seg_angle - main_angle)
            if angle_diff > 90:
                angle_diff = 180 - angle_diff

            if self.arrow_angle_range[0] <= angle_diff <= self.arrow_angle_range[1]:
                # Check if it's a short line (potential arrow head)
                length = self._get_segment_length(seg)
                main_length = self._get_segment_length(main_seg)
                if length < main_length * 0.3:  # Arrow heads are short
                    arrow_angles.append(angle_diff)

        # Arrow pattern: at least 2 lines at arrow angles
        return len(arrow_angles) >= 2

    def _get_arrow_segments(
        self,
        point: Tuple[float, float],
        endpoint_map: Dict,
        segments: List[Any]
    ) -> List[int]:
        """Get indices of potential arrow head segments at a point."""
        quantize = 5
        key = (int(point[0] / quantize), int(point[1] / quantize))

        indices = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                neighbor_key = (key[0] + dx, key[1] + dy)
                if neighbor_key in endpoint_map:
                    for idx, _ in endpoint_map[neighbor_key]:
                        # Check if it's a short line
                        seg = segments[idx]
                        length = self._get_segment_length(seg)
                        if length < 20:  # Short lines near dimension endpoints
                            indices.append(idx)

        return indices

    def _get_segment_angle(self, seg: Any) -> float:
        """Get segment angle in degrees (0-180)."""
        if hasattr(seg, 'angle'):
            return seg.angle
        dx = seg.end[0] - seg.start[0]
        dy = seg.end[1] - seg.start[1]
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0:
            angle += 180
        if angle >= 180:
            angle -= 180
        return angle

    def _get_segment_length(self, seg: Any) -> float:
        """Get segment length."""
        if hasattr(seg, 'length'):
            return seg.length
        dx = seg.end[0] - seg.start[0]
        dy = seg.end[1] - seg.start[1]
        return math.sqrt(dx * dx + dy * dy)


# =============================================================================
# ANNOTATION LINE FILTER
# =============================================================================

class AnnotationLineFilter:
    """
    Detects and removes thin annotation/markup lines.

    Annotation lines are characterized by:
    - Very thin line width
    - Often colored (non-black)
    - Connected paths forming symbols/marks
    """

    def __init__(
        self,
        max_line_width: float = ANNOTATION_LINE_WIDTH_MAX,
        min_connected_segments: int = ANNOTATION_MIN_SEGMENTS
    ):
        """
        Initialize the annotation filter.

        Args:
            max_line_width: Maximum line width for annotation lines
            min_connected_segments: Minimum connected segments to identify annotation path
        """
        self.max_line_width = max_line_width
        self.min_connected_segments = min_connected_segments

    def filter(
        self,
        segments: List[Any],
        page_width: float,
        page_height: float
    ) -> Tuple[List[Any], FilterStats]:
        """
        Remove annotation lines from segments.

        Args:
            segments: List of Segment objects
            page_width: Page width in PDF points
            page_height: Page height in PDF points

        Returns:
            Tuple of (filtered_segments, stats)
        """
        if not segments:
            return [], FilterStats(reason="annotation")

        annotation_indices: Set[int] = set()

        for i, seg in enumerate(segments):
            # Check line width
            width = seg.width if hasattr(seg, 'width') and seg.width is not None else 1.0
            if width <= self.max_line_width:
                # Check if it's colored (non-black)
                if hasattr(seg, 'color') and seg.color is not None:
                    # Non-black colored thin lines are likely annotations
                    if not self._is_black_color(seg.color):
                        annotation_indices.add(i)

        # Filter out annotation segments
        filtered = [
            seg for i, seg in enumerate(segments)
            if i not in annotation_indices
        ]

        stats = FilterStats(
            total_input=len(segments),
            total_output=len(filtered),
            removed=len(annotation_indices),
            reason="annotation"
        )

        if stats.removed > 0:
            logger.info(
                f"AnnotationLineFilter: Removed {stats.removed} segments "
                f"({stats.removal_rate:.1%} of total)"
            )

        return filtered, stats

    def _is_black_color(self, color: Tuple) -> bool:
        """Check if a color is black or near-black."""
        if color is None:
            return True  # Assume black if no color

        # Color can be grayscale (single value) or RGB tuple
        if isinstance(color, (int, float)):
            return color < 0.1
        elif len(color) == 1:
            return color[0] < 0.1
        elif len(color) >= 3:
            # RGB - check if all channels are low
            return all(c < 0.1 for c in color[:3])

        return True
