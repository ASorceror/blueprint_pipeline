"""
Boundary Builder Module

Builds room boundaries starting from label centroids.
This is the core of the label-driven detection approach.

Strategies (in order of preference):
1. Closed Polygon - if walls form complete enclosure around label
2. Partial Boundary - if 60%+ of boundary has walls
3. Estimated Boundary - use nearby walls + Voronoi for missing
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict
from enum import Enum

import numpy as np
from shapely.geometry import Point, LineString, Polygon, box
from shapely.ops import polygonize, unary_union
from shapely.strtree import STRtree

from ..constants import POINTS_PER_INCH

logger = logging.getLogger(__name__)


class BoundaryMethod(Enum):
    """How the boundary was determined."""
    CLOSED_POLYGON = "closed_polygon"    # Complete enclosure from walls
    PARTIAL_WALLS = "partial_walls"      # 60%+ walls, rest estimated
    RAY_CAST = "ray_cast"                # Ray casting to walls
    VORONOI = "voronoi"                  # Voronoi partitioning
    ESTIMATED = "estimated"              # Heuristic estimation


@dataclass
class WallSegment:
    """A wall segment (line)."""
    x1: float
    y1: float
    x2: float
    y2: float
    weight: float = 1.0  # Line weight (thickness)

    @property
    def line(self) -> LineString:
        return LineString([(self.x1, self.y1), (self.x2, self.y2)])

    @property
    def length(self) -> float:
        return math.sqrt((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)

    @property
    def midpoint(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass
class RayHit:
    """Result of a ray cast."""
    direction: float           # Angle in radians
    distance: float            # Distance to hit
    hit_point: Tuple[float, float]
    hit_segment: Optional[WallSegment]
    is_wall: bool              # True if hit a wall segment
    is_boundary: bool          # True if hit page boundary


@dataclass
class BoundaryResult:
    """Result of boundary building."""
    polygon: Optional[Polygon]
    method: BoundaryMethod
    confidence: float                    # 0-1
    completeness: float                  # Fraction of boundary with walls
    wall_segments: List[WallSegment]     # Walls used in boundary
    ray_hits: List[RayHit]               # Ray cast results
    centroid: Tuple[float, float]        # Label centroid
    estimated_area_sqft: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


# Constants
DEFAULT_RAY_COUNT = 16        # Number of rays to cast
MIN_RAY_DISTANCE = 10.0       # Minimum ray length (points)
MAX_RAY_DISTANCE = 2000.0     # Maximum ray length (points)
MIN_COMPLETENESS = 0.60       # Minimum wall coverage for partial boundary
MIN_AREA_THRESHOLD = 100.0    # Minimum area in sq points
WALL_SNAP_TOLERANCE = 15.0    # Snap to walls within this distance


class BoundaryBuilder:
    """
    Build room boundaries starting from label positions.

    Uses ray casting from label centroid to find walls,
    then constructs polygon from intersection points.
    """

    def __init__(
        self,
        ray_count: int = DEFAULT_RAY_COUNT,
        max_ray_distance: float = MAX_RAY_DISTANCE,
        min_completeness: float = MIN_COMPLETENESS,
    ):
        """
        Initialize boundary builder.

        Args:
            ray_count: Number of rays to cast in all directions
            max_ray_distance: Maximum distance for rays
            min_completeness: Minimum wall coverage required
        """
        self.ray_count = ray_count
        self.max_ray_distance = max_ray_distance
        self.min_completeness = min_completeness

        # Precompute ray angles
        self.ray_angles = [
            2 * math.pi * i / ray_count
            for i in range(ray_count)
        ]

    def build_boundary(
        self,
        centroid: Tuple[float, float],
        segments: List[WallSegment],
        page_bounds: Tuple[float, float, float, float],
        neighbor_centroids: Optional[List[Tuple[float, float]]] = None
    ) -> BoundaryResult:
        """
        Build boundary around a label centroid.

        Args:
            centroid: (x, y) label centroid in PDF coordinates
            segments: List of wall segments
            page_bounds: (x0, y0, x1, y1) page boundaries
            neighbor_centroids: Centroids of neighboring labels

        Returns:
            BoundaryResult with polygon and metadata
        """
        x0, y0, x1, y1 = page_bounds
        cx, cy = centroid

        # Build spatial index for segments
        segment_lines = [s.line for s in segments]
        if segment_lines:
            tree = STRtree(segment_lines)
        else:
            tree = None

        # Create page boundary as polygon
        page_polygon = box(x0, y0, x1, y1)
        page_boundary = page_polygon.exterior

        # Cast rays in all directions
        ray_hits = []
        for angle in self.ray_angles:
            hit = self._cast_ray(
                centroid, angle, segments, tree,
                page_boundary, neighbor_centroids
            )
            ray_hits.append(hit)

        # Count wall hits
        wall_hits = [h for h in ray_hits if h.is_wall]
        completeness = len(wall_hits) / len(ray_hits)

        # Build polygon from hit points
        hit_points = [h.hit_point for h in ray_hits]

        # Close the polygon
        polygon = None
        try:
            if len(hit_points) >= 3:
                polygon = Polygon(hit_points)

                # Validate polygon
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)  # Fix self-intersections

                if polygon.is_empty or polygon.area < MIN_AREA_THRESHOLD:
                    polygon = None
        except Exception as e:
            logger.warning(f"Failed to create polygon: {e}")
            polygon = None

        # Determine method and confidence
        if polygon:
            if completeness >= 0.9:
                method = BoundaryMethod.CLOSED_POLYGON
                confidence = 0.95
            elif completeness >= self.min_completeness:
                method = BoundaryMethod.PARTIAL_WALLS
                confidence = 0.6 + (completeness - self.min_completeness) * 0.5
            else:
                method = BoundaryMethod.RAY_CAST
                confidence = 0.3 + completeness * 0.4
        else:
            method = BoundaryMethod.ESTIMATED
            confidence = 0.2

        # Collect wall segments used
        used_segments = [h.hit_segment for h in ray_hits if h.hit_segment]

        return BoundaryResult(
            polygon=polygon,
            method=method,
            confidence=confidence,
            completeness=completeness,
            wall_segments=used_segments,
            ray_hits=ray_hits,
            centroid=centroid,
        )

    def _cast_ray(
        self,
        origin: Tuple[float, float],
        angle: float,
        segments: List[WallSegment],
        tree: Optional[STRtree],
        page_boundary: LineString,
        neighbor_centroids: Optional[List[Tuple[float, float]]]
    ) -> RayHit:
        """
        Cast a ray from origin in direction of angle.

        Returns the first wall intersection, or page boundary if no wall.
        """
        ox, oy = origin

        # Ray endpoint (max distance)
        dx = math.cos(angle) * self.max_ray_distance
        dy = math.sin(angle) * self.max_ray_distance
        ray_end = (ox + dx, oy + dy)

        ray_line = LineString([origin, ray_end])

        # Find closest wall intersection
        closest_hit = None
        closest_distance = float('inf')
        closest_segment = None

        if tree is not None:
            # Query segments near the ray
            candidates = tree.query(ray_line)

            for idx in candidates:
                segment = segments[idx]
                seg_line = segment.line

                if ray_line.intersects(seg_line):
                    intersection = ray_line.intersection(seg_line)

                    if intersection.is_empty:
                        continue

                    # Get intersection point
                    if intersection.geom_type == 'Point':
                        ix, iy = intersection.x, intersection.y
                    elif intersection.geom_type == 'MultiPoint':
                        # Take closest point
                        points = list(intersection.geoms)
                        ix, iy = min(
                            [(p.x, p.y) for p in points],
                            key=lambda p: (p[0]-ox)**2 + (p[1]-oy)**2
                        )
                    else:
                        continue

                    distance = math.sqrt((ix - ox)**2 + (iy - oy)**2)

                    if distance > MIN_RAY_DISTANCE and distance < closest_distance:
                        closest_distance = distance
                        closest_hit = (ix, iy)
                        closest_segment = segment

        # If wall hit found
        if closest_hit:
            return RayHit(
                direction=angle,
                distance=closest_distance,
                hit_point=closest_hit,
                hit_segment=closest_segment,
                is_wall=True,
                is_boundary=False
            )

        # Check neighbor centroid midpoints (create virtual boundaries)
        if neighbor_centroids:
            for nx, ny in neighbor_centroids:
                # Midpoint between origin and neighbor
                mx, my = (ox + nx) / 2, (oy + ny) / 2

                # Check if ray passes near midpoint
                mid_point = Point(mx, my)
                dist_to_ray = ray_line.distance(mid_point)

                if dist_to_ray < 50:  # Within tolerance
                    distance = math.sqrt((mx - ox)**2 + (my - oy)**2)
                    if MIN_RAY_DISTANCE < distance < closest_distance:
                        closest_distance = distance
                        closest_hit = (mx, my)

        # Fall back to page boundary
        if not closest_hit:
            boundary_intersection = ray_line.intersection(page_boundary)

            if not boundary_intersection.is_empty:
                if boundary_intersection.geom_type == 'Point':
                    closest_hit = (boundary_intersection.x, boundary_intersection.y)
                elif boundary_intersection.geom_type == 'MultiPoint':
                    points = list(boundary_intersection.geoms)
                    closest_hit = min(
                        [(p.x, p.y) for p in points],
                        key=lambda p: (p[0]-ox)**2 + (p[1]-oy)**2
                    )

                if closest_hit:
                    closest_distance = math.sqrt(
                        (closest_hit[0] - ox)**2 + (closest_hit[1] - oy)**2
                    )

        # Last resort: use max distance
        if not closest_hit:
            closest_hit = ray_end
            closest_distance = self.max_ray_distance

        return RayHit(
            direction=angle,
            distance=closest_distance,
            hit_point=closest_hit,
            hit_segment=None,
            is_wall=False,
            is_boundary=True
        )

    def build_boundaries_for_labels(
        self,
        label_centroids: List[Tuple[str, Tuple[float, float]]],
        segments: List[WallSegment],
        page_bounds: Tuple[float, float, float, float]
    ) -> Dict[str, BoundaryResult]:
        """
        Build boundaries for multiple labels.

        Args:
            label_centroids: List of (label_text, (x, y)) tuples
            segments: List of wall segments
            page_bounds: Page boundaries

        Returns:
            Dict mapping label text to BoundaryResult
        """
        results = {}

        # Extract just centroids for neighbor calculation
        all_centroids = [c for _, c in label_centroids]

        for label_text, centroid in label_centroids:
            # Get neighbors (all centroids except this one)
            neighbors = [c for c in all_centroids if c != centroid]

            result = self.build_boundary(
                centroid=centroid,
                segments=segments,
                page_bounds=page_bounds,
                neighbor_centroids=neighbors
            )

            results[label_text] = result

        return results


def convert_segments_to_walls(
    raw_segments: List[Tuple[float, float, float, float, float]]
) -> List[WallSegment]:
    """
    Convert raw segment tuples to WallSegment objects.

    Args:
        raw_segments: List of (x1, y1, x2, y2, weight) tuples

    Returns:
        List of WallSegment objects
    """
    return [
        WallSegment(x1=s[0], y1=s[1], x2=s[2], y2=s[3], weight=s[4] if len(s) > 4 else 1.0)
        for s in raw_segments
    ]


def merge_overlapping_boundaries(
    boundaries: Dict[str, BoundaryResult],
    overlap_threshold: float = 0.3
) -> Dict[str, BoundaryResult]:
    """
    Handle overlapping room boundaries.

    If two boundaries overlap significantly, adjust them.

    Args:
        boundaries: Dict of label -> BoundaryResult
        overlap_threshold: Max allowed overlap ratio

    Returns:
        Adjusted boundaries
    """
    labels = list(boundaries.keys())

    for i, label1 in enumerate(labels):
        b1 = boundaries[label1]
        if not b1.polygon:
            continue

        for label2 in labels[i+1:]:
            b2 = boundaries[label2]
            if not b2.polygon:
                continue

            # Check overlap
            intersection = b1.polygon.intersection(b2.polygon)
            if intersection.is_empty:
                continue

            overlap_ratio = intersection.area / min(b1.polygon.area, b2.polygon.area)

            if overlap_ratio > overlap_threshold:
                # Significant overlap - split at midpoint
                # For now, just log warning
                b1.warnings.append(
                    f"Overlaps {overlap_ratio:.1%} with {label2}"
                )

    return boundaries

# =============================================================================
# POLYGON CONTAINMENT APPROACH (NEW - PREFERRED)
# =============================================================================
# Instead of ray-casting from labels, this approach:
# 1. First generates all closed polygons from wall segments
# 2. For each room label, finds which polygon contains the label centroid
# 3. This gives accurate room boundaries that follow actual walls

class PolygonContainmentBuilder:
    """
    Build room boundaries by finding which polygon contains each label.

    This is the preferred approach over ray-casting because:
    - Boundaries follow actual walls instead of arbitrary rays
    - More accurate room areas
    - Works correctly for irregular room shapes
    """

    def __init__(self):
        self.polygons: List[Polygon] = []
        self.polygon_tree: Optional[STRtree] = None

    def set_polygons(self, polygons: List[Polygon], min_area: float = 50000.0):
        """
        Set the available room polygons, filtering out small fragments.

        Args:
            polygons: List of Shapely Polygon objects from polygonizer
            min_area: Minimum area in PDF points squared (default 50000 = ~48 sq inches)
                      At 1/4" scale, this is ~190 sq ft minimum
        """
        # Filter out small polygons that are likely fragments, not rooms
        filtered = [p for p in polygons if p.area >= min_area]
        
        # Sort by area (largest first) so we find the largest containing polygon
        filtered.sort(key=lambda p: p.area, reverse=True)
        
        self.polygons = filtered
        if filtered:
            self.polygon_tree = STRtree(filtered)
            logger.info(f"PolygonContainmentBuilder: indexed {len(filtered)}/{len(polygons)} polygons (min_area={min_area})")
        else:
            self.polygon_tree = None
            logger.warning(f"PolygonContainmentBuilder: no polygons meet min_area={min_area}")

    def find_containing_polygon(
        self,
        centroid: Tuple[float, float]
    ) -> Optional[Polygon]:
        """
        Find the polygon that contains the given centroid.

        Args:
            centroid: (x, y) label centroid

        Returns:
            Polygon containing the centroid, or None if not found
        """
        if not self.polygon_tree:
            return None

        point = Point(centroid)

        # Query polygons near the point
        candidates = self.polygon_tree.query(point)

        for idx in candidates:
            polygon = self.polygons[idx]
            if polygon.contains(point):
                return polygon

        # If no exact containment, find polygon with centroid closest to point
        # This handles edge cases where label is on boundary
        min_distance = float('inf')
        closest_polygon = None

        for polygon in self.polygons:
            # Check if point is within polygon or very close to it
            if polygon.distance(point) < 5.0:  # Within 5 points
                poly_centroid = polygon.centroid
                distance = point.distance(poly_centroid)
                if distance < min_distance:
                    min_distance = distance
                    closest_polygon = polygon

        return closest_polygon

    def build_boundary(
        self,
        centroid: Tuple[float, float],
        label_text: str = ""
    ) -> BoundaryResult:
        """
        Build boundary for a label by finding containing polygon.

        Args:
            centroid: (x, y) label centroid
            label_text: Label text (for logging)

        Returns:
            BoundaryResult with polygon and metadata
        """
        polygon = self.find_containing_polygon(centroid)

        if polygon:
            # Success - found containing polygon
            logger.debug(f"Found polygon for '{label_text}' at {centroid}: area={polygon.area:.0f}")
            return BoundaryResult(
                polygon=polygon,
                method=BoundaryMethod.CLOSED_POLYGON,
                confidence=0.95,  # High confidence - actual wall boundary
                completeness=1.0,  # Complete boundary from walls
                wall_segments=[],  # Segments tracked elsewhere
                ray_hits=[],       # No ray casting
                centroid=centroid,
            )
        else:
            # Failed - no containing polygon
            logger.warning(f"No polygon found for '{label_text}' at {centroid}")
            return BoundaryResult(
                polygon=None,
                method=BoundaryMethod.ESTIMATED,
                confidence=0.0,
                completeness=0.0,
                wall_segments=[],
                ray_hits=[],
                centroid=centroid,
                warnings=["No containing polygon found"]
            )

    def build_boundaries_for_labels(
        self,
        label_centroids: List[Tuple[str, Tuple[float, float]]]
    ) -> Dict[str, BoundaryResult]:
        """
        Build boundaries for multiple labels.

        Args:
            label_centroids: List of (label_text, (x, y)) tuples

        Returns:
            Dict mapping label text to BoundaryResult
        """
        results = {}
        found_count = 0

        for label_text, centroid in label_centroids:
            result = self.build_boundary(centroid, label_text)
            results[label_text] = result
            if result.polygon:
                found_count += 1

        logger.info(f"PolygonContainment: found {found_count}/{len(label_centroids)} labels in polygons")
        return results


def build_boundaries_with_polygons(
    label_centroids: List[Tuple[str, Tuple[float, float]]],
    polygons: List[Polygon],
    segments: Optional[List[WallSegment]] = None,
    page_bounds: Optional[Tuple[float, float, float, float]] = None,
    fallback_to_raycasting: bool = True
) -> Dict[str, BoundaryResult]:
    """
    Build room boundaries using polygon containment with optional ray-casting fallback.

    This is the NEW preferred approach:
    1. Try to find containing polygon for each label
    2. If not found and fallback enabled, use ray-casting

    Args:
        label_centroids: List of (label_text, (x, y)) tuples
        polygons: List of closed polygons from polygonizer
        segments: Wall segments (for fallback ray-casting)
        page_bounds: Page boundaries (for fallback ray-casting)
        fallback_to_raycasting: If True, use ray-casting for labels not in polygons

    Returns:
        Dict mapping label text to BoundaryResult
    """
    results = {}

    # Try polygon containment first
    containment_builder = PolygonContainmentBuilder()
    containment_builder.set_polygons(polygons)

    polygon_results = containment_builder.build_boundaries_for_labels(label_centroids)

    # Collect labels that were not found in polygons
    missing_labels = []
    for label_text, result in polygon_results.items():
        if result.polygon:
            results[label_text] = result
        else:
            missing_labels.append((label_text, result.centroid))

    # Fallback to ray-casting for missing labels
    if fallback_to_raycasting and missing_labels and segments and page_bounds:
        logger.info(f"Falling back to ray-casting for {len(missing_labels)} labels")

        ray_builder = BoundaryBuilder()
        ray_results = ray_builder.build_boundaries_for_labels(
            missing_labels, segments, page_bounds
        )

        for label_text, result in ray_results.items():
            results[label_text] = result
    elif missing_labels:
        # No fallback - add failed results
        for label_text, centroid in missing_labels:
            results[label_text] = BoundaryResult(
                polygon=None,
                method=BoundaryMethod.ESTIMATED,
                confidence=0.0,
                completeness=0.0,
                wall_segments=[],
                ray_hits=[],
                centroid=centroid,
                warnings=["No containing polygon, fallback disabled"]
            )

    return results
