"""
Polygonizer Module

Bridges gaps at doorways and converts line segments to room polygons.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Optional

from shapely.geometry import LineString, Polygon, MultiPolygon, Point, GeometryCollection
from shapely.ops import polygonize_full, unary_union
from shapely.validation import make_valid

from ..constants import (
    DEFAULT_GAP_TOLERANCE_POINTS,
    MIN_GAP_TOLERANCE_POINTS,
    MAX_GAP_TOLERANCE_POINTS,
    BRIDGE_CROSS_SEGMENT_BUFFER,
    MIN_AREA_SQ_POINTS,
    MIN_VERTICES,
    MAX_AREA_PAGE_RATIO,
    Confidence,
)
from .extractor import Segment

logger = logging.getLogger(__name__)


@dataclass
class RoomPolygon:
    """A detected room polygon with metadata."""
    polygon_id: str
    vertices: List[Tuple[float, float]]
    area_sq_points: float
    shapely_polygon: Polygon
    source: str = "vector"
    confidence: str = Confidence.HIGH
    required_bridging: bool = False


def find_endpoint_neighbors(
    segments: List[Segment],
    tolerance: float = 2.0
) -> dict:
    """
    Build a map of which endpoints are connected to others.

    Args:
        segments: List of segments
        tolerance: Distance tolerance for considering points connected

    Returns:
        Dictionary mapping point tuple to count of connections
    """
    from collections import defaultdict

    # Round points to tolerance grid
    def round_point(p):
        return (round(p[0] / tolerance) * tolerance,
                round(p[1] / tolerance) * tolerance)

    connections = defaultdict(int)

    for seg in segments:
        start_key = round_point(seg.start)
        end_key = round_point(seg.end)
        connections[start_key] += 1
        connections[end_key] += 1

    return connections


def find_dangling_endpoints(segments: List[Segment]) -> List[Tuple[float, float]]:
    """
    Find all endpoints that don't connect to another segment.

    Args:
        segments: List of segments

    Returns:
        List of dangling endpoint coordinates
    """
    # Build connection count map
    connections = find_endpoint_neighbors(segments, tolerance=2.0)

    dangling = []

    for seg in segments:
        # Check start point
        start_key = (round(seg.start[0] / 2.0) * 2.0,
                     round(seg.start[1] / 2.0) * 2.0)
        if connections[start_key] == 1:
            dangling.append(seg.start)

        # Check end point
        end_key = (round(seg.end[0] / 2.0) * 2.0,
                   round(seg.end[1] / 2.0) * 2.0)
        if connections[end_key] == 1:
            dangling.append(seg.end)

    logger.debug(f"Found {len(dangling)} dangling endpoints")
    return dangling


def point_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate distance between two points."""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def bridge_crosses_segment(
    bridge_start: Tuple[float, float],
    bridge_end: Tuple[float, float],
    segments: List[Segment],
    buffer: float = BRIDGE_CROSS_SEGMENT_BUFFER
) -> bool:
    """
    Check if a bridge segment would cross any existing segment.

    Args:
        bridge_start: Start point of bridge
        bridge_end: End point of bridge
        segments: Existing segments to check against
        buffer: Buffer distance for crossing check

    Returns:
        True if bridge crosses an existing segment
    """
    bridge_line = LineString([bridge_start, bridge_end])

    for seg in segments:
        seg_line = LineString([seg.start, seg.end])

        # Check for intersection
        if bridge_line.crosses(seg_line):
            return True

        # Also check if bridge is too close to existing segment
        # (but not at the endpoints which should touch)
        if bridge_line.distance(seg_line) < buffer:
            # Check if it's just touching at endpoints (OK)
            bridge_mid = ((bridge_start[0] + bridge_end[0]) / 2,
                          (bridge_start[1] + bridge_end[1]) / 2)
            mid_point = Point(bridge_mid)
            if seg_line.distance(mid_point) < buffer:
                return True

    return False


def bridge_gaps(
    segments: List[Segment],
    gap_tolerance: float = DEFAULT_GAP_TOLERANCE_POINTS
) -> Tuple[List[Segment], int]:
    """
    Bridge gaps at doorways to create closed room polygons.

    Algorithm from spec 2.4:
    1. Extract all segment endpoints
    2. Find "dangling" endpoints (not connected to another segment)
    3. For each dangling endpoint, find nearby dangling endpoints
    4. If distance is within tolerance and bridge doesn't cross existing segments,
       add bridge segment

    Args:
        segments: List of wall segments
        gap_tolerance: Maximum gap distance to bridge (default from constants)

    Returns:
        Tuple of (segments with bridges, number of bridges added)
    """
    if not segments:
        return [], 0

    # Clamp tolerance to valid range
    gap_tolerance = max(MIN_GAP_TOLERANCE_POINTS, min(MAX_GAP_TOLERANCE_POINTS, gap_tolerance))

    # Find dangling endpoints
    dangling = find_dangling_endpoints(segments)

    if len(dangling) < 2:
        return segments, 0

    # Find valid bridges
    bridges = []
    used_points: Set[Tuple[float, float]] = set()

    # Round point for set membership
    def point_key(p):
        return (round(p[0], 1), round(p[1], 1))

    for i, p1 in enumerate(dangling):
        p1_key = point_key(p1)
        if p1_key in used_points:
            continue

        best_candidate = None
        best_distance = float('inf')

        for j, p2 in enumerate(dangling):
            if i == j:
                continue

            p2_key = point_key(p2)
            if p2_key in used_points:
                continue

            dist = point_distance(p1, p2)

            # Check distance bounds
            if dist > gap_tolerance:
                continue
            if dist < MIN_GAP_TOLERANCE_POINTS:
                continue  # Too close, probably same point

            # Check if this is better than current best
            if dist < best_distance:
                # Check if bridge would cross existing segments
                if not bridge_crosses_segment(p1, p2, segments):
                    best_candidate = (p2, dist)
                    best_distance = dist

        if best_candidate is not None:
            p2, dist = best_candidate
            # Create bridge segment
            bridge = Segment(start=p1, end=p2, width=0.5)
            bridges.append(bridge)
            used_points.add(p1_key)
            used_points.add(point_key(p2))

    # Combine original segments with bridges
    result = list(segments) + bridges

    logger.info(f"Gap bridging: added {len(bridges)} bridges")
    return result, len(bridges)


def segments_to_shapely_lines(segments: List[Segment]) -> List[LineString]:
    """
    Convert segments to Shapely LineString objects.

    Args:
        segments: List of Segment objects

    Returns:
        List of LineString objects
    """
    lines = []
    for seg in segments:
        line = LineString([seg.start, seg.end])
        if line.is_valid and line.length > 0:
            lines.append(line)
    return lines


def segments_to_polygons(
    segments: List[Segment],
    page_width: float,
    page_height: float
) -> Tuple[List[RoomPolygon], dict]:
    """
    Convert line segments to room polygons using Shapely polygonize.

    Algorithm from spec 2.5:
    1. Node the segments (find intersections, split at intersections)
    2. Apply Shapely polygonize_full
    3. Filter polygons

    Args:
        segments: List of wall segments (after gap bridging)
        page_width: Page width in points
        page_height: Page height in points

    Returns:
        Tuple of (list of RoomPolygon objects, debug info dict)
    """
    if not segments:
        return [], {"error": "no segments"}

    # Convert to Shapely lines
    lines = segments_to_shapely_lines(segments)

    if not lines:
        return [], {"error": "no valid lines"}

    # Union all lines to properly node them at intersections
    try:
        merged = unary_union(lines)
    except Exception as e:
        logger.warning(f"Error merging lines: {e}")
        merged = lines[0]
        for line in lines[1:]:
            try:
                merged = merged.union(line)
            except:
                pass

    # Polygonize
    try:
        result = polygonize_full(merged)
        polygons, dangles, cut_edges, invalid_rings = result
    except Exception as e:
        logger.warning(f"Polygonization failed: {e}")
        return [], {"error": str(e)}

    # Convert to list - handle various geometry types
    raw_polygons = []
    if isinstance(polygons, MultiPolygon):
        raw_polygons = list(polygons.geoms)
    elif isinstance(polygons, Polygon):
        raw_polygons = [polygons]
    elif isinstance(polygons, GeometryCollection):
        raw_polygons = [g for g in polygons.geoms if isinstance(g, Polygon)]
    elif polygons is not None:
        try:
            # Try iterating if it's a generator or other iterable
            for p in polygons:
                if isinstance(p, Polygon):
                    raw_polygons.append(p)
        except TypeError:
            pass

    def count_geoms(geom):
        """Count geometries in a geometry object."""
        if geom is None or geom.is_empty:
            return 0
        if hasattr(geom, 'geoms'):
            return len(list(geom.geoms))
        return 1

    debug_info = {
        "raw_polygon_count": len(raw_polygons),
        "dangles_count": count_geoms(dangles),
        "cut_edges_count": count_geoms(cut_edges),
        "invalid_rings_count": count_geoms(invalid_rings),
    }

    logger.debug(f"Polygonization: {debug_info}")

    # Filter polygons
    filtered = filter_room_polygons(raw_polygons, page_width, page_height)

    # Create RoomPolygon objects
    room_polygons = []
    for i, poly in enumerate(filtered):
        vertices = list(poly.exterior.coords)[:-1]  # Remove closing point
        room_poly = RoomPolygon(
            polygon_id=f"R{i+1:03d}",
            vertices=vertices,
            area_sq_points=poly.area,
            shapely_polygon=poly,
            source="vector",
            confidence=Confidence.HIGH,
        )
        room_polygons.append(room_poly)

    debug_info["filtered_count"] = len(room_polygons)
    return room_polygons, debug_info


def filter_room_polygons(
    polygons: List[Polygon],
    page_width: float,
    page_height: float
) -> List[Polygon]:
    """
    Filter polygons to keep only likely room polygons.

    Filter criteria from spec 2.6:
    1. Area >= MIN_AREA_SQ_POINTS
    2. Area <= page_area * MAX_AREA_PAGE_RATIO
    3. Vertex count >= MIN_VERTICES
    4. Polygon is valid (not self-intersecting)
    5. Holes not > 50% of polygon area

    Also:
    - If two polygons overlap > 90%, keep only smaller one
    - If polygon is entirely inside another, keep both (inner room)

    Args:
        polygons: List of raw Shapely polygons
        page_width: Page width in points
        page_height: Page height in points

    Returns:
        Filtered list of valid room polygons
    """
    page_area = page_width * page_height
    max_area = page_area * MAX_AREA_PAGE_RATIO

    valid_polygons = []

    for poly in polygons:
        # Make valid if needed
        if not poly.is_valid:
            try:
                poly = make_valid(poly)
            except:
                continue

        # Handle different geometry types from make_valid
        if isinstance(poly, MultiPolygon):
            # Take largest polygon
            poly = max(poly.geoms, key=lambda p: p.area if isinstance(p, Polygon) else 0)
        elif isinstance(poly, GeometryCollection):
            # Extract polygons from collection
            polys_in_collection = [g for g in poly.geoms if isinstance(g, Polygon)]
            if polys_in_collection:
                poly = max(polys_in_collection, key=lambda p: p.area)
            else:
                continue

        if not isinstance(poly, Polygon):
            continue

        # Filter 1: Minimum area
        if poly.area < MIN_AREA_SQ_POINTS:
            continue

        # Filter 2: Maximum area (page coverage)
        if poly.area > max_area:
            continue

        # Filter 3: Minimum vertices
        num_vertices = len(poly.exterior.coords) - 1  # -1 for closing point
        if num_vertices < MIN_VERTICES:
            continue

        # Filter 5: Holes not too large
        if poly.interiors:
            hole_area = sum(Polygon(hole).area for hole in poly.interiors)
            if hole_area > poly.area * 0.5:
                continue

        valid_polygons.append(poly)

    # Handle overlapping polygons
    final_polygons = []
    skip_indices: Set[int] = set()

    for i, poly_a in enumerate(valid_polygons):
        if i in skip_indices:
            continue

        should_keep = True

        for j, poly_b in enumerate(valid_polygons):
            if i == j or j in skip_indices:
                continue

            try:
                # Check overlap
                if poly_a.intersects(poly_b):
                    intersection = poly_a.intersection(poly_b)
                    overlap_ratio_a = intersection.area / poly_a.area if poly_a.area > 0 else 0
                    overlap_ratio_b = intersection.area / poly_b.area if poly_b.area > 0 else 0

                    # If >90% overlap, keep only smaller
                    if overlap_ratio_a > 0.9 and overlap_ratio_b > 0.9:
                        if poly_a.area > poly_b.area:
                            should_keep = False
                            break
                        else:
                            skip_indices.add(j)
            except:
                pass

        if should_keep:
            final_polygons.append(poly_a)

    logger.info(f"Polygon filter: {len(polygons)} -> {len(final_polygons)} room polygons")
    return final_polygons


def extract_room_polygons(
    segments: List[Segment],
    page_width: float,
    page_height: float,
    gap_tolerance: float = DEFAULT_GAP_TOLERANCE_POINTS
) -> Tuple[List[RoomPolygon], dict]:
    """
    Main entry point: extract room polygons from wall segments.

    Args:
        segments: List of wall segments
        page_width: Page width in points
        page_height: Page height in points
        gap_tolerance: Gap bridging tolerance in points

    Returns:
        Tuple of (list of RoomPolygon objects, debug info dict)
    """
    # Bridge gaps
    bridged_segments, bridge_count = bridge_gaps(segments, gap_tolerance)

    # Polygonize
    rooms, debug_info = segments_to_polygons(bridged_segments, page_width, page_height)

    # Mark rooms that required bridging
    if bridge_count > 0:
        for room in rooms:
            room.required_bridging = True
            room.confidence = Confidence.MEDIUM

    debug_info["bridges_added"] = bridge_count
    debug_info["input_segments"] = len(segments)
    debug_info["final_rooms"] = len(rooms)

    return rooms, debug_info
