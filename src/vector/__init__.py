# Vector extraction and polygonization module

from .extractor import (
    Segment,
    extract_all_paths,
    path_to_segments,
    paths_to_segments,
    filter_wall_segments,
    extract_wall_segments,
)

from .wall_merger import (
    MergedWall,
    detect_and_merge_double_walls,
    segments_are_parallel,
    calculate_centerline,
)

from .polygonizer import (
    RoomPolygon,
    bridge_gaps,
    segments_to_polygons,
    filter_room_polygons,
    extract_room_polygons,
)

__all__ = [
    # Extractor
    "Segment",
    "extract_all_paths",
    "path_to_segments",
    "paths_to_segments",
    "filter_wall_segments",
    "extract_wall_segments",
    # Wall Merger
    "MergedWall",
    "detect_and_merge_double_walls",
    "segments_are_parallel",
    "calculate_centerline",
    # Polygonizer
    "RoomPolygon",
    "bridge_gaps",
    "segments_to_polygons",
    "filter_room_polygons",
    "extract_room_polygons",
]
