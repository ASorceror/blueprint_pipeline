"""
Hybrid Room Detector Module

Orchestrates label-driven room detection with geometry fallback.

Detection Strategy (in order):
1. Extract room labels from text
2. Build boundaries around labels using walls
3. Fall back to geometry-first for areas without labels
4. Use Voronoi partitioning as final fallback
"""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum

import numpy as np
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union

from ..text.room_label_extractor import (
    RoomLabelExtractor,
    ExtractedLabel,
    LabelCluster,
    LabelType,
)
from ..text.ocr_engine import TextBlock
from ..geometry.boundary_builder import (
    BoundaryBuilder,
    BoundaryResult,
    BoundaryMethod,
    WallSegment,
    convert_segments_to_walls,
    PolygonContainmentBuilder,
    build_boundaries_with_polygons,
)
from ..vector.polygonizer import extract_room_polygons
from ..vector.extractor import Segment
from ..geometry.room import Room
from ..constants import (
    Confidence,
    POINTS_PER_INCH,
)

logger = logging.getLogger(__name__)


class DetectionSource(Enum):
    """Source of room detection."""
    LABEL_BOUNDARY = "label_boundary"     # Label + boundary builder
    LABEL_VORONOI = "label_voronoi"       # Label + Voronoi partition
    GEOMETRY_ONLY = "geometry_only"       # Traditional geometry-first
    HYBRID = "hybrid"                     # Combination


@dataclass
class DetectedRoom:
    """A room detected by the hybrid detector."""
    room_id: str
    room_name: str
    polygon: Polygon
    centroid: Tuple[float, float]
    source: DetectionSource
    confidence: float
    label: Optional[ExtractedLabel] = None
    boundary_result: Optional[BoundaryResult] = None
    area_sqft: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class HybridDetectionResult:
    """Result from hybrid detection."""
    rooms: List[DetectedRoom]
    labels_found: int
    labels_with_boundaries: int
    geometry_fallback_rooms: int
    total_area_sqft: float
    warnings: List[str]
    stats: Dict


class HybridRoomDetector:
    """
    Detect rooms using label-driven approach with geometry fallback.

    This is the main orchestrator for the label-driven pipeline.
    """

    def __init__(
        self,
        drawing_bounds: Optional[Tuple[float, float, float, float]] = None,
        min_label_confidence: float = 0.5,
        min_boundary_completeness: float = 0.6,
        scale_factor: float = 1.0,
    ):
        """
        Initialize hybrid detector.

        Args:
            drawing_bounds: (x0, y0, x1, y1) of drawing area (excludes title block)
            min_label_confidence: Minimum confidence for labels
            min_boundary_completeness: Minimum wall coverage for boundaries
            scale_factor: Scale factor for area calculations
        """
        self.drawing_bounds = drawing_bounds
        self.scale_factor = scale_factor

        # Initialize sub-components
        self.label_extractor = RoomLabelExtractor(
            drawing_bounds=drawing_bounds,
            min_confidence=min_label_confidence
        )
        self.boundary_builder = BoundaryBuilder(
            min_completeness=min_boundary_completeness
        )

    def detect_rooms(
        self,
        text_blocks: List[TextBlock],
        wall_segments: List[Tuple[float, float, float, float, float]],
        page_width: float,
        page_height: float,
        geometry_polygons: Optional[List[Polygon]] = None
    ) -> HybridDetectionResult:
        """
        Detect rooms using hybrid approach.

        Args:
            text_blocks: Text blocks from OCR/embedded text
            wall_segments: Wall segments as (x1, y1, x2, y2, weight) tuples
            page_width: Page width in points
            page_height: Page height in points
            geometry_polygons: Optional pre-computed geometry polygons

        Returns:
            HybridDetectionResult with detected rooms
        """
        warnings = []
        detected_rooms = []

        # Phase 1: Extract room labels
        logger.info("Phase 1: Extracting room labels...")
        labels = self.label_extractor.extract(
            text_blocks, page_width, page_height
        )
        logger.info(f"Found {len(labels)} room labels")

        if not labels:
            warnings.append("No room labels found - falling back to geometry-only")
            # TODO: Implement geometry-only fallback
            return HybridDetectionResult(
                rooms=[],
                labels_found=0,
                labels_with_boundaries=0,
                geometry_fallback_rooms=0,
                total_area_sqft=0,
                warnings=warnings,
                stats={"method": "geometry_fallback"}
            )

        # Phase 2: Cluster labels (some rooms have number + name)
        logger.info("Phase 2: Clustering nearby labels...")
        clusters = self.label_extractor.cluster_labels(labels)
        logger.info(f"Created {len(clusters)} label clusters")

        # Phase 3: Convert segments to Segment objects and generate polygons
        logger.info("Phase 3: Processing wall segments and generating polygons...")
        
        # Convert raw segment tuples to Segment objects for polygonizer
        segments = [
            Segment(
                start=(s[0], s[1]),
                end=(s[2], s[3]),
                width=s[4] if len(s) > 4 else 1.0
            )
            for s in wall_segments
        ]
        
        # Also convert to WallSegment for fallback ray-casting
        walls = convert_segments_to_walls(wall_segments)
        logger.info(f"Converted {len(walls)} wall segments")
        
        # Generate polygons from wall segments using improved polygonizer
        # This uses shapely.node() to properly split lines at intersections
        room_polygons, poly_debug = extract_room_polygons(
            segments, page_width, page_height
        )
        logger.info(f"Generated {len(room_polygons)} room polygons from walls")
        
        # Extract Shapely Polygon objects
        shapely_polygons = [rp.shapely_polygon for rp in room_polygons]

        # Phase 4: Build boundaries around labels using ray-casting (proven approach)
        # Note: Polygon containment is available but ray-casting works better for 
        # architectural floor plans where closed polygons are often fragmented
        logger.info("Phase 4: Building room boundaries with ray-casting...")
        page_bounds = (0, 0, page_width, page_height)
        if self.drawing_bounds:
            # Use drawing bounds if available
            page_bounds = self.drawing_bounds

        label_centroids = [
            (cluster.primary_label.text, cluster.centroid)
            for cluster in clusters
        ]

        # Use ray-casting approach which expands outward from label to walls
        # This creates boundaries that follow the space around each label
        boundaries = self.boundary_builder.build_boundaries_for_labels(
            label_centroids=label_centroids,
            segments=walls,
            page_bounds=page_bounds
        )

        # Phase 5: Create detected rooms
        logger.info("Phase 5: Creating room objects...")
        rooms_with_boundaries = 0

        # Track rooms by name to handle duplicates
        seen_rooms: Dict[str, DetectedRoom] = {}

        for i, cluster in enumerate(clusters):
            label = cluster.primary_label
            label_text = label.text
            boundary = boundaries.get(label_text)

            if boundary and boundary.polygon:
                # Calculate area
                area_sqpts = boundary.polygon.area
                area_sqft = area_sqpts * (self.scale_factor ** 2) / (POINTS_PER_INCH ** 2)

                # Determine confidence
                if boundary.method == BoundaryMethod.CLOSED_POLYGON:
                    confidence = min(0.95, label.confidence * 1.1)
                elif boundary.method == BoundaryMethod.PARTIAL_WALLS:
                    confidence = min(0.85, label.confidence * boundary.confidence)
                else:
                    confidence = min(0.7, label.confidence * boundary.confidence)

                new_room = DetectedRoom(
                    room_id=f"room_{i:03d}",
                    room_name=label_text,
                    polygon=boundary.polygon,
                    centroid=boundary.centroid,
                    source=DetectionSource.LABEL_BOUNDARY,
                    confidence=confidence,
                    label=label,
                    boundary_result=boundary,
                    area_sqft=area_sqft,
                    warnings=boundary.warnings
                )

                # Handle duplicates - keep the one with higher confidence or larger area
                if label_text in seen_rooms:
                    existing = seen_rooms[label_text]
                    # Keep the larger area (likely the actual room, not a label duplicate)
                    if new_room.area_sqft > existing.area_sqft * 1.5:
                        # Significantly larger - replace
                        seen_rooms[label_text] = new_room
                        logger.debug(f"Replacing duplicate {label_text}: {existing.area_sqft:.0f} -> {new_room.area_sqft:.0f} SF")
                    elif new_room.area_sqft > existing.area_sqft:
                        # Slightly larger - keep with suffix
                        new_room.room_name = f"{label_text}_ALT"
                        seen_rooms[new_room.room_name] = new_room
                    else:
                        # Smaller - add with suffix
                        new_room.room_name = f"{label_text}_ALT"
                        seen_rooms[new_room.room_name] = new_room
                else:
                    seen_rooms[label_text] = new_room

                rooms_with_boundaries += 1
            else:
                # Failed to build boundary - use estimated area
                warnings.append(f"Could not build boundary for {label_text}")

        # Convert to list
        detected_rooms = list(seen_rooms.values())

        # Calculate totals
        total_area = sum(r.area_sqft or 0 for r in detected_rooms)

        # Stats
        stats = {
            "labels_extracted": len(labels),
            "clusters_formed": len(clusters),
            "walls_processed": len(walls),
            "boundaries_built": rooms_with_boundaries,
            "boundary_methods": self._count_methods(detected_rooms),
        }

        logger.info(f"Detection complete: {len(detected_rooms)} rooms, {total_area:.0f} SF")

        return HybridDetectionResult(
            rooms=detected_rooms,
            labels_found=len(labels),
            labels_with_boundaries=rooms_with_boundaries,
            geometry_fallback_rooms=0,
            total_area_sqft=total_area,
            warnings=warnings,
            stats=stats
        )

    def _count_methods(self, rooms: List[DetectedRoom]) -> Dict[str, int]:
        """Count rooms by boundary method."""
        counts = {}
        for room in rooms:
            if room.boundary_result:
                method = room.boundary_result.method.value
                counts[method] = counts.get(method, 0) + 1
        return counts

    def filter_grid_lines(self, labels: List[ExtractedLabel], page_height: float) -> List[ExtractedLabel]:
        """
        Filter out labels that are likely grid line markers.

        Grid lines typically:
        - Are at page edges
        - Are single digits or letters
        - Form regular patterns
        """
        filtered = []
        edge_threshold = 50  # Points from edge

        for label in labels:
            x, y = label.centroid

            # Check if at edge of page
            is_at_edge = (
                y < edge_threshold or
                y > page_height - edge_threshold
            )

            # Check if likely grid line (single digit or letter)
            is_grid_pattern = (
                len(label.text) <= 2 and
                (label.text.isdigit() or label.text.isalpha())
            )

            if is_at_edge and is_grid_pattern:
                logger.debug(f"Filtering grid line: {label.text} at ({x:.0f}, {y:.0f})")
                continue

            filtered.append(label)

        return filtered


def create_rooms_from_detection(
    detection_result: HybridDetectionResult,
    page_num: int,
    scale_factor: float,
    ceiling_height_ft: float = 10.0
) -> List[Room]:
    """
    Convert DetectedRooms to Room objects for output.

    Args:
        detection_result: Result from hybrid detection
        page_num: Page number
        scale_factor: Scale factor
        ceiling_height_ft: Default ceiling height

    Returns:
        List of Room objects
    """
    from ..geometry.calculator import create_room_from_polygon

    rooms = []

    for i, detected in enumerate(detection_result.rooms):
        room_id = f"room_{page_num:03d}_{i:03d}"

        # Get vertices from polygon
        vertices = list(detected.polygon.exterior.coords)[:-1]

        # Determine confidence level
        if detected.confidence >= 0.85:
            confidence = Confidence.HIGH
        elif detected.confidence >= 0.6:
            confidence = Confidence.MEDIUM
        else:
            confidence = Confidence.LOW

        room = create_room_from_polygon(
            polygon_id=room_id,
            polygon=detected.polygon,
            vertices=vertices,
            room_name=detected.room_name,
            sheet_number=page_num,
            scale_factor=scale_factor,
            ceiling_height_ft=ceiling_height_ft,
            source=detected.source.value,
            confidence=confidence
        )

        # Add detection metadata
        room.detection_method = detected.source.value
        if detected.boundary_result:
            room.boundary_completeness = detected.boundary_result.completeness

        rooms.append(room)

    return rooms
