"""
Fill Pattern Extractor

Extracts filled regions from PyMuPDF drawings for construction phase detection.

This module detects:
- Gray-filled rectangles (EXISTING construction walls)
- Hatched regions (N.I.C. / NOT IN CONTRACT areas)
- Other fill patterns for phase classification

The extracted regions are used to classify wall segments spatially -
a wall segment that falls within a gray-filled region is EXISTING,
while one in a hatched region is NOT_IN_CONTRACT.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Set

import pymupdf

from ..construction_phase import (
    ConstructionPhase,
    FillPattern,
    is_gray_fill,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Gray fill detection (EXISTING construction)
GRAY_FILL_MIN = 0.25  # Minimum gray value (darker bound)
GRAY_FILL_MAX = 0.75  # Maximum gray value (lighter bound)
GRAY_TOLERANCE = 0.05  # Max RGB variance for grayscale check

# Minimum dimensions for wall-like shapes
MIN_WALL_ASPECT_RATIO = 3.0  # Length/width ratio for wall-like shapes
MIN_WALL_LENGTH_POINTS = 20.0  # Minimum length in PDF points
MIN_REGION_AREA_SQ_POINTS = 100.0  # Minimum area for regions

# Hatching detection
HATCHING_ANGLE_TOLERANCE = 5.0  # Degrees for parallel line grouping
HATCHING_MIN_LINES = 5  # Minimum lines to form a hatching pattern
HATCHING_DIAGONAL_ANGLES = [(40, 50), (130, 140)]  # Typical hatching angles
HATCHING_MAX_SPACING_POINTS = 20.0  # Maximum spacing between hatch lines
HATCHING_CLUSTER_TOLERANCE = 50.0  # Spatial clustering tolerance


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FilledRegion:
    """
    A filled region extracted from PDF drawings.

    Represents a contiguous area with a specific fill pattern,
    used for classifying wall segments by construction phase.

    Attributes:
        bbox: Bounding box (x0, y0, x1, y1) in PDF points
        fill_pattern: Type of fill (gray, hatched, etc.)
        phase: Inferred construction phase
        fill_color: RGB fill color if solid
        confidence: Confidence in classification
        area: Area in square PDF points
        source: How this region was detected
    """
    bbox: Tuple[float, float, float, float]
    fill_pattern: FillPattern
    phase: ConstructionPhase
    fill_color: Optional[Tuple[float, float, float]] = None
    confidence: float = 0.85
    area: float = 0.0
    source: str = "fill_detection"

    def contains_point(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Check if a point is inside this region."""
        x0, y0, x1, y1 = self.bbox
        return (x0 - margin) <= x <= (x1 + margin) and (y0 - margin) <= y <= (y1 + margin)

    def contains_segment_midpoint(self, segment: Any) -> bool:
        """Check if a segment's midpoint is inside this region."""
        mx, my = segment.midpoint
        return self.contains_point(mx, my)

    def overlaps(self, other: 'FilledRegion') -> bool:
        """Check if this region overlaps with another."""
        x0, y0, x1, y1 = self.bbox
        ox0, oy0, ox1, oy1 = other.bbox
        return not (x1 < ox0 or ox1 < x0 or y1 < oy0 or oy1 < y0)

    @property
    def width(self) -> float:
        """Width of the region."""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        """Height of the region."""
        return self.bbox[3] - self.bbox[1]

    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio (length/width) of the region."""
        w, h = self.width, self.height
        if w == 0 or h == 0:
            return 0.0
        return max(w, h) / min(w, h)

    @property
    def center(self) -> Tuple[float, float]:
        """Center point of the region."""
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2, (y0 + y1) / 2)


@dataclass
class FillExtractionResult:
    """
    Result from fill pattern extraction.

    Contains all detected filled regions organized by construction phase.

    Attributes:
        gray_regions: Regions with gray fill (EXISTING)
        hatched_regions: Regions with hatching (N.I.C.)
        other_regions: Other filled regions
        total_drawings: Total drawings processed
        detection_stats: Statistics about detection
    """
    gray_regions: List[FilledRegion] = field(default_factory=list)
    hatched_regions: List[FilledRegion] = field(default_factory=list)
    other_regions: List[FilledRegion] = field(default_factory=list)
    total_drawings: int = 0
    detection_stats: Dict[str, int] = field(default_factory=dict)

    @property
    def existing_regions(self) -> List[FilledRegion]:
        """Get all regions classified as EXISTING."""
        return self.gray_regions

    @property
    def nic_regions(self) -> List[FilledRegion]:
        """Get all regions classified as NOT_IN_CONTRACT."""
        return self.hatched_regions

    @property
    def all_regions(self) -> List[FilledRegion]:
        """Get all detected regions."""
        return self.gray_regions + self.hatched_regions + self.other_regions

    def get_phase_for_point(
        self,
        x: float,
        y: float
    ) -> Tuple[ConstructionPhase, float]:
        """
        Get the construction phase for a point based on region containment.

        Args:
            x: X coordinate in PDF points
            y: Y coordinate in PDF points

        Returns:
            Tuple of (phase, confidence)
        """
        # Check gray regions first (EXISTING)
        for region in self.gray_regions:
            if region.contains_point(x, y):
                return ConstructionPhase.EXISTING, region.confidence

        # Check hatched regions (N.I.C.)
        for region in self.hatched_regions:
            if region.contains_point(x, y):
                return ConstructionPhase.NOT_IN_CONTRACT, region.confidence

        # Default to NEW if not in any region
        return ConstructionPhase.NEW, 0.75

    def summary(self) -> str:
        """Generate a summary string."""
        lines = [
            f"Fill Extraction Results:",
            f"  Total drawings: {self.total_drawings}",
            f"  Gray regions (EXISTING): {len(self.gray_regions)}",
            f"  Hatched regions (N.I.C.): {len(self.hatched_regions)}",
            f"  Other regions: {len(self.other_regions)}",
        ]
        if self.detection_stats:
            lines.append("  Detection stats:")
            for key, value in self.detection_stats.items():
                lines.append(f"    {key}: {value}")
        return "\n".join(lines)


# =============================================================================
# FILL EXTRACTOR CLASS
# =============================================================================

class FillExtractor:
    """
    Extracts filled regions from PyMuPDF drawings.

    Detects gray-filled shapes (EXISTING construction) and hatched
    regions (N.I.C.) for construction phase classification.

    Usage:
        extractor = FillExtractor()
        result = extractor.extract(page)

        # Classify a segment
        phase, conf = result.get_phase_for_point(*segment.midpoint)
    """

    def __init__(
        self,
        gray_min: float = GRAY_FILL_MIN,
        gray_max: float = GRAY_FILL_MAX,
        min_wall_aspect: float = MIN_WALL_ASPECT_RATIO,
        min_wall_length: float = MIN_WALL_LENGTH_POINTS,
        min_region_area: float = MIN_REGION_AREA_SQ_POINTS,
        detect_hatching: bool = True,
    ):
        """
        Initialize the fill extractor.

        Args:
            gray_min: Minimum gray value for EXISTING detection
            gray_max: Maximum gray value for EXISTING detection
            min_wall_aspect: Minimum aspect ratio for wall-like shapes
            min_wall_length: Minimum length for wall shapes
            min_region_area: Minimum area for regions
            detect_hatching: Whether to detect hatching patterns
        """
        self.gray_min = gray_min
        self.gray_max = gray_max
        self.min_wall_aspect = min_wall_aspect
        self.min_wall_length = min_wall_length
        self.min_region_area = min_region_area
        self.detect_hatching = detect_hatching

    def extract(self, page: pymupdf.Page) -> FillExtractionResult:
        """
        Extract filled regions from a PDF page.

        Args:
            page: PyMuPDF Page object

        Returns:
            FillExtractionResult with detected regions
        """
        result = FillExtractionResult()

        try:
            drawings = page.get_drawings()
            result.total_drawings = len(drawings)
        except Exception as e:
            logger.warning(f"Error getting drawings: {e}")
            return result

        stats = {
            "total": len(drawings),
            "with_fill": 0,
            "gray_fills": 0,
            "wall_shapes": 0,
            "hatching_candidates": 0,
        }

        # Process drawings for gray fills
        for drawing in drawings:
            fill = drawing.get("fill")
            if fill is None:
                continue

            stats["with_fill"] += 1

            # Check for gray fill
            if self._is_gray_fill(fill):
                stats["gray_fills"] += 1
                region = self._extract_gray_region(drawing)
                if region:
                    stats["wall_shapes"] += 1
                    result.gray_regions.append(region)

        # Detect hatching patterns if enabled
        if self.detect_hatching:
            hatched = self._detect_hatching_regions(drawings, page)
            result.hatched_regions.extend(hatched)
            stats["hatching_candidates"] = len(hatched)

        result.detection_stats = stats
        logger.info(f"Fill extraction: {stats['gray_fills']} gray fills, "
                   f"{stats['wall_shapes']} wall shapes, "
                   f"{len(result.hatched_regions)} hatched regions")

        return result

    def _is_gray_fill(self, fill: Tuple[float, ...]) -> bool:
        """Check if a fill color is medium gray."""
        if fill is None or len(fill) < 3:
            if fill and len(fill) == 1:
                gray = fill[0]
                return self.gray_min <= gray <= self.gray_max
            return False

        r, g, b = fill[0], fill[1], fill[2]

        # Check if grayscale (R ≈ G ≈ B)
        max_diff = max(abs(r - g), abs(g - b), abs(r - b))
        if max_diff > GRAY_TOLERANCE:
            return False

        # Check gray value is in range
        gray = (r + g + b) / 3
        return self.gray_min <= gray <= self.gray_max

    def _extract_gray_region(self, drawing: Dict) -> Optional[FilledRegion]:
        """
        Extract a gray-filled region from a drawing.

        Args:
            drawing: PyMuPDF drawing dictionary

        Returns:
            FilledRegion if valid wall shape, None otherwise
        """
        fill = drawing.get("fill")
        rect = drawing.get("rect")

        if rect is None:
            # Try to compute bbox from items
            rect = self._compute_bbox_from_items(drawing.get("items", []))
            if rect is None:
                return None

        # Get bbox
        if hasattr(rect, 'x0'):
            bbox = (rect.x0, rect.y0, rect.x1, rect.y1)
        else:
            bbox = tuple(rect)

        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height

        # Filter by minimum area
        if area < self.min_region_area:
            return None

        # Check if wall-like shape (elongated rectangle)
        if width > 0 and height > 0:
            aspect = max(width, height) / min(width, height)
            length = max(width, height)

            # Wall shapes are elongated
            if aspect < self.min_wall_aspect or length < self.min_wall_length:
                return None

        # Extract fill color
        fill_color = None
        if fill and len(fill) >= 3:
            fill_color = (fill[0], fill[1], fill[2])

        return FilledRegion(
            bbox=bbox,
            fill_pattern=FillPattern.SOLID_GRAY,
            phase=ConstructionPhase.EXISTING,
            fill_color=fill_color,
            confidence=0.85,
            area=area,
            source="gray_fill_detection"
        )

    def _compute_bbox_from_items(
        self,
        items: List
    ) -> Optional[Tuple[float, float, float, float]]:
        """Compute bounding box from drawing items."""
        if not items:
            return None

        points = []
        for item in items:
            item_type = item[0]
            if item_type == "l":  # Line
                points.append((item[1].x, item[1].y))
                points.append((item[2].x, item[2].y))
            elif item_type == "m":  # Move
                points.append((item[1].x, item[1].y))
            elif item_type == "c":  # Curve
                for pt in item[1:]:
                    if hasattr(pt, 'x'):
                        points.append((pt.x, pt.y))
            elif item_type == "re":  # Rectangle
                rect = item[1]
                points.extend([
                    (rect.x0, rect.y0), (rect.x1, rect.y0),
                    (rect.x1, rect.y1), (rect.x0, rect.y1)
                ])

        if not points:
            return None

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (min(xs), min(ys), max(xs), max(ys))

    def _detect_hatching_regions(
        self,
        drawings: List[Dict],
        page: pymupdf.Page
    ) -> List[FilledRegion]:
        """
        Detect hatching patterns (parallel diagonal lines).

        Hatching indicates N.I.C. (Not In Contract) areas.

        Args:
            drawings: List of PyMuPDF drawing dictionaries
            page: Page object for dimensions

        Returns:
            List of FilledRegion objects for hatched areas
        """
        # Collect diagonal line segments
        diagonal_lines = []

        for drawing in drawings:
            fill = drawing.get("fill")
            # Hatching is typically stroke-only
            if fill is not None:
                continue

            items = drawing.get("items", [])
            for item in items:
                if item[0] != "l":  # Only lines
                    continue

                start = (item[1].x, item[1].y)
                end = (item[2].x, item[2].y)

                # Calculate angle
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                length = math.sqrt(dx * dx + dy * dy)

                if length < 5:  # Too short
                    continue

                angle = math.degrees(math.atan2(dy, dx))
                if angle < 0:
                    angle += 180

                # Check if diagonal (roughly 45 or 135 degrees)
                is_diagonal = False
                for min_a, max_a in HATCHING_DIAGONAL_ANGLES:
                    if min_a <= angle <= max_a:
                        is_diagonal = True
                        break

                if is_diagonal:
                    diagonal_lines.append({
                        "start": start,
                        "end": end,
                        "angle": angle,
                        "length": length,
                        "midpoint": ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
                    })

        if len(diagonal_lines) < HATCHING_MIN_LINES:
            return []

        # Cluster diagonal lines spatially
        regions = self._cluster_diagonal_lines(diagonal_lines)

        return regions

    def _cluster_diagonal_lines(
        self,
        lines: List[Dict]
    ) -> List[FilledRegion]:
        """
        Cluster diagonal lines into hatched regions.

        Groups nearby parallel lines into regions.

        Args:
            lines: List of diagonal line info dicts

        Returns:
            List of FilledRegion objects
        """
        if not lines:
            return []

        # Simple spatial clustering based on midpoints
        clusters: List[List[Dict]] = []
        used: Set[int] = set()

        for i, line in enumerate(lines):
            if i in used:
                continue

            cluster = [line]
            used.add(i)

            # Find nearby lines with similar angle
            for j, other in enumerate(lines):
                if j in used:
                    continue

                # Check angle similarity
                angle_diff = abs(line["angle"] - other["angle"])
                if angle_diff > HATCHING_ANGLE_TOLERANCE:
                    continue

                # Check spatial proximity
                dist = math.sqrt(
                    (line["midpoint"][0] - other["midpoint"][0]) ** 2 +
                    (line["midpoint"][1] - other["midpoint"][1]) ** 2
                )
                if dist < HATCHING_CLUSTER_TOLERANCE:
                    cluster.append(other)
                    used.add(j)

            if len(cluster) >= HATCHING_MIN_LINES:
                clusters.append(cluster)

        # Convert clusters to regions
        regions = []
        for cluster in clusters:
            # Compute bounding box of cluster
            all_points = []
            for line in cluster:
                all_points.append(line["start"])
                all_points.append(line["end"])

            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]
            bbox = (min(xs), min(ys), max(xs), max(ys))

            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height

            if area < self.min_region_area:
                continue

            region = FilledRegion(
                bbox=bbox,
                fill_pattern=FillPattern.HATCHED,
                phase=ConstructionPhase.NOT_IN_CONTRACT,
                fill_color=None,
                confidence=0.75,
                area=area,
                source="hatching_detection"
            )
            regions.append(region)

        logger.debug(f"Clustered {len(lines)} diagonal lines into {len(regions)} hatched regions")
        return regions


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def extract_filled_regions(page: pymupdf.Page) -> FillExtractionResult:
    """
    Convenience function to extract filled regions from a page.

    Args:
        page: PyMuPDF Page object

    Returns:
        FillExtractionResult with detected regions
    """
    extractor = FillExtractor()
    return extractor.extract(page)


def classify_segment_by_region(
    segment: Any,
    result: FillExtractionResult
) -> Tuple[ConstructionPhase, float, str]:
    """
    Classify a segment's construction phase based on region containment.

    Args:
        segment: Segment object with midpoint property
        result: FillExtractionResult from extraction

    Returns:
        Tuple of (phase, confidence, method)
    """
    mx, my = segment.midpoint
    phase, confidence = result.get_phase_for_point(mx, my)

    if phase == ConstructionPhase.EXISTING:
        method = "spatial_region"
    elif phase == ConstructionPhase.NOT_IN_CONTRACT:
        method = "spatial_region"
    else:
        method = "default"

    return phase, confidence, method
