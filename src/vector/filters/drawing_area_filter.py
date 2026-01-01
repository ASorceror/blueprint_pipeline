"""
Drawing Area Filter

Identifies and excludes non-drawing areas (title blocks, legends, tables)
from room detection. Uses the TitleBlockDetector to find the drawing boundary.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from PIL import Image

from .title_block_detector import TitleBlockDetector

logger = logging.getLogger(__name__)


@dataclass
class DrawingBounds:
    """Defines the bounds of the actual drawing area in a PDF page."""
    # Bounds as fractions of page dimensions (0.0-1.0)
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0

    # Confidence and metadata
    confidence: float = 1.0
    method: str = "default"

    # Detection details
    title_block_detected: bool = False
    title_block_x1: Optional[float] = None
    legend_detected: bool = False
    table_detected: bool = False

    def to_pixel_bounds(
        self,
        page_width: float,
        page_height: float
    ) -> Tuple[float, float, float, float]:
        """
        Convert fractional bounds to pixel coordinates.

        Returns:
            Tuple of (x_min_px, y_min_px, x_max_px, y_max_px)
        """
        return (
            self.x_min * page_width,
            self.y_min * page_height,
            self.x_max * page_width,
            self.y_max * page_height
        )

    def to_pdf_bounds(
        self,
        page_width: float,
        page_height: float
    ) -> Tuple[float, float, float, float]:
        """
        Convert fractional bounds to PDF point coordinates.

        Returns:
            Tuple of (x_min, y_min, x_max, y_max) in PDF points
        """
        return self.to_pixel_bounds(page_width, page_height)

    def contains_point(
        self,
        x: float,
        y: float,
        page_width: float,
        page_height: float
    ) -> bool:
        """
        Check if a point (in PDF coordinates) is within the drawing area.
        """
        x_frac = x / page_width
        y_frac = y / page_height

        return (self.x_min <= x_frac <= self.x_max and
                self.y_min <= y_frac <= self.y_max)

    def contains_segment(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        page_width: float,
        page_height: float,
        require_both_endpoints: bool = True
    ) -> bool:
        """
        Check if a segment is within the drawing area.

        Args:
            start: Start point (x, y) in PDF coordinates
            end: End point (x, y) in PDF coordinates
            page_width: Page width in PDF points
            page_height: Page height in PDF points
            require_both_endpoints: If True, both endpoints must be in bounds.
                                   If False, at least one endpoint must be in bounds.

        Returns:
            True if segment is (at least partially) in the drawing area
        """
        start_in = self.contains_point(start[0], start[1], page_width, page_height)
        end_in = self.contains_point(end[0], end[1], page_width, page_height)

        if require_both_endpoints:
            return start_in and end_in
        else:
            return start_in or end_in


class DrawingAreaFilter:
    """
    Filters segments to only include those within the actual drawing area.

    This filter uses the TitleBlockDetector to identify the title block
    boundary and excludes segments in that region.

    Usage:
        filter = DrawingAreaFilter()
        filter.detect_bounds(page_images)  # Run once per PDF
        filtered_segments = filter.filter_segments(segments, page_width, page_height)
    """

    def __init__(
        self,
        search_region: Tuple[float, float] = (0.60, 0.98),
        default_x1: float = 0.85,
        margin: float = 0.01,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize the filter.

        Args:
            search_region: Region to search for title block (start, end as fractions)
            default_x1: Fallback title block boundary if detection fails
            margin: Safety margin to add around detected boundary
            output_dir: Optional directory to save debug visualizations
        """
        self.search_region = search_region
        self.default_x1 = default_x1
        self.margin = margin
        self.output_dir = output_dir

        self._detector = TitleBlockDetector(
            search_region=search_region,
            default_x1=default_x1
        )
        self._bounds: Optional[DrawingBounds] = None
        self._detection_result: Optional[Dict] = None

    def detect_bounds(
        self,
        page_images: List[Image.Image],
        strategy: str = 'balanced'
    ) -> DrawingBounds:
        """
        Detect the drawing area bounds from sample page images.

        This should be called once per PDF before filtering segments.

        Args:
            page_images: List of PIL Images (sample pages from PDF)
            strategy: Detection strategy ('balanced', 'conservative', 'aggressive')

        Returns:
            DrawingBounds object defining the drawing area
        """
        if not page_images:
            logger.warning("No page images provided - using default bounds")
            self._bounds = DrawingBounds(
                x_max=self.default_x1,
                method='default_no_images'
            )
            return self._bounds

        # Run title block detection
        result = self._detector.detect(page_images, strategy=strategy)
        self._detection_result = result

        x1 = result['x1']
        confidence = result.get('confidence', 0.5)
        method = result.get('method', 'unknown')

        # Apply safety margin
        x_max = x1 - self.margin if x1 > self.margin else x1

        self._bounds = DrawingBounds(
            x_min=0.0,
            x_max=x_max,
            y_min=0.0,
            y_max=1.0,
            confidence=confidence,
            method=method,
            title_block_detected=True,
            title_block_x1=x1
        )

        logger.info(
            f"Drawing area bounds detected: x_max={x_max:.3f} "
            f"(title block at {x1:.3f}, method={method}, confidence={confidence:.2f})"
        )

        # Save debug visualization if output_dir specified
        if self.output_dir and page_images:
            self._save_debug_visualization(page_images[0], result)

        return self._bounds

    def get_bounds(self) -> Optional[DrawingBounds]:
        """Get the currently detected bounds, or None if not yet detected."""
        return self._bounds

    def filter_segments(
        self,
        segments: List[Any],
        page_width: float,
        page_height: float,
        require_both_endpoints: bool = True
    ) -> Tuple[List[Any], int]:
        """
        Filter segments to only include those in the drawing area.

        Args:
            segments: List of Segment objects (with start, end attributes)
            page_width: Page width in PDF points
            page_height: Page height in PDF points
            require_both_endpoints: If True, both endpoints must be in drawing area

        Returns:
            Tuple of (filtered_segments, num_removed)
        """
        if self._bounds is None:
            logger.warning("Bounds not detected - using default")
            self._bounds = DrawingBounds(
                x_max=self.default_x1,
                method='default_not_detected'
            )

        filtered = []
        removed = 0

        for seg in segments:
            if self._bounds.contains_segment(
                seg.start,
                seg.end,
                page_width,
                page_height,
                require_both_endpoints=require_both_endpoints
            ):
                filtered.append(seg)
            else:
                removed += 1

        if removed > 0:
            logger.debug(
                f"DrawingAreaFilter: Removed {removed} segments outside drawing area "
                f"(kept {len(filtered)} of {len(segments)})"
            )

        return filtered, removed

    def filter_polygons(
        self,
        polygons: List[Any],
        page_width: float,
        page_height: float,
        centroid_must_be_in: bool = True
    ) -> Tuple[List[Any], int]:
        """
        Filter room polygons to only include those in the drawing area.

        Args:
            polygons: List of RoomPolygon objects (with vertices or shapely_polygon)
            page_width: Page width in PDF points
            page_height: Page height in PDF points
            centroid_must_be_in: If True, polygon centroid must be in drawing area

        Returns:
            Tuple of (filtered_polygons, num_removed)
        """
        if self._bounds is None:
            self._bounds = DrawingBounds(
                x_max=self.default_x1,
                method='default_not_detected'
            )

        filtered = []
        removed = 0

        for poly in polygons:
            # Get centroid
            if hasattr(poly, 'shapely_polygon') and poly.shapely_polygon is not None:
                centroid = poly.shapely_polygon.centroid
                cx, cy = centroid.x, centroid.y
            elif hasattr(poly, 'vertices'):
                cx = sum(v[0] for v in poly.vertices) / len(poly.vertices)
                cy = sum(v[1] for v in poly.vertices) / len(poly.vertices)
            else:
                # Can't determine centroid, keep it
                filtered.append(poly)
                continue

            if centroid_must_be_in:
                if self._bounds.contains_point(cx, cy, page_width, page_height):
                    filtered.append(poly)
                else:
                    removed += 1
            else:
                # Check if any vertex is in bounds
                vertices = poly.vertices if hasattr(poly, 'vertices') else []
                any_in = any(
                    self._bounds.contains_point(v[0], v[1], page_width, page_height)
                    for v in vertices
                )
                if any_in:
                    filtered.append(poly)
                else:
                    removed += 1

        if removed > 0:
            logger.debug(
                f"DrawingAreaFilter: Removed {removed} polygons outside drawing area "
                f"(kept {len(filtered)} of {len(polygons)})"
            )

        return filtered, removed

    def _save_debug_visualization(
        self,
        page_image: Image.Image,
        result: Dict
    ) -> None:
        """Save debug visualization to output directory."""
        if not self.output_dir:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        viz = self._detector.visualize_detection(
            page_image,
            result,
            output_path=self.output_dir / "title_block_detection.png"
        )

        logger.debug(f"Saved title block detection visualization to {self.output_dir}")

    def get_detection_result(self) -> Optional[Dict]:
        """Get the raw detection result from TitleBlockDetector."""
        return self._detection_result

    def reset(self) -> None:
        """Reset the filter state for a new PDF."""
        self._bounds = None
        self._detection_result = None
        self._detector._learned_template = None
