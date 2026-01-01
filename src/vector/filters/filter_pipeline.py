"""
Segment Filter Pipeline

Orchestrates all segment filters in a configurable pipeline:
1. DrawingAreaFilter - Excludes title blocks, legends, tables
2. GridLineFilter - Removes architectural grid lines (CRITICAL for room detection)
3. HatchingFilter - Removes cross-hatch patterns
4. DimensionLineFilter - Removes dimension annotations
5. AnnotationLineFilter - Removes thin markup lines

The pipeline can be configured to enable/disable individual filters
and provides detailed statistics on each filtering stage.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from PIL import Image

from .drawing_area_filter import DrawingAreaFilter, DrawingBounds
from .segment_filters import (
    GridLineFilter,
    HatchingFilter,
    DimensionLineFilter,
    AnnotationLineFilter,
    FilterStats
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    """Aggregate statistics from the filter pipeline."""
    total_input: int = 0
    total_output: int = 0
    stage_stats: Dict[str, FilterStats] = field(default_factory=dict)
    drawing_bounds: Optional[DrawingBounds] = None

    @property
    def total_removed(self) -> int:
        return self.total_input - self.total_output

    @property
    def total_removal_rate(self) -> float:
        if self.total_input == 0:
            return 0.0
        return self.total_removed / self.total_input

    def summary(self) -> str:
        """Generate a summary string of pipeline stats."""
        lines = [
            f"Pipeline Stats: {self.total_input} -> {self.total_output} segments "
            f"({self.total_removed} removed, {self.total_removal_rate:.1%})"
        ]

        for stage_name, stats in self.stage_stats.items():
            lines.append(
                f"  {stage_name}: -{stats.removed} ({stats.removal_rate:.1%})"
            )

        return "\n".join(lines)


@dataclass
class FilterConfig:
    """Configuration for the filter pipeline."""
    # Enable/disable individual filters
    enable_drawing_area: bool = True
    enable_grid_line: bool = True  # CRITICAL: Must be True for accurate room detection
    enable_hatching: bool = True
    enable_dimension: bool = True
    enable_annotation: bool = True

    # DrawingAreaFilter settings
    title_block_search_region: Tuple[float, float] = (0.60, 0.98)
    title_block_default_x1: float = 0.85
    title_block_margin: float = 0.01

    # GridLineFilter settings
    grid_line_max_width: float = 1.5
    grid_line_min_span_ratio: float = 0.7
    grid_line_edge_threshold: float = 100.0
    grid_line_require_dashed: bool = False

    # HatchingFilter settings
    hatching_angle_tolerance: float = 2.0
    hatching_min_group_size: int = 5

    # DimensionLineFilter settings
    dimension_max_width: float = 0.5

    # AnnotationLineFilter settings
    annotation_max_width: float = 0.3

    # Debug output
    debug_output_dir: Optional[Path] = None


class SegmentFilterPipeline:
    """
    Orchestrates segment filtering through multiple stages.

    Usage:
        # Initialize once per document
        pipeline = SegmentFilterPipeline(config)
        pipeline.initialize(sample_page_images)

        # Filter each page
        for page in pages:
            segments = extract_segments(page)
            filtered, stats = pipeline.filter(segments, page_width, page_height)
    """

    def __init__(self, config: Optional[FilterConfig] = None):
        """
        Initialize the pipeline.

        Args:
            config: Optional FilterConfig, uses defaults if not provided
        """
        self.config = config or FilterConfig()

        # Initialize filters
        self._drawing_area_filter: Optional[DrawingAreaFilter] = None
        self._grid_line_filter: Optional[GridLineFilter] = None
        self._hatching_filter: Optional[HatchingFilter] = None
        self._dimension_filter: Optional[DimensionLineFilter] = None
        self._annotation_filter: Optional[AnnotationLineFilter] = None

        self._initialized = False

        self._setup_filters()

    def _setup_filters(self) -> None:
        """Set up filter instances based on config."""
        if self.config.enable_drawing_area:
            self._drawing_area_filter = DrawingAreaFilter(
                search_region=self.config.title_block_search_region,
                default_x1=self.config.title_block_default_x1,
                margin=self.config.title_block_margin,
                output_dir=self.config.debug_output_dir
            )

        if self.config.enable_grid_line:
            self._grid_line_filter = GridLineFilter(
                max_width=self.config.grid_line_max_width,
                min_span_ratio=self.config.grid_line_min_span_ratio,
                edge_threshold=self.config.grid_line_edge_threshold,
                require_dashed=self.config.grid_line_require_dashed
            )

        if self.config.enable_hatching:
            self._hatching_filter = HatchingFilter(
                angle_tolerance=self.config.hatching_angle_tolerance,
                min_group_size=self.config.hatching_min_group_size
            )

        if self.config.enable_dimension:
            self._dimension_filter = DimensionLineFilter(
                max_line_width=self.config.dimension_max_width
            )

        if self.config.enable_annotation:
            self._annotation_filter = AnnotationLineFilter(
                max_line_width=self.config.annotation_max_width
            )

    def initialize(
        self,
        sample_page_images: List[Image.Image],
        strategy: str = 'balanced'
    ) -> DrawingBounds:
        """
        Initialize the pipeline with sample page images.

        This runs the title block detection to determine drawing area bounds.
        Should be called once per PDF before filtering pages.

        Args:
            sample_page_images: List of PIL Images from sample pages
            strategy: Detection strategy for title block

        Returns:
            DrawingBounds object
        """
        if self._drawing_area_filter is not None:
            bounds = self._drawing_area_filter.detect_bounds(
                sample_page_images,
                strategy=strategy
            )
        else:
            bounds = DrawingBounds()  # Default full-page bounds

        self._initialized = True
        return bounds

    def filter(
        self,
        segments: List[Any],
        page_width: float,
        page_height: float,
        text_blocks: Optional[List[Any]] = None
    ) -> Tuple[List[Any], PipelineStats]:
        """
        Run all enabled filters on segments.

        Args:
            segments: List of Segment objects
            page_width: Page width in PDF points
            page_height: Page height in PDF points
            text_blocks: Optional text blocks for dimension detection

        Returns:
            Tuple of (filtered_segments, pipeline_stats)
        """
        if not self._initialized:
            logger.warning(
                "Pipeline not initialized - call initialize() first. "
                "Using default bounds."
            )
            self.initialize([])

        stats = PipelineStats(total_input=len(segments))
        current_segments = segments

        # Stage 1: Drawing Area Filter
        if self._drawing_area_filter is not None:
            current_segments, removed = self._drawing_area_filter.filter_segments(
                current_segments, page_width, page_height
            )
            stats.stage_stats['drawing_area'] = FilterStats(
                total_input=len(segments),
                total_output=len(current_segments),
                removed=removed,
                reason='drawing_area'
            )
            stats.drawing_bounds = self._drawing_area_filter.get_bounds()

        # Stage 2: Grid Line Filter (CRITICAL - must run before other filters)
        if self._grid_line_filter is not None:
            input_count = len(current_segments)
            current_segments, stage_stats = self._grid_line_filter.filter(
                current_segments, page_width, page_height
            )
            stats.stage_stats['grid_line'] = stage_stats

        # Stage 3: Hatching Filter
        if self._hatching_filter is not None:
            input_count = len(current_segments)
            current_segments, stage_stats = self._hatching_filter.filter(
                current_segments, page_width, page_height
            )
            stats.stage_stats['hatching'] = stage_stats

        # Stage 4: Dimension Line Filter
        if self._dimension_filter is not None:
            input_count = len(current_segments)
            current_segments, stage_stats = self._dimension_filter.filter(
                current_segments, page_width, page_height, text_blocks
            )
            stats.stage_stats['dimension'] = stage_stats

        # Stage 5: Annotation Line Filter
        if self._annotation_filter is not None:
            input_count = len(current_segments)
            current_segments, stage_stats = self._annotation_filter.filter(
                current_segments, page_width, page_height
            )
            stats.stage_stats['annotation'] = stage_stats

        stats.total_output = len(current_segments)

        logger.info(stats.summary())

        return current_segments, stats

    def filter_polygons(
        self,
        polygons: List[Any],
        page_width: float,
        page_height: float
    ) -> Tuple[List[Any], int]:
        """
        Filter room polygons using the drawing area bounds.

        Args:
            polygons: List of RoomPolygon objects
            page_width: Page width in PDF points
            page_height: Page height in PDF points

        Returns:
            Tuple of (filtered_polygons, num_removed)
        """
        if self._drawing_area_filter is not None:
            return self._drawing_area_filter.filter_polygons(
                polygons, page_width, page_height
            )
        return polygons, 0

    def get_drawing_bounds(self) -> Optional[DrawingBounds]:
        """Get the detected drawing area bounds."""
        if self._drawing_area_filter is not None:
            return self._drawing_area_filter.get_bounds()
        return None

    def get_detection_result(self) -> Optional[Dict]:
        """Get the raw title block detection result."""
        if self._drawing_area_filter is not None:
            return self._drawing_area_filter.get_detection_result()
        return None

    def reset(self) -> None:
        """Reset the pipeline for a new PDF."""
        if self._drawing_area_filter is not None:
            self._drawing_area_filter.reset()
        self._initialized = False


def create_default_pipeline(
    debug_output_dir: Optional[Path] = None
) -> SegmentFilterPipeline:
    """
    Create a pipeline with default configuration.

    Args:
        debug_output_dir: Optional directory for debug output

    Returns:
        Configured SegmentFilterPipeline
    """
    config = FilterConfig(
        enable_drawing_area=True,
        enable_grid_line=True,  # CRITICAL: Must be enabled for accurate room detection
        enable_hatching=True,
        enable_dimension=True,
        enable_annotation=True,
        debug_output_dir=debug_output_dir
    )
    return SegmentFilterPipeline(config)


def create_minimal_pipeline() -> SegmentFilterPipeline:
    """
    Create a minimal pipeline with only drawing area filter.

    Use this for basic title block exclusion without other filters.
    """
    config = FilterConfig(
        enable_drawing_area=True,
        enable_hatching=False,
        enable_dimension=False,
        enable_annotation=False
    )
    return SegmentFilterPipeline(config)
