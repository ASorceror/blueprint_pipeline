"""
Vector Filters Module

Provides filtering capabilities for segment extraction:
- DrawingAreaFilter: Excludes title blocks, legends, tables from room detection
- GridLineFilter: Removes architectural grid lines (CRITICAL for room detection)
- HatchingFilter: Removes cross-hatch and fill patterns
- DimensionLineFilter: Removes dimension annotations
- AnnotationLineFilter: Removes thin annotation lines
- SegmentFilterPipeline: Orchestrates all filters
- FilterConfig: Configuration dataclass for the pipeline
"""

from .drawing_area_filter import DrawingAreaFilter
from .segment_filters import (
    GridLineFilter,
    HatchingFilter,
    DimensionLineFilter,
    AnnotationLineFilter,
)
from .filter_pipeline import SegmentFilterPipeline, FilterConfig

__all__ = [
    'DrawingAreaFilter',
    'GridLineFilter',
    'HatchingFilter',
    'DimensionLineFilter',
    'AnnotationLineFilter',
    'SegmentFilterPipeline',
    'FilterConfig',
]
