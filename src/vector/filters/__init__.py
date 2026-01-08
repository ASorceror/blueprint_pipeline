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
- FillExtractor: Extracts filled regions for construction phase detection
"""

from .drawing_area_filter import DrawingAreaFilter
from .segment_filters import (
    GridLineFilter,
    HatchingFilter,
    DimensionLineFilter,
    AnnotationLineFilter,
)
from .filter_pipeline import SegmentFilterPipeline, FilterConfig
from .fill_extractor import (
    FillExtractor,
    FilledRegion,
    FillExtractionResult,
    extract_filled_regions,
    classify_segment_by_region,
)
from .legend_detector import (
    LegendDetector,
    detect_legend,
    get_legend_or_defaults,
)
from .phase_classifier import (
    ConstructionPhaseClassifier,
    ClassificationResult,
    PhaseClassificationResult,
    classify_segments,
    classify_segments_simple,
)

__all__ = [
    'DrawingAreaFilter',
    'GridLineFilter',
    'HatchingFilter',
    'DimensionLineFilter',
    'AnnotationLineFilter',
    'SegmentFilterPipeline',
    'FilterConfig',
    'FillExtractor',
    'FilledRegion',
    'FillExtractionResult',
    'extract_filled_regions',
    'classify_segment_by_region',
    'LegendDetector',
    'detect_legend',
    'get_legend_or_defaults',
    'ConstructionPhaseClassifier',
    'ClassificationResult',
    'PhaseClassificationResult',
    'classify_segments',
    'classify_segments_simple',
]
