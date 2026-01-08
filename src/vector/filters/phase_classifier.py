"""
Construction Phase Classifier

Classifies wall segments by construction phase using multiple strategies:
1. Fill pattern from segment (primary for filled shapes)
2. Spatial region matching (segment falls within gray/hatched region)
3. Legend mapping (if legend was detected)
4. Industry defaults (fallback)

The classifier combines legend detection and fill extraction to provide
accurate phase classification for each wall segment.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

import pymupdf

from ..construction_phase import (
    ConstructionPhase,
    FillPattern,
    ClassificationMethod,
    LegendEntry,
    LegendDetectionResult,
    PhaseClassificationStats,
    classify_fill_color,
    get_default_legend,
)
from ..extractor import Segment
from .fill_extractor import (
    FillExtractor,
    FillExtractionResult,
    FilledRegion,
)
from .legend_detector import (
    LegendDetector,
    LegendDetectionResult,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Classification confidence thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.85
MEDIUM_CONFIDENCE_THRESHOLD = 0.70
LOW_CONFIDENCE_THRESHOLD = 0.50


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ClassificationResult:
    """
    Result from classifying a single segment.

    Attributes:
        phase: Classified construction phase
        confidence: Confidence in classification (0-1)
        method: Method used for classification
        matched_region: FilledRegion if spatial match, None otherwise
        matched_entry: LegendEntry if legend match, None otherwise
    """
    phase: ConstructionPhase
    confidence: float
    method: ClassificationMethod
    matched_region: Optional[FilledRegion] = None
    matched_entry: Optional[LegendEntry] = None


@dataclass
class PhaseClassificationResult:
    """
    Result from classifying all segments on a page.

    Attributes:
        segments: List of classified segments (with phase set)
        stats: Classification statistics
        legend_result: Legend detection result used
        fill_result: Fill extraction result used
    """
    segments: List[Segment] = field(default_factory=list)
    stats: PhaseClassificationStats = field(default_factory=PhaseClassificationStats)
    legend_result: Optional[LegendDetectionResult] = None
    fill_result: Optional[FillExtractionResult] = None

    def get_segments_by_phase(
        self,
        phase: ConstructionPhase
    ) -> List[Segment]:
        """Get all segments with a specific phase."""
        return [s for s in self.segments if s.construction_phase == phase.value]

    def get_new_segments(self) -> List[Segment]:
        """Get all NEW construction segments."""
        return self.get_segments_by_phase(ConstructionPhase.NEW)

    def get_existing_segments(self) -> List[Segment]:
        """Get all EXISTING segments."""
        return self.get_segments_by_phase(ConstructionPhase.EXISTING)

    def get_nic_segments(self) -> List[Segment]:
        """Get all NOT_IN_CONTRACT segments."""
        return self.get_segments_by_phase(ConstructionPhase.NOT_IN_CONTRACT)

    def summary(self) -> str:
        """Generate a summary string."""
        return self.stats.summary()


# =============================================================================
# PHASE CLASSIFIER CLASS
# =============================================================================

class ConstructionPhaseClassifier:
    """
    Classifies wall segments by construction phase.

    Combines multiple classification strategies:
    1. Segment fill pattern (for segments with fill info)
    2. Spatial region matching (for segments within filled regions)
    3. Legend mapping (using detected legend)
    4. Industry defaults (when no other method applies)

    Usage:
        classifier = ConstructionPhaseClassifier()
        result = classifier.classify_page(page, segments)

        for segment in result.segments:
            print(f"{segment.construction_phase}: confidence={segment.phase_confidence}")
    """

    def __init__(
        self,
        use_segment_fill: bool = True,
        use_spatial_regions: bool = True,
        use_legend: bool = True,
        use_industry_defaults: bool = True,
        fill_extractor: Optional[FillExtractor] = None,
        legend_detector: Optional[LegendDetector] = None,
    ):
        """
        Initialize the phase classifier.

        Args:
            use_segment_fill: Use fill info from segments
            use_spatial_regions: Use spatial region matching
            use_legend: Detect and use legend
            use_industry_defaults: Fall back to industry defaults
            fill_extractor: Custom fill extractor (or None for default)
            legend_detector: Custom legend detector (or None for default)
        """
        self.use_segment_fill = use_segment_fill
        self.use_spatial_regions = use_spatial_regions
        self.use_legend = use_legend
        self.use_industry_defaults = use_industry_defaults

        self._fill_extractor = fill_extractor or FillExtractor()
        self._legend_detector = legend_detector or LegendDetector()

    def classify_page(
        self,
        page: pymupdf.Page,
        segments: List[Segment],
    ) -> PhaseClassificationResult:
        """
        Classify all segments on a page.

        Args:
            page: PyMuPDF Page object
            segments: List of Segment objects to classify

        Returns:
            PhaseClassificationResult with classified segments
        """
        result = PhaseClassificationResult()
        result.stats = PhaseClassificationStats()

        if not segments:
            return result

        # Step 1: Detect legend
        legend_result = None
        if self.use_legend:
            legend_result = self._legend_detector.detect(page)
            result.legend_result = legend_result
            logger.debug(f"Legend detection: {legend_result.detection_method}, "
                        f"{len(legend_result.entries)} entries")

        # Step 2: Extract filled regions
        fill_result = None
        if self.use_spatial_regions:
            fill_result = self._fill_extractor.extract(page)
            result.fill_result = fill_result
            logger.debug(f"Fill extraction: {len(fill_result.gray_regions)} gray, "
                        f"{len(fill_result.hatched_regions)} hatched")

        # Step 3: Classify each segment
        for segment in segments:
            classification = self._classify_segment(
                segment,
                legend_result,
                fill_result,
            )

            # Apply classification to segment
            segment.set_phase(
                classification.phase.value,
                classification.confidence,
                classification.method.value,
            )

            # Update stats
            result.stats.add_classification(
                classification.phase,
                classification.confidence,
                classification.method,
            )

            result.segments.append(segment)

        logger.info(f"Phase classification: {len(segments)} segments, "
                   f"NEW={result.stats.new_count}, "
                   f"EXISTING={result.stats.existing_count}, "
                   f"N.I.C.={result.stats.nic_count}")

        return result

    def classify_segments(
        self,
        segments: List[Segment],
        legend_result: Optional[LegendDetectionResult] = None,
        fill_result: Optional[FillExtractionResult] = None,
    ) -> PhaseClassificationResult:
        """
        Classify segments using pre-computed legend and fill results.

        Args:
            segments: List of Segment objects
            legend_result: Pre-computed legend detection result
            fill_result: Pre-computed fill extraction result

        Returns:
            PhaseClassificationResult
        """
        result = PhaseClassificationResult()
        result.stats = PhaseClassificationStats()
        result.legend_result = legend_result
        result.fill_result = fill_result

        for segment in segments:
            classification = self._classify_segment(
                segment,
                legend_result,
                fill_result,
            )

            segment.set_phase(
                classification.phase.value,
                classification.confidence,
                classification.method.value,
            )

            result.stats.add_classification(
                classification.phase,
                classification.confidence,
                classification.method,
            )

            result.segments.append(segment)

        return result

    def _classify_segment(
        self,
        segment: Segment,
        legend_result: Optional[LegendDetectionResult],
        fill_result: Optional[FillExtractionResult],
    ) -> ClassificationResult:
        """
        Classify a single segment using available methods.

        Classification priority:
        1. Segment fill pattern (if segment has fill)
        2. Spatial region matching (if segment in gray/hatched region)
        3. Legend mapping (using fill to legend match)
        4. Industry defaults
        """
        # Method 1: Segment fill pattern
        if self.use_segment_fill and segment.fill is not None:
            result = self._classify_by_fill(segment)
            if result.confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
                return result

        # Method 2: Spatial region matching
        if self.use_spatial_regions and fill_result is not None:
            result = self._classify_by_region(segment, fill_result)
            if result.confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
                return result

        # Method 3: Default based on stroke-only (no fill = NEW)
        if segment.is_stroke_only:
            return ClassificationResult(
                phase=ConstructionPhase.NEW,
                confidence=0.75,
                method=ClassificationMethod.INDUSTRY_DEFAULT,
            )

        # Method 4: Industry defaults
        if self.use_industry_defaults:
            return ClassificationResult(
                phase=ConstructionPhase.NEW,
                confidence=0.60,
                method=ClassificationMethod.INDUSTRY_DEFAULT,
            )

        # Unknown
        return ClassificationResult(
            phase=ConstructionPhase.UNKNOWN,
            confidence=0.0,
            method=ClassificationMethod.UNKNOWN,
        )

    def _classify_by_fill(self, segment: Segment) -> ClassificationResult:
        """Classify segment by its fill pattern."""
        pattern, phase, confidence = classify_fill_color(
            segment.fill,
            segment.fill_type
        )

        return ClassificationResult(
            phase=phase,
            confidence=confidence,
            method=ClassificationMethod.FILL_PATTERN,
        )

    def _classify_by_region(
        self,
        segment: Segment,
        fill_result: FillExtractionResult,
    ) -> ClassificationResult:
        """Classify segment by spatial region containment."""
        mx, my = segment.midpoint

        # Check gray regions (EXISTING)
        for region in fill_result.gray_regions:
            if region.contains_point(mx, my):
                return ClassificationResult(
                    phase=ConstructionPhase.EXISTING,
                    confidence=region.confidence,
                    method=ClassificationMethod.SPATIAL_REGION,
                    matched_region=region,
                )

        # Check hatched regions (N.I.C.)
        for region in fill_result.hatched_regions:
            if region.contains_point(mx, my):
                return ClassificationResult(
                    phase=ConstructionPhase.NOT_IN_CONTRACT,
                    confidence=region.confidence,
                    method=ClassificationMethod.SPATIAL_REGION,
                    matched_region=region,
                )

        # Not in any region - low confidence
        return ClassificationResult(
            phase=ConstructionPhase.UNKNOWN,
            confidence=0.30,
            method=ClassificationMethod.SPATIAL_REGION,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def classify_segments(
    page: pymupdf.Page,
    segments: List[Segment],
) -> PhaseClassificationResult:
    """
    Convenience function to classify segments on a page.

    Args:
        page: PyMuPDF Page object
        segments: List of Segment objects

    Returns:
        PhaseClassificationResult
    """
    classifier = ConstructionPhaseClassifier()
    return classifier.classify_page(page, segments)


def classify_segments_simple(
    segments: List[Segment],
) -> List[Segment]:
    """
    Classify segments using industry defaults only.

    For use when no page context is available.

    Args:
        segments: List of Segment objects

    Returns:
        Same segments with phase info set
    """
    for segment in segments:
        if segment.is_gray_fill:
            segment.set_phase(
                ConstructionPhase.EXISTING.value,
                0.85,
                ClassificationMethod.FILL_PATTERN.value,
            )
        elif segment.is_stroke_only:
            segment.set_phase(
                ConstructionPhase.NEW.value,
                0.75,
                ClassificationMethod.INDUSTRY_DEFAULT.value,
            )
        else:
            segment.set_phase(
                ConstructionPhase.UNKNOWN.value,
                0.50,
                ClassificationMethod.UNKNOWN.value,
            )

    return segments
