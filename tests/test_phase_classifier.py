#!/usr/bin/env python
"""
Phase Classifier Tests

Tests for:
- ConstructionPhaseClassifier class
- Classification by fill pattern
- Classification by spatial region
- Classification result dataclasses
- Convenience functions
"""

import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pymupdf

from src.vector.filters.phase_classifier import (
    ConstructionPhaseClassifier,
    ClassificationResult,
    PhaseClassificationResult,
    classify_segments,
    classify_segments_simple,
)
from src.vector.filters.fill_extractor import (
    FillExtractionResult,
    FilledRegion,
)
from src.vector.construction_phase import (
    ConstructionPhase,
    FillPattern,
    ClassificationMethod,
    LegendDetectionResult,
    LegendEntry,
)
from src.vector.extractor import Segment


def create_test_pdf(filepath: str) -> None:
    """Create a test PDF with some shapes."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    shape = page.new_shape()

    # Gray filled rectangle (EXISTING)
    rect1 = pymupdf.Rect(100, 100, 300, 110)
    shape.draw_rect(rect1)
    shape.finish(color=(0, 0, 0), fill=(0.5, 0.5, 0.5), width=0.5)
    shape.commit()

    # Outline only (NEW)
    shape = page.new_shape()
    rect2 = pymupdf.Rect(100, 200, 300, 210)
    shape.draw_rect(rect2)
    shape.finish(color=(0, 0, 0), width=1.0)
    shape.commit()

    # Add legend text
    page.insert_text((450, 100), "LEGEND", fontsize=10)
    page.insert_text((450, 130), "NEW CONSTRUCTION", fontsize=8)
    page.insert_text((450, 150), "EXISTING CONSTRUCTION", fontsize=8)

    doc.save(filepath)
    doc.close()


class TestClassificationResult:
    """Tests for ClassificationResult dataclass."""

    def test_basic_creation(self):
        """Test creating a classification result."""
        result = ClassificationResult(
            phase=ConstructionPhase.NEW,
            confidence=0.85,
            method=ClassificationMethod.FILL_PATTERN,
        )
        assert result.phase == ConstructionPhase.NEW
        assert result.confidence == 0.85
        assert result.method == ClassificationMethod.FILL_PATTERN
        print("  [PASS] Basic classification result")

    def test_with_matched_region(self):
        """Test result with matched region."""
        region = FilledRegion(
            bbox=(100, 100, 200, 200),
            fill_pattern=FillPattern.SOLID_GRAY,
            phase=ConstructionPhase.EXISTING,
        )
        result = ClassificationResult(
            phase=ConstructionPhase.EXISTING,
            confidence=0.85,
            method=ClassificationMethod.SPATIAL_REGION,
            matched_region=region,
        )
        assert result.matched_region is not None
        print("  [PASS] Result with matched region")


class TestPhaseClassificationResult:
    """Tests for PhaseClassificationResult dataclass."""

    def test_empty_result(self):
        """Test empty result."""
        result = PhaseClassificationResult()
        assert len(result.segments) == 0
        assert result.stats.total_segments == 0
        print("  [PASS] Empty result")

    def test_get_segments_by_phase(self):
        """Test getting segments by phase."""
        # Create segments with phases set
        seg1 = Segment(start=(0, 0), end=(100, 0), width=1.0)
        seg1.set_phase("NEW", 0.85, "fill_pattern")

        seg2 = Segment(start=(0, 50), end=(100, 50), width=1.0)
        seg2.set_phase("EXISTING", 0.80, "spatial_region")

        seg3 = Segment(start=(0, 100), end=(100, 100), width=1.0)
        seg3.set_phase("NEW", 0.75, "industry_default")

        result = PhaseClassificationResult(segments=[seg1, seg2, seg3])

        new_segs = result.get_new_segments()
        existing_segs = result.get_existing_segments()

        assert len(new_segs) == 2
        assert len(existing_segs) == 1
        print("  [PASS] Get segments by phase")


class TestConstructionPhaseClassifier:
    """Tests for ConstructionPhaseClassifier class."""

    def test_classify_segment_with_gray_fill(self):
        """Test classifying segment with gray fill."""
        segment = Segment(
            start=(0, 0), end=(100, 0), width=1.0,
            fill=(0.5, 0.5, 0.5), fill_type='f'
        )

        classifier = ConstructionPhaseClassifier(
            use_spatial_regions=False,
            use_legend=False,
        )

        # Classify without page context
        result = classifier.classify_segments([segment])

        assert len(result.segments) == 1
        assert result.segments[0].construction_phase == "EXISTING"
        assert result.stats.existing_count == 1
        print("  [PASS] Classify segment with gray fill")

    def test_classify_segment_stroke_only(self):
        """Test classifying stroke-only segment."""
        segment = Segment(
            start=(0, 0), end=(100, 0), width=1.0,
            fill=None, fill_type='s'
        )

        classifier = ConstructionPhaseClassifier(
            use_spatial_regions=False,
            use_legend=False,
        )

        result = classifier.classify_segments([segment])

        assert result.segments[0].construction_phase == "NEW"
        assert result.stats.new_count == 1
        print("  [PASS] Classify stroke-only segment")

    def test_classify_by_spatial_region(self):
        """Test classifying segment by spatial region."""
        # Segment in a gray region
        segment = Segment(
            start=(120, 150), end=(180, 150), width=1.0,
            fill=None, fill_type='s'
        )

        # Create fill result with gray region containing the segment
        gray_region = FilledRegion(
            bbox=(100, 100, 200, 200),
            fill_pattern=FillPattern.SOLID_GRAY,
            phase=ConstructionPhase.EXISTING,
            confidence=0.85,
        )
        fill_result = FillExtractionResult(gray_regions=[gray_region])

        classifier = ConstructionPhaseClassifier(
            use_segment_fill=False,  # Disable fill check
            use_legend=False,
        )

        result = classifier.classify_segments(
            [segment],
            fill_result=fill_result,
        )

        assert result.segments[0].construction_phase == "EXISTING"
        assert result.segments[0].phase_method == "spatial_region"
        print("  [PASS] Classify by spatial region")

    def test_classify_page_integration(self):
        """Test full page classification."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            filepath = f.name

        try:
            create_test_pdf(filepath)
            doc = pymupdf.open(filepath)
            page = doc.load_page(0)

            # Create some test segments
            segments = [
                Segment(start=(150, 105), end=(250, 105), width=1.0),
                Segment(start=(150, 205), end=(250, 205), width=1.0),
            ]

            classifier = ConstructionPhaseClassifier()
            result = classifier.classify_page(page, segments)

            assert len(result.segments) == 2
            assert result.stats.total_segments == 2
            assert result.legend_result is not None
            assert result.fill_result is not None

            doc.close()
            print(f"  [PASS] Page classification integration")
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_custom_parameters(self):
        """Test classifier with custom parameters."""
        classifier = ConstructionPhaseClassifier(
            use_segment_fill=False,
            use_spatial_regions=False,
            use_legend=False,
            use_industry_defaults=True,
        )
        assert classifier.use_segment_fill is False
        assert classifier.use_spatial_regions is False
        print("  [PASS] Custom parameters")


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_classify_segments_simple(self):
        """Test simple classification without page context."""
        segments = [
            Segment(start=(0, 0), end=(100, 0), width=1.0,
                   fill=(0.5, 0.5, 0.5), fill_type='f'),  # Gray = EXISTING
            Segment(start=(0, 50), end=(100, 50), width=1.0,
                   fill=None, fill_type='s'),  # No fill = NEW
        ]

        result = classify_segments_simple(segments)

        assert result[0].construction_phase == "EXISTING"
        assert result[1].construction_phase == "NEW"
        print("  [PASS] classify_segments_simple")


def run_all_tests():
    """Run all phase classifier tests."""
    print("=" * 60)
    print("Phase Classifier Tests")
    print("=" * 60)
    print()

    all_passed = True

    # ClassificationResult Tests
    print("ClassificationResult Tests:")
    print("-" * 40)
    try:
        tests = TestClassificationResult()
        tests.test_basic_creation()
        tests.test_with_matched_region()
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        all_passed = False
    except Exception as e:
        print(f"  [ERROR] {e}")
        all_passed = False

    print()

    # PhaseClassificationResult Tests
    print("PhaseClassificationResult Tests:")
    print("-" * 40)
    try:
        tests = TestPhaseClassificationResult()
        tests.test_empty_result()
        tests.test_get_segments_by_phase()
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        all_passed = False
    except Exception as e:
        print(f"  [ERROR] {e}")
        all_passed = False

    print()

    # ConstructionPhaseClassifier Tests
    print("ConstructionPhaseClassifier Tests:")
    print("-" * 40)
    try:
        tests = TestConstructionPhaseClassifier()
        tests.test_classify_segment_with_gray_fill()
        tests.test_classify_segment_stroke_only()
        tests.test_classify_by_spatial_region()
        tests.test_classify_page_integration()
        tests.test_custom_parameters()
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        all_passed = False
    except Exception as e:
        print(f"  [ERROR] {e}")
        all_passed = False

    print()

    # Convenience Function Tests
    print("Convenience Function Tests:")
    print("-" * 40)
    try:
        tests = TestConvenienceFunctions()
        tests.test_classify_segments_simple()
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        all_passed = False
    except Exception as e:
        print(f"  [ERROR] {e}")
        all_passed = False

    print()
    print("=" * 60)
    if all_passed:
        print("ALL PHASE CLASSIFIER TESTS PASSED")
        return 0
    else:
        print("SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
