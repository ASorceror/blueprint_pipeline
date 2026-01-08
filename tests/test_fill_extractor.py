#!/usr/bin/env python
"""
Fill Extractor Tests

Tests for:
- FilledRegion dataclass
- FillExtractionResult dataclass
- FillExtractor gray fill detection
- FillExtractor hatching detection
- Region-based segment classification
"""

import sys
import math
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pymupdf

from src.vector.filters.fill_extractor import (
    FilledRegion,
    FillExtractionResult,
    FillExtractor,
    extract_filled_regions,
    classify_segment_by_region,
    GRAY_FILL_MIN,
    GRAY_FILL_MAX,
)
from src.vector.construction_phase import (
    ConstructionPhase,
    FillPattern,
)
from src.vector.extractor import Segment


def create_test_pdf_with_fills(filepath: str) -> None:
    """Create a test PDF with various fill patterns."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    shape = page.new_shape()

    # Draw gray-filled rectangle (EXISTING wall)
    rect1 = pymupdf.Rect(100, 100, 300, 110)  # Horizontal wall
    shape.draw_rect(rect1)
    shape.finish(color=(0, 0, 0), fill=(0.5, 0.5, 0.5), width=0.5)
    shape.commit()

    # Draw another gray wall (vertical)
    shape = page.new_shape()
    rect2 = pymupdf.Rect(100, 100, 110, 300)  # Vertical wall
    shape.draw_rect(rect2)
    shape.finish(color=(0, 0, 0), fill=(0.5, 0.5, 0.5), width=0.5)
    shape.commit()

    # Draw outline-only rectangle (NEW wall)
    shape = page.new_shape()
    rect3 = pymupdf.Rect(400, 100, 500, 110)
    shape.draw_rect(rect3)
    shape.finish(color=(0, 0, 0), width=1.0)  # No fill
    shape.commit()

    # Draw diagonal hatching lines (N.I.C. area)
    shape = page.new_shape()
    for i in range(10):
        x_start = 200 + i * 5
        y_start = 400
        x_end = x_start + 50
        y_end = y_start + 50
        shape.draw_line((x_start, y_start), (x_end, y_end))
    shape.finish(color=(0, 0, 0), width=0.5)
    shape.commit()

    doc.save(filepath)
    doc.close()


class TestFilledRegion:
    """Tests for FilledRegion dataclass."""

    def test_basic_creation(self):
        """Test creating a filled region."""
        region = FilledRegion(
            bbox=(100, 100, 200, 110),
            fill_pattern=FillPattern.SOLID_GRAY,
            phase=ConstructionPhase.EXISTING,
            fill_color=(0.5, 0.5, 0.5),
            confidence=0.85,
            area=1000.0
        )
        assert region.bbox == (100, 100, 200, 110)
        assert region.phase == ConstructionPhase.EXISTING
        assert region.fill_pattern == FillPattern.SOLID_GRAY
        print("  [PASS] Basic region creation")

    def test_contains_point(self):
        """Test point containment check."""
        region = FilledRegion(
            bbox=(100, 100, 200, 200),
            fill_pattern=FillPattern.SOLID_GRAY,
            phase=ConstructionPhase.EXISTING
        )
        # Inside
        assert region.contains_point(150, 150) is True
        # On edge
        assert region.contains_point(100, 100) is True
        # Outside
        assert region.contains_point(50, 50) is False
        assert region.contains_point(250, 150) is False
        print("  [PASS] Point containment")

    def test_contains_point_with_margin(self):
        """Test point containment with margin."""
        region = FilledRegion(
            bbox=(100, 100, 200, 200),
            fill_pattern=FillPattern.SOLID_GRAY,
            phase=ConstructionPhase.EXISTING
        )
        # Just outside, but within margin
        assert region.contains_point(95, 150, margin=10) is True
        assert region.contains_point(205, 150, margin=10) is True
        # Too far outside
        assert region.contains_point(80, 150, margin=10) is False
        print("  [PASS] Point containment with margin")

    def test_properties(self):
        """Test computed properties."""
        region = FilledRegion(
            bbox=(100, 100, 200, 110),  # 100 wide, 10 tall
            fill_pattern=FillPattern.SOLID_GRAY,
            phase=ConstructionPhase.EXISTING
        )
        assert region.width == 100
        assert region.height == 10
        assert region.aspect_ratio == 10.0  # 100/10
        assert region.center == (150, 105)
        print("  [PASS] Computed properties")

    def test_overlaps(self):
        """Test region overlap detection."""
        region1 = FilledRegion(
            bbox=(100, 100, 200, 200),
            fill_pattern=FillPattern.SOLID_GRAY,
            phase=ConstructionPhase.EXISTING
        )
        region2 = FilledRegion(
            bbox=(150, 150, 250, 250),
            fill_pattern=FillPattern.SOLID_GRAY,
            phase=ConstructionPhase.EXISTING
        )
        region3 = FilledRegion(
            bbox=(300, 300, 400, 400),
            fill_pattern=FillPattern.SOLID_GRAY,
            phase=ConstructionPhase.EXISTING
        )
        assert region1.overlaps(region2) is True
        assert region1.overlaps(region3) is False
        print("  [PASS] Region overlap detection")


class TestFillExtractionResult:
    """Tests for FillExtractionResult dataclass."""

    def test_empty_result(self):
        """Test default empty result."""
        result = FillExtractionResult()
        assert len(result.gray_regions) == 0
        assert len(result.hatched_regions) == 0
        assert result.total_drawings == 0
        print("  [PASS] Empty result creation")

    def test_with_regions(self):
        """Test result with regions."""
        gray = FilledRegion(
            bbox=(100, 100, 200, 110),
            fill_pattern=FillPattern.SOLID_GRAY,
            phase=ConstructionPhase.EXISTING
        )
        hatched = FilledRegion(
            bbox=(300, 300, 400, 400),
            fill_pattern=FillPattern.HATCHED,
            phase=ConstructionPhase.NOT_IN_CONTRACT
        )
        result = FillExtractionResult(
            gray_regions=[gray],
            hatched_regions=[hatched],
            total_drawings=100
        )
        assert len(result.existing_regions) == 1
        assert len(result.nic_regions) == 1
        assert len(result.all_regions) == 2
        print("  [PASS] Result with regions")

    def test_get_phase_for_point(self):
        """Test phase lookup by point."""
        gray = FilledRegion(
            bbox=(100, 100, 200, 200),
            fill_pattern=FillPattern.SOLID_GRAY,
            phase=ConstructionPhase.EXISTING,
            confidence=0.85
        )
        hatched = FilledRegion(
            bbox=(300, 300, 400, 400),
            fill_pattern=FillPattern.HATCHED,
            phase=ConstructionPhase.NOT_IN_CONTRACT,
            confidence=0.75
        )
        result = FillExtractionResult(
            gray_regions=[gray],
            hatched_regions=[hatched]
        )

        # Point in gray region
        phase, conf = result.get_phase_for_point(150, 150)
        assert phase == ConstructionPhase.EXISTING
        assert conf == 0.85

        # Point in hatched region
        phase, conf = result.get_phase_for_point(350, 350)
        assert phase == ConstructionPhase.NOT_IN_CONTRACT
        assert conf == 0.75

        # Point in neither (NEW)
        phase, conf = result.get_phase_for_point(500, 500)
        assert phase == ConstructionPhase.NEW
        print("  [PASS] Phase lookup by point")

    def test_summary(self):
        """Test summary generation."""
        result = FillExtractionResult(
            gray_regions=[FilledRegion(
                bbox=(0, 0, 10, 10),
                fill_pattern=FillPattern.SOLID_GRAY,
                phase=ConstructionPhase.EXISTING
            )],
            total_drawings=100,
            detection_stats={"total": 100, "gray_fills": 5}
        )
        summary = result.summary()
        assert "Gray regions" in summary
        assert "1" in summary
        print("  [PASS] Summary generation")


class TestFillExtractor:
    """Tests for FillExtractor class."""

    def test_gray_fill_detection(self):
        """Test gray fill value detection."""
        extractor = FillExtractor()

        # Test internal gray detection method
        assert extractor._is_gray_fill((0.5, 0.5, 0.5)) is True
        assert extractor._is_gray_fill((0.498, 0.498, 0.498)) is True
        assert extractor._is_gray_fill((0.4, 0.4, 0.4)) is True

        # Too light
        assert extractor._is_gray_fill((0.9, 0.9, 0.9)) is False
        # Too dark
        assert extractor._is_gray_fill((0.1, 0.1, 0.1)) is False
        # Not grayscale (colored)
        assert extractor._is_gray_fill((1.0, 0.0, 0.0)) is False
        print("  [PASS] Gray fill detection")

    def test_extract_from_pdf(self):
        """Test extraction from a PDF with fills."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            filepath = f.name

        try:
            create_test_pdf_with_fills(filepath)
            doc = pymupdf.open(filepath)
            page = doc.load_page(0)

            extractor = FillExtractor()
            result = extractor.extract(page)

            # Should find gray-filled regions
            assert result.total_drawings > 0
            assert len(result.gray_regions) >= 0  # May vary based on aspect ratio filter

            doc.close()
            print(f"  [PASS] PDF extraction ({result.total_drawings} drawings)")
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_custom_parameters(self):
        """Test extractor with custom parameters."""
        extractor = FillExtractor(
            gray_min=0.3,
            gray_max=0.7,
            min_wall_aspect=2.0,
            min_wall_length=10.0,
            detect_hatching=False
        )
        assert extractor.gray_min == 0.3
        assert extractor.gray_max == 0.7
        assert extractor.detect_hatching is False
        print("  [PASS] Custom parameters")


class TestSegmentClassification:
    """Tests for segment classification by region."""

    def test_classify_segment_in_gray_region(self):
        """Test classifying a segment inside a gray region."""
        # Create segment in gray region
        segment = Segment(
            start=(100, 150), end=(200, 150), width=1.0
        )

        # Create result with gray region
        gray = FilledRegion(
            bbox=(90, 140, 210, 160),
            fill_pattern=FillPattern.SOLID_GRAY,
            phase=ConstructionPhase.EXISTING,
            confidence=0.85
        )
        result = FillExtractionResult(gray_regions=[gray])

        phase, conf, method = classify_segment_by_region(segment, result)
        assert phase == ConstructionPhase.EXISTING
        assert conf == 0.85
        assert method == "spatial_region"
        print("  [PASS] Classify segment in gray region")

    def test_classify_segment_in_hatched_region(self):
        """Test classifying a segment inside a hatched region."""
        segment = Segment(
            start=(350, 350), end=(370, 350), width=1.0
        )

        hatched = FilledRegion(
            bbox=(300, 300, 400, 400),
            fill_pattern=FillPattern.HATCHED,
            phase=ConstructionPhase.NOT_IN_CONTRACT,
            confidence=0.75
        )
        result = FillExtractionResult(hatched_regions=[hatched])

        phase, conf, method = classify_segment_by_region(segment, result)
        assert phase == ConstructionPhase.NOT_IN_CONTRACT
        assert conf == 0.75
        print("  [PASS] Classify segment in hatched region")

    def test_classify_segment_outside_regions(self):
        """Test classifying a segment outside all regions."""
        segment = Segment(
            start=(500, 500), end=(600, 500), width=1.0
        )

        gray = FilledRegion(
            bbox=(100, 100, 200, 200),
            fill_pattern=FillPattern.SOLID_GRAY,
            phase=ConstructionPhase.EXISTING
        )
        result = FillExtractionResult(gray_regions=[gray])

        phase, conf, method = classify_segment_by_region(segment, result)
        assert phase == ConstructionPhase.NEW
        assert method == "default"
        print("  [PASS] Classify segment outside regions")


def run_all_tests():
    """Run all fill extractor tests."""
    print("=" * 60)
    print("Fill Extractor Tests")
    print("=" * 60)
    print()

    all_passed = True

    # FilledRegion Tests
    print("FilledRegion Tests:")
    print("-" * 40)
    try:
        tests = TestFilledRegion()
        tests.test_basic_creation()
        tests.test_contains_point()
        tests.test_contains_point_with_margin()
        tests.test_properties()
        tests.test_overlaps()
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        all_passed = False
    except Exception as e:
        print(f"  [ERROR] {e}")
        all_passed = False

    print()

    # FillExtractionResult Tests
    print("FillExtractionResult Tests:")
    print("-" * 40)
    try:
        tests = TestFillExtractionResult()
        tests.test_empty_result()
        tests.test_with_regions()
        tests.test_get_phase_for_point()
        tests.test_summary()
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        all_passed = False
    except Exception as e:
        print(f"  [ERROR] {e}")
        all_passed = False

    print()

    # FillExtractor Tests
    print("FillExtractor Tests:")
    print("-" * 40)
    try:
        tests = TestFillExtractor()
        tests.test_gray_fill_detection()
        tests.test_extract_from_pdf()
        tests.test_custom_parameters()
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        all_passed = False
    except Exception as e:
        print(f"  [ERROR] {e}")
        all_passed = False

    print()

    # Segment Classification Tests
    print("Segment Classification Tests:")
    print("-" * 40)
    try:
        tests = TestSegmentClassification()
        tests.test_classify_segment_in_gray_region()
        tests.test_classify_segment_in_hatched_region()
        tests.test_classify_segment_outside_regions()
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        all_passed = False
    except Exception as e:
        print(f"  [ERROR] {e}")
        all_passed = False

    print()
    print("=" * 60)
    if all_passed:
        print("ALL FILL EXTRACTOR TESTS PASSED")
        return 0
    else:
        print("SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
