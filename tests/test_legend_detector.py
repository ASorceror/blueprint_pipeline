#!/usr/bin/env python
"""
Legend Detector Tests

Tests for:
- LegendDetector class
- Text block extraction
- Legend keyword search
- Phase pattern matching
- Industry default fallback
"""

import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pymupdf

from src.vector.filters.legend_detector import (
    LegendDetector,
    TextBlock,
    LegendSearchResult,
    detect_legend,
    get_legend_or_defaults,
    LEGEND_KEYWORDS,
    PHASE_PATTERNS,
)
from src.vector.construction_phase import (
    ConstructionPhase,
    FillPattern,
    LegendDetectionResult,
)


def create_test_pdf_with_legend(filepath: str) -> None:
    """Create a test PDF with a construction plan legend."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)

    # Add legend title (in right portion of page)
    page.insert_text((450, 100), "LEGEND - CONSTRUCTION PLAN", fontsize=10)

    # Add legend entries
    page.insert_text((450, 130), "NEW CONSTRUCTION", fontsize=8)
    page.insert_text((450, 150), "EXISTING CONSTRUCTION", fontsize=8)
    page.insert_text((450, 170), "AREA NOT IN CONTRACT", fontsize=8)
    page.insert_text((450, 190), "DEMOLITION", fontsize=8)

    # Add some other text elsewhere
    page.insert_text((50, 300), "FLOOR PLAN - LEVEL 1", fontsize=12)
    page.insert_text((50, 350), "ROOM 101", fontsize=8)

    doc.save(filepath)
    doc.close()


def create_test_pdf_without_legend(filepath: str) -> None:
    """Create a test PDF without a legend."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)

    # Just floor plan text, no legend
    page.insert_text((50, 100), "FLOOR PLAN - LEVEL 1", fontsize=12)
    page.insert_text((50, 150), "ROOM 101", fontsize=8)
    page.insert_text((50, 200), "ROOM 102", fontsize=8)

    doc.save(filepath)
    doc.close()


class TestTextBlock:
    """Tests for TextBlock dataclass."""

    def test_basic_creation(self):
        """Test creating a text block."""
        block = TextBlock(
            text="NEW CONSTRUCTION",
            bbox=(100, 100, 200, 120),
            font_size=10.0
        )
        assert block.text == "NEW CONSTRUCTION"
        assert block.x0 == 100
        assert block.y0 == 100
        assert block.x1 == 200
        assert block.y1 == 120
        print("  [PASS] Basic text block creation")

    def test_center_property(self):
        """Test center calculation."""
        block = TextBlock(
            text="TEST",
            bbox=(100, 100, 200, 200),
            font_size=10.0
        )
        assert block.center == (150, 150)
        print("  [PASS] Center property")


class TestLegendDetector:
    """Tests for LegendDetector class."""

    def test_detect_with_legend(self):
        """Test detection on PDF with legend."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            filepath = f.name

        try:
            create_test_pdf_with_legend(filepath)
            doc = pymupdf.open(filepath)
            page = doc.load_page(0)

            detector = LegendDetector()
            result = detector.detect(page)

            assert result.has_legend is True
            assert result.detection_method == "keyword_search"
            assert len(result.entries) >= 2  # At least NEW and EXISTING

            # Check that expected phases were found
            phases = [e.phase for e in result.entries]
            assert ConstructionPhase.NEW in phases
            assert ConstructionPhase.EXISTING in phases

            doc.close()
            print(f"  [PASS] Detect with legend ({len(result.entries)} entries)")
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_detect_without_legend_uses_defaults(self):
        """Test detection falls back to defaults when no legend."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            filepath = f.name

        try:
            create_test_pdf_without_legend(filepath)
            doc = pymupdf.open(filepath)
            page = doc.load_page(0)

            detector = LegendDetector(use_industry_defaults=True)
            result = detector.detect(page)

            # Should get industry defaults
            assert result.detection_method == "industry_default"
            assert len(result.entries) >= 4  # Default has 4 entries

            doc.close()
            print("  [PASS] Detect without legend uses defaults")
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_detect_without_legend_no_defaults(self):
        """Test detection returns empty when no legend and defaults disabled."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            filepath = f.name

        try:
            create_test_pdf_without_legend(filepath)
            doc = pymupdf.open(filepath)
            page = doc.load_page(0)

            detector = LegendDetector(use_industry_defaults=False)
            result = detector.detect(page)

            assert result.has_legend is False
            assert len(result.entries) == 0

            doc.close()
            print("  [PASS] Detect without legend, no defaults")
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_custom_parameters(self):
        """Test detector with custom parameters."""
        detector = LegendDetector(
            search_title_block=False,
            title_block_x_ratio=0.5,
            search_margin=100,
            use_industry_defaults=False
        )
        assert detector.search_title_block is False
        assert detector.title_block_x_ratio == 0.5
        assert detector.search_margin == 100
        print("  [PASS] Custom parameters")


class TestPhasePatterns:
    """Tests for phase pattern matching."""

    def test_new_patterns(self):
        """Test NEW phase pattern matching."""
        import re
        patterns = [re.compile(p, re.IGNORECASE) for p in PHASE_PATTERNS[ConstructionPhase.NEW]]

        test_strings = [
            "NEW CONSTRUCTION",
            "NEW WALLS",
            "NEW WORK",
            "PROPOSED CONSTRUCTION",
            "NEW INTERIOR WALLS",
        ]

        for s in test_strings:
            matched = any(p.search(s) for p in patterns)
            assert matched, f"Should match NEW: {s}"

        print("  [PASS] NEW phase patterns")

    def test_existing_patterns(self):
        """Test EXISTING phase pattern matching."""
        import re
        patterns = [re.compile(p, re.IGNORECASE) for p in PHASE_PATTERNS[ConstructionPhase.EXISTING]]

        test_strings = [
            "EXISTING CONSTRUCTION",
            "EXISTING WALLS",
            "EXISTING TO REMAIN",
            "WALLS TO REMAIN",
            "EXISTING BUILDING",
        ]

        for s in test_strings:
            matched = any(p.search(s) for p in patterns)
            assert matched, f"Should match EXISTING: {s}"

        print("  [PASS] EXISTING phase patterns")

    def test_nic_patterns(self):
        """Test NOT_IN_CONTRACT phase pattern matching."""
        import re
        patterns = [re.compile(p, re.IGNORECASE) for p in PHASE_PATTERNS[ConstructionPhase.NOT_IN_CONTRACT]]

        test_strings = [
            "N.I.C.",
            "NIC",
            "NOT IN CONTRACT",
            "AREA NOT IN CONTRACT",
            "BY OTHERS",
        ]

        for s in test_strings:
            matched = any(p.search(s) for p in patterns)
            assert matched, f"Should match N.I.C.: {s}"

        print("  [PASS] N.I.C. phase patterns")

    def test_demo_patterns(self):
        """Test DEMO phase pattern matching."""
        import re
        patterns = [re.compile(p, re.IGNORECASE) for p in PHASE_PATTERNS[ConstructionPhase.DEMO]]

        test_strings = [
            "DEMOLITION",
            "DEMO",
            "REMOVE",
            "TO BE REMOVED",
        ]

        for s in test_strings:
            matched = any(p.search(s) for p in patterns)
            assert matched, f"Should match DEMO: {s}"

        print("  [PASS] DEMO phase patterns")


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_detect_legend_function(self):
        """Test detect_legend convenience function."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            filepath = f.name

        try:
            create_test_pdf_with_legend(filepath)
            doc = pymupdf.open(filepath)
            page = doc.load_page(0)

            result = detect_legend(page)
            assert isinstance(result, LegendDetectionResult)

            doc.close()
            print("  [PASS] detect_legend function")
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_get_legend_or_defaults_function(self):
        """Test get_legend_or_defaults convenience function."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            filepath = f.name

        try:
            create_test_pdf_without_legend(filepath)
            doc = pymupdf.open(filepath)
            page = doc.load_page(0)

            result = get_legend_or_defaults(page)
            # Should always return entries (from defaults if no legend)
            assert len(result.entries) > 0

            doc.close()
            print("  [PASS] get_legend_or_defaults function")
        finally:
            Path(filepath).unlink(missing_ok=True)


def run_all_tests():
    """Run all legend detector tests."""
    print("=" * 60)
    print("Legend Detector Tests")
    print("=" * 60)
    print()

    all_passed = True

    # TextBlock Tests
    print("TextBlock Tests:")
    print("-" * 40)
    try:
        tests = TestTextBlock()
        tests.test_basic_creation()
        tests.test_center_property()
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        all_passed = False
    except Exception as e:
        print(f"  [ERROR] {e}")
        all_passed = False

    print()

    # LegendDetector Tests
    print("LegendDetector Tests:")
    print("-" * 40)
    try:
        tests = TestLegendDetector()
        tests.test_detect_with_legend()
        tests.test_detect_without_legend_uses_defaults()
        tests.test_detect_without_legend_no_defaults()
        tests.test_custom_parameters()
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        all_passed = False
    except Exception as e:
        print(f"  [ERROR] {e}")
        all_passed = False

    print()

    # Phase Pattern Tests
    print("Phase Pattern Tests:")
    print("-" * 40)
    try:
        tests = TestPhasePatterns()
        tests.test_new_patterns()
        tests.test_existing_patterns()
        tests.test_nic_patterns()
        tests.test_demo_patterns()
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
        tests.test_detect_legend_function()
        tests.test_get_legend_or_defaults_function()
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        all_passed = False
    except Exception as e:
        print(f"  [ERROR] {e}")
        all_passed = False

    print()
    print("=" * 60)
    if all_passed:
        print("ALL LEGEND DETECTOR TESTS PASSED")
        return 0
    else:
        print("SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
