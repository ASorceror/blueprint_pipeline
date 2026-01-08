#!/usr/bin/env python
"""
Construction Phase Detection Tests

Tests for:
- ConstructionPhase enum and parsing
- FillPattern and ClassificationMethod enums
- LegendEntry and LegendDetectionResult dataclasses
- PhaseClassificationStats tracking
- Fill color classification utilities
- Segment fill attribute integration
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vector.construction_phase import (
    ConstructionPhase,
    FillPattern,
    ClassificationMethod,
    LegendEntry,
    LegendDetectionResult,
    PhaseClassificationStats,
    is_gray_fill,
    classify_fill_color,
    get_default_legend,
    get_phase_color,
    PHASE_COLORS,
)
from src.vector.extractor import Segment


class TestConstructionPhaseEnum:
    """Tests for ConstructionPhase enum."""

    def test_enum_values(self):
        """Test that all expected phase values exist."""
        assert ConstructionPhase.NEW.value == "NEW"
        assert ConstructionPhase.EXISTING.value == "EXISTING"
        assert ConstructionPhase.NOT_IN_CONTRACT.value == "NOT_IN_CONTRACT"
        assert ConstructionPhase.DEMO.value == "DEMO"
        assert ConstructionPhase.UNKNOWN.value == "UNKNOWN"
        print("  [PASS] Enum values defined correctly")

    def test_from_string_exact_match(self):
        """Test parsing exact enum values."""
        assert ConstructionPhase.from_string("NEW") == ConstructionPhase.NEW
        assert ConstructionPhase.from_string("EXISTING") == ConstructionPhase.EXISTING
        assert ConstructionPhase.from_string("NOT_IN_CONTRACT") == ConstructionPhase.NOT_IN_CONTRACT
        assert ConstructionPhase.from_string("DEMO") == ConstructionPhase.DEMO
        print("  [PASS] Exact string matching")

    def test_from_string_case_insensitive(self):
        """Test case-insensitive parsing."""
        assert ConstructionPhase.from_string("new") == ConstructionPhase.NEW
        assert ConstructionPhase.from_string("Existing") == ConstructionPhase.EXISTING
        assert ConstructionPhase.from_string("demo") == ConstructionPhase.DEMO
        print("  [PASS] Case-insensitive parsing")

    def test_from_string_fuzzy_matching(self):
        """Test fuzzy matching for common variations."""
        # NEW variations
        assert ConstructionPhase.from_string("NEW CONSTRUCTION") == ConstructionPhase.NEW
        assert ConstructionPhase.from_string("NEW WALL") == ConstructionPhase.NEW
        assert ConstructionPhase.from_string("NEW WORK") == ConstructionPhase.NEW

        # EXISTING variations
        assert ConstructionPhase.from_string("EXISTING CONSTRUCTION") == ConstructionPhase.EXISTING
        assert ConstructionPhase.from_string("EXISTING TO REMAIN") == ConstructionPhase.EXISTING
        assert ConstructionPhase.from_string("EXIST") == ConstructionPhase.EXISTING

        # N.I.C. variations
        assert ConstructionPhase.from_string("N.I.C.") == ConstructionPhase.NOT_IN_CONTRACT
        assert ConstructionPhase.from_string("NIC") == ConstructionPhase.NOT_IN_CONTRACT
        assert ConstructionPhase.from_string("NOT IN CONTRACT") == ConstructionPhase.NOT_IN_CONTRACT

        # DEMO variations
        assert ConstructionPhase.from_string("DEMOLITION") == ConstructionPhase.DEMO
        assert ConstructionPhase.from_string("REMOVE") == ConstructionPhase.DEMO
        assert ConstructionPhase.from_string("DEMOLISH") == ConstructionPhase.DEMO

        print("  [PASS] Fuzzy matching for variations")

    def test_from_string_unknown(self):
        """Test that unknown strings return UNKNOWN."""
        assert ConstructionPhase.from_string("FOOBAR") == ConstructionPhase.UNKNOWN
        assert ConstructionPhase.from_string("") == ConstructionPhase.UNKNOWN
        assert ConstructionPhase.from_string(None) == ConstructionPhase.UNKNOWN
        print("  [PASS] Unknown strings handled")

    def test_is_in_scope(self):
        """Test scope checking method."""
        assert ConstructionPhase.NEW.is_in_scope() is True
        assert ConstructionPhase.EXISTING.is_in_scope() is True
        assert ConstructionPhase.NOT_IN_CONTRACT.is_in_scope() is False
        assert ConstructionPhase.DEMO.is_in_scope() is False
        assert ConstructionPhase.UNKNOWN.is_in_scope() is False
        print("  [PASS] Scope checking")

    def test_is_new_work(self):
        """Test new work checking method."""
        assert ConstructionPhase.NEW.is_new_work() is True
        assert ConstructionPhase.EXISTING.is_new_work() is False
        assert ConstructionPhase.NOT_IN_CONTRACT.is_new_work() is False
        print("  [PASS] New work checking")


class TestFillPatternEnum:
    """Tests for FillPattern enum."""

    def test_enum_values(self):
        """Test that all expected fill patterns exist."""
        assert FillPattern.NONE.value == "none"
        assert FillPattern.SOLID_GRAY.value == "gray"
        assert FillPattern.SOLID_BLACK.value == "black"
        assert FillPattern.HATCHED.value == "hatched"
        assert FillPattern.CROSS_HATCHED.value == "cross_hatched"
        assert FillPattern.WHITE.value == "white"
        assert FillPattern.UNKNOWN.value == "unknown"
        print("  [PASS] Fill pattern values defined")


class TestLegendEntry:
    """Tests for LegendEntry dataclass."""

    def test_basic_creation(self):
        """Test creating a legend entry."""
        entry = LegendEntry(
            label="NEW CONSTRUCTION",
            phase=ConstructionPhase.NEW,
            fill_pattern=FillPattern.NONE,
            fill_color=None,
            confidence=0.90
        )
        assert entry.label == "NEW CONSTRUCTION"
        assert entry.phase == ConstructionPhase.NEW
        assert entry.fill_pattern == FillPattern.NONE
        assert entry.fill_color is None
        assert entry.confidence == 0.90
        print("  [PASS] Basic legend entry creation")

    def test_matches_fill_no_fill(self):
        """Test fill matching for no-fill pattern."""
        entry = LegendEntry(
            label="NEW",
            phase=ConstructionPhase.NEW,
            fill_pattern=FillPattern.NONE,
            fill_color=None,
            confidence=0.85
        )
        assert entry.matches_fill(None) is True
        assert entry.matches_fill((0.5, 0.5, 0.5)) is False
        print("  [PASS] No-fill pattern matching")

    def test_matches_fill_gray(self):
        """Test fill matching for gray fill."""
        entry = LegendEntry(
            label="EXISTING",
            phase=ConstructionPhase.EXISTING,
            fill_pattern=FillPattern.SOLID_GRAY,
            fill_color=(0.5, 0.5, 0.5),
            confidence=0.85
        )
        # Exact match
        assert entry.matches_fill((0.5, 0.5, 0.5)) is True
        # Close match (within tolerance)
        assert entry.matches_fill((0.498, 0.498, 0.498)) is True
        # Not a match (different color)
        assert entry.matches_fill((0.8, 0.8, 0.8)) is False
        # No fill doesn't match
        assert entry.matches_fill(None) is False
        print("  [PASS] Gray fill matching")


class TestLegendDetectionResult:
    """Tests for LegendDetectionResult dataclass."""

    def test_empty_result(self):
        """Test default empty result."""
        result = LegendDetectionResult()
        assert result.entries == []
        assert result.has_legend is False
        assert result.confidence == 0.0
        print("  [PASS] Empty result creation")

    def test_with_entries(self):
        """Test result with legend entries."""
        entries = [
            LegendEntry("NEW", ConstructionPhase.NEW, FillPattern.NONE),
            LegendEntry("EXISTING", ConstructionPhase.EXISTING, FillPattern.SOLID_GRAY),
        ]
        result = LegendDetectionResult(
            entries=entries,
            has_legend=True,
            confidence=0.90,
            detection_method="text_search"
        )
        assert len(result.entries) == 2
        assert result.has_legend is True
        assert result.get_entry_for_phase(ConstructionPhase.NEW) is not None
        assert result.get_entry_for_phase(ConstructionPhase.DEMO) is None
        print("  [PASS] Result with entries")

    def test_get_phases(self):
        """Test getting list of phases."""
        entries = [
            LegendEntry("NEW", ConstructionPhase.NEW, FillPattern.NONE),
            LegendEntry("EXISTING", ConstructionPhase.EXISTING, FillPattern.SOLID_GRAY),
        ]
        result = LegendDetectionResult(entries=entries, has_legend=True)
        phases = result.get_phases()
        assert ConstructionPhase.NEW in phases
        assert ConstructionPhase.EXISTING in phases
        assert ConstructionPhase.DEMO not in phases
        print("  [PASS] Get phases list")


class TestPhaseClassificationStats:
    """Tests for PhaseClassificationStats dataclass."""

    def test_add_classifications(self):
        """Test adding classification results."""
        stats = PhaseClassificationStats()

        # Add some classifications
        stats.add_classification(ConstructionPhase.NEW, 0.85, ClassificationMethod.FILL_PATTERN)
        stats.add_classification(ConstructionPhase.NEW, 0.90, ClassificationMethod.FILL_PATTERN)
        stats.add_classification(ConstructionPhase.EXISTING, 0.80, ClassificationMethod.LEGEND_MATCH)
        stats.add_classification(ConstructionPhase.UNKNOWN, 0.50, ClassificationMethod.UNKNOWN)

        assert stats.total_segments == 4
        assert stats.new_count == 2
        assert stats.existing_count == 1
        assert stats.unknown_count == 1
        assert stats.nic_count == 0
        print("  [PASS] Add classifications")

    def test_confidence_tracking(self):
        """Test confidence statistics."""
        stats = PhaseClassificationStats()
        stats.add_classification(ConstructionPhase.NEW, 0.80, ClassificationMethod.FILL_PATTERN)
        stats.add_classification(ConstructionPhase.NEW, 0.90, ClassificationMethod.FILL_PATTERN)
        stats.add_classification(ConstructionPhase.EXISTING, 0.70, ClassificationMethod.LEGEND_MATCH)

        assert stats.min_confidence == 0.70
        assert abs(stats.avg_confidence - 0.80) < 0.01  # (0.8 + 0.9 + 0.7) / 3 = 0.8
        print("  [PASS] Confidence tracking")

    def test_phase_distribution(self):
        """Test phase distribution calculation."""
        stats = PhaseClassificationStats()
        stats.add_classification(ConstructionPhase.NEW, 0.85, ClassificationMethod.FILL_PATTERN)
        stats.add_classification(ConstructionPhase.NEW, 0.85, ClassificationMethod.FILL_PATTERN)
        stats.add_classification(ConstructionPhase.EXISTING, 0.85, ClassificationMethod.FILL_PATTERN)
        stats.add_classification(ConstructionPhase.NOT_IN_CONTRACT, 0.85, ClassificationMethod.FILL_PATTERN)

        dist = stats.get_phase_distribution()
        assert dist["NEW"] == 50.0  # 2 out of 4
        assert dist["EXISTING"] == 25.0  # 1 out of 4
        assert dist["NOT_IN_CONTRACT"] == 25.0  # 1 out of 4
        assert dist["DEMO"] == 0.0
        print("  [PASS] Phase distribution")

    def test_method_tracking(self):
        """Test classification method tracking."""
        stats = PhaseClassificationStats()
        stats.add_classification(ConstructionPhase.NEW, 0.85, ClassificationMethod.FILL_PATTERN)
        stats.add_classification(ConstructionPhase.NEW, 0.85, ClassificationMethod.FILL_PATTERN)
        stats.add_classification(ConstructionPhase.EXISTING, 0.75, ClassificationMethod.INDUSTRY_DEFAULT)

        assert stats.methods_used["fill_pattern"] == 2
        assert stats.methods_used["industry_default"] == 1
        print("  [PASS] Method tracking")

    def test_summary_generation(self):
        """Test summary string generation."""
        stats = PhaseClassificationStats()
        stats.add_classification(ConstructionPhase.NEW, 0.85, ClassificationMethod.FILL_PATTERN)
        stats.add_classification(ConstructionPhase.EXISTING, 0.80, ClassificationMethod.LEGEND_MATCH)

        summary = stats.summary()
        assert "Total segments: 2" in summary
        assert "NEW" in summary
        assert "EXISTING" in summary
        print("  [PASS] Summary generation")


class TestFillUtilities:
    """Tests for fill classification utility functions."""

    def test_is_gray_fill_true(self):
        """Test gray fill detection for actual grays."""
        assert is_gray_fill((0.5, 0.5, 0.5)) is True
        assert is_gray_fill((0.498, 0.498, 0.498)) is True
        assert is_gray_fill((0.4, 0.4, 0.4)) is True
        assert is_gray_fill((0.6, 0.6, 0.6)) is True
        print("  [PASS] Gray fill detection (true cases)")

    def test_is_gray_fill_false(self):
        """Test gray fill detection for non-grays."""
        # Too light
        assert is_gray_fill((0.9, 0.9, 0.9)) is False
        # Too dark
        assert is_gray_fill((0.1, 0.1, 0.1)) is False
        # Colored (not grayscale)
        assert is_gray_fill((1.0, 0.0, 0.0)) is False
        assert is_gray_fill((0.5, 0.6, 0.4)) is False
        # No fill
        assert is_gray_fill(None) is False
        print("  [PASS] Gray fill detection (false cases)")

    def test_classify_fill_color_new(self):
        """Test classification of no-fill as NEW."""
        pattern, phase, conf = classify_fill_color(None, 's')
        assert phase == ConstructionPhase.NEW
        assert pattern == FillPattern.NONE
        assert conf >= 0.75
        print("  [PASS] No-fill classified as NEW")

    def test_classify_fill_color_existing(self):
        """Test classification of gray fill as EXISTING."""
        pattern, phase, conf = classify_fill_color((0.5, 0.5, 0.5), 'f')
        assert phase == ConstructionPhase.EXISTING
        assert pattern == FillPattern.SOLID_GRAY
        assert conf >= 0.80
        print("  [PASS] Gray fill classified as EXISTING")

    def test_classify_fill_color_colored(self):
        """Test classification of colored fill as UNKNOWN."""
        pattern, phase, conf = classify_fill_color((1.0, 0.0, 0.0), 'f')  # Red
        assert phase == ConstructionPhase.UNKNOWN
        print("  [PASS] Colored fill classified as UNKNOWN")


class TestDefaultLegend:
    """Tests for default legend utilities."""

    def test_get_default_legend(self):
        """Test getting industry standard default legend."""
        legend = get_default_legend()
        assert legend.has_legend is False
        assert legend.detection_method == "industry_default"
        assert len(legend.entries) >= 4

        # Should have standard entries
        phases = legend.get_phases()
        assert ConstructionPhase.NEW in phases
        assert ConstructionPhase.EXISTING in phases
        assert ConstructionPhase.NOT_IN_CONTRACT in phases
        assert ConstructionPhase.DEMO in phases
        print("  [PASS] Default legend creation")

    def test_phase_colors(self):
        """Test phase visualization colors."""
        assert ConstructionPhase.NEW in PHASE_COLORS
        assert ConstructionPhase.EXISTING in PHASE_COLORS
        assert ConstructionPhase.NOT_IN_CONTRACT in PHASE_COLORS
        assert ConstructionPhase.DEMO in PHASE_COLORS

        # Test get_phase_color function
        color = get_phase_color(ConstructionPhase.NEW)
        assert len(color) == 3  # RGB tuple
        assert all(0.0 <= c <= 1.0 for c in color)
        print("  [PASS] Phase colors defined")


class TestSegmentFillIntegration:
    """Tests for Segment dataclass fill attribute integration."""

    def test_segment_with_fill(self):
        """Test segment with fill attributes."""
        seg = Segment(
            start=(0, 0), end=(100, 0), width=1.0,
            fill=(0.5, 0.5, 0.5), fill_type='fs'
        )
        assert seg.has_fill is True
        assert seg.is_stroke_only is False
        assert seg.is_gray_fill is True
        print("  [PASS] Segment with fill")

    def test_segment_without_fill(self):
        """Test segment without fill (NEW construction)."""
        seg = Segment(
            start=(0, 0), end=(100, 0), width=1.0,
            fill=None, fill_type='s'
        )
        assert seg.has_fill is False
        assert seg.is_stroke_only is True
        assert seg.is_gray_fill is False
        print("  [PASS] Segment without fill")

    def test_segment_set_phase(self):
        """Test setting phase on segment."""
        seg = Segment(start=(0, 0), end=(100, 0), width=1.0)
        seg.set_phase("NEW", 0.85, "fill_pattern")

        assert seg.construction_phase == "NEW"
        assert seg.phase_confidence == 0.85
        assert seg.phase_method == "fill_pattern"
        print("  [PASS] Set phase on segment")

    def test_segment_default_phase_attributes(self):
        """Test default phase attributes on new segment."""
        seg = Segment(start=(0, 0), end=(100, 0), width=1.0)
        assert seg.construction_phase is None
        assert seg.phase_confidence == 0.0
        assert seg.phase_method is None
        assert seg.fill is None
        assert seg.fill_type is None
        print("  [PASS] Default phase attributes")


def run_all_tests():
    """Run all construction phase tests."""
    print("=" * 60)
    print("Construction Phase Detection Tests")
    print("=" * 60)
    print()

    all_passed = True

    # ConstructionPhase Enum Tests
    print("ConstructionPhase Enum Tests:")
    print("-" * 40)
    try:
        tests = TestConstructionPhaseEnum()
        tests.test_enum_values()
        tests.test_from_string_exact_match()
        tests.test_from_string_case_insensitive()
        tests.test_from_string_fuzzy_matching()
        tests.test_from_string_unknown()
        tests.test_is_in_scope()
        tests.test_is_new_work()
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        all_passed = False
    except Exception as e:
        print(f"  [ERROR] {e}")
        all_passed = False

    print()

    # FillPattern Enum Tests
    print("FillPattern Enum Tests:")
    print("-" * 40)
    try:
        tests = TestFillPatternEnum()
        tests.test_enum_values()
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        all_passed = False
    except Exception as e:
        print(f"  [ERROR] {e}")
        all_passed = False

    print()

    # LegendEntry Tests
    print("LegendEntry Tests:")
    print("-" * 40)
    try:
        tests = TestLegendEntry()
        tests.test_basic_creation()
        tests.test_matches_fill_no_fill()
        tests.test_matches_fill_gray()
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        all_passed = False
    except Exception as e:
        print(f"  [ERROR] {e}")
        all_passed = False

    print()

    # LegendDetectionResult Tests
    print("LegendDetectionResult Tests:")
    print("-" * 40)
    try:
        tests = TestLegendDetectionResult()
        tests.test_empty_result()
        tests.test_with_entries()
        tests.test_get_phases()
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        all_passed = False
    except Exception as e:
        print(f"  [ERROR] {e}")
        all_passed = False

    print()

    # PhaseClassificationStats Tests
    print("PhaseClassificationStats Tests:")
    print("-" * 40)
    try:
        tests = TestPhaseClassificationStats()
        tests.test_add_classifications()
        tests.test_confidence_tracking()
        tests.test_phase_distribution()
        tests.test_method_tracking()
        tests.test_summary_generation()
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        all_passed = False
    except Exception as e:
        print(f"  [ERROR] {e}")
        all_passed = False

    print()

    # Fill Utilities Tests
    print("Fill Utility Tests:")
    print("-" * 40)
    try:
        tests = TestFillUtilities()
        tests.test_is_gray_fill_true()
        tests.test_is_gray_fill_false()
        tests.test_classify_fill_color_new()
        tests.test_classify_fill_color_existing()
        tests.test_classify_fill_color_colored()
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        all_passed = False
    except Exception as e:
        print(f"  [ERROR] {e}")
        all_passed = False

    print()

    # Default Legend Tests
    print("Default Legend Tests:")
    print("-" * 40)
    try:
        tests = TestDefaultLegend()
        tests.test_get_default_legend()
        tests.test_phase_colors()
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        all_passed = False
    except Exception as e:
        print(f"  [ERROR] {e}")
        all_passed = False

    print()

    # Segment Integration Tests
    print("Segment Fill Integration Tests:")
    print("-" * 40)
    try:
        tests = TestSegmentFillIntegration()
        tests.test_segment_with_fill()
        tests.test_segment_without_fill()
        tests.test_segment_set_phase()
        tests.test_segment_default_phase_attributes()
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        all_passed = False
    except Exception as e:
        print(f"  [ERROR] {e}")
        all_passed = False

    print()
    print("=" * 60)
    if all_passed:
        print("ALL CONSTRUCTION PHASE TESTS PASSED")
        return 0
    else:
        print("SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
