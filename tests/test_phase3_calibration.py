"""
Phase 3 Tests: Calibration and Scale Detection

Tests for dimension pattern matching, scale detection, unit conversion,
and dimension-to-line association.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.calibration import (
    ScaleResult,
    DimensionMatch,
    find_dimension_matches,
    find_scale_from_text,
    parse_manual_scale,
    check_scale_conflict,
    pdf_points_to_real,
    real_to_pdf_points,
    area_points_to_real,
    real_area_to_points,
    format_imperial_length,
    format_area,
    calculate_scale_factor_from_calibration,
    parse_calibration_string,
    DimensionAssociation,
)
from src.constants import Confidence


def test_dimension_pattern_imperial_feet_inches():
    """Test imperial feet-inches patterns like 10'-6", 10' 6", 10'6"."""
    test_cases = [
        ("10'-6\"", 126.0),      # 10 feet 6 inches = 126 inches
        ("10' 6\"", 126.0),
        ("5'-0\"", 60.0),        # 5 feet = 60 inches
        ("12'-3 1/2\"", 147.5),  # 12 feet 3.5 inches
    ]

    passed = 0
    for text, expected in test_cases:
        matches = find_dimension_matches(text)
        if matches and abs(matches[0][1] - expected) < 0.1:
            passed += 1
        else:
            print(f"  FAIL: '{text}' expected {expected}, got {matches}")

    print(f"  [{'PASS' if passed == len(test_cases) else 'FAIL'}] Imperial feet-inches: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_dimension_pattern_imperial_feet_only():
    """Test imperial feet-only patterns like 10', 10 FT."""
    test_cases = [
        ("10'", 120.0),          # 10 feet = 120 inches
        ("10 FT", 120.0),
        ("25.5'", 306.0),        # 25.5 feet = 306 inches
    ]

    passed = 0
    for text, expected in test_cases:
        matches = find_dimension_matches(text)
        # Check if expected value is in any of the matches
        found = any(abs(m[1] - expected) < 0.1 for m in matches) if matches else False
        if found:
            passed += 1
        else:
            print(f"  FAIL: '{text}' expected {expected}, got {matches}")

    print(f"  [{'PASS' if passed == len(test_cases) else 'FAIL'}] Imperial feet-only: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_dimension_pattern_imperial_inches_only():
    """Test imperial inches-only patterns like 126", 126 IN."""
    test_cases = [
        ("126\"", 126.0),
        ("126 IN", 126.0),
        ("48 inches", 48.0),
    ]

    passed = 0
    for text, expected in test_cases:
        matches = find_dimension_matches(text)
        if matches and abs(matches[0][1] - expected) < 0.1:
            passed += 1
        else:
            print(f"  FAIL: '{text}' expected {expected}, got {matches}")

    print(f"  [{'PASS' if passed == len(test_cases) else 'FAIL'}] Imperial inches-only: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_dimension_pattern_metric():
    """Test metric patterns like 3000mm, 300cm, 3.0m."""
    test_cases = [
        ("3000mm", 3000 / 25.4),     # mm to inches
        ("300cm", 300 / 2.54),       # cm to inches
        ("3.0m", 3.0 * 39.37),       # m to inches
    ]

    passed = 0
    for text, expected in test_cases:
        matches = find_dimension_matches(text)
        if matches:
            value = matches[0][1]
            is_metric = matches[0][2]
            if abs(value - expected) < 0.1 and is_metric:
                passed += 1
            else:
                print(f"  FAIL: '{text}' expected {expected:.1f} (metric), got {value:.1f} (metric={is_metric})")
        else:
            print(f"  FAIL: '{text}' no matches found")

    print(f"  [{'PASS' if passed == len(test_cases) else 'FAIL'}] Metric patterns: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_scale_fractional_pattern():
    """Test fractional scale patterns like 1/8" = 1'-0"."""
    test_cases = [
        ('1/8" = 1\'-0"', 0.75),     # 1/8 * 72 / 12 = 0.75 points per foot
        ('1/4" = 1\'-0"', 1.5),      # 1/4 * 72 / 12 = 1.5 points per foot
        ('1/2" = 1\'', 3.0),         # 1/2 * 72 / 12 = 3.0 points per foot
    ]

    passed = 0
    for text, expected in test_cases:
        result = find_scale_from_text(text)
        if result:
            factor, notation = result
            if abs(factor - expected) < 0.01:
                passed += 1
            else:
                print(f"  FAIL: '{text}' expected factor {expected}, got {factor}")
        else:
            print(f"  FAIL: '{text}' no scale found")

    print(f"  [{'PASS' if passed == len(test_cases) else 'FAIL'}] Scale fractional: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_scale_ratio_pattern():
    """Test metric ratio patterns like 1:100."""
    test_cases = [
        ("1:100", 28.35),    # 2835 / 100 = 28.35 points per meter
        ("1:50", 56.7),      # 2835 / 50 = 56.7 points per meter
        ("1 : 200", 14.175), # 2835 / 200 = 14.175 points per meter
    ]

    passed = 0
    for text, expected in test_cases:
        result = find_scale_from_text(text)
        if result:
            factor, notation = result
            if abs(factor - expected) < 0.01:
                passed += 1
            else:
                print(f"  FAIL: '{text}' expected factor {expected}, got {factor}")
        else:
            print(f"  FAIL: '{text}' no scale found")

    print(f"  [{'PASS' if passed == len(test_cases) else 'FAIL'}] Scale ratio: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_parse_manual_scale():
    """Test manual scale parsing."""
    # Test known scale string
    result = parse_manual_scale("1/8 inch = 1 foot")
    assert result is not None, "Should parse known scale"
    assert result.scale_source == "manual"
    assert result.scale_confidence == Confidence.HIGH

    # Test direct factor
    result = parse_manual_scale("96")
    assert result is not None, "Should parse direct factor"
    assert result.scale_factor == 96.0

    print("  [PASS] Manual scale parsing")
    return True


def test_scale_conflict_detection():
    """Test scale conflict detection."""
    # Same scale - no conflict
    assert not check_scale_conflict(96, 96), "Same scales should not conflict"

    # Small difference (within 10%) - no conflict
    assert not check_scale_conflict(96, 100), "Small diff should not conflict"

    # Large difference - conflict
    assert check_scale_conflict(96, 150), "Large diff should conflict"

    # Zero handling
    assert not check_scale_conflict(0, 96), "Zero should not conflict"

    print("  [PASS] Scale conflict detection")
    return True


def test_pdf_points_to_real():
    """Test PDF points to real unit conversion."""
    scale_factor = 96  # 96 points per foot (1/8" = 1')

    # 96 points should equal 1 foot
    result = pdf_points_to_real(96, scale_factor, "feet")
    assert abs(result - 1.0) < 0.001, f"Expected 1.0 feet, got {result}"

    # 96 points should equal 12 inches
    result = pdf_points_to_real(96, scale_factor, "inches")
    assert abs(result - 12.0) < 0.001, f"Expected 12.0 inches, got {result}"

    # Test meters conversion
    result = pdf_points_to_real(96, scale_factor, "meters")
    expected = 1.0 / 3.28084  # 1 foot in meters
    assert abs(result - expected) < 0.01, f"Expected {expected} meters, got {result}"

    # Test zero scale factor
    result = pdf_points_to_real(96, 0, "feet")
    assert result == 0.0, "Zero scale should return 0"

    print("  [PASS] PDF points to real conversion")
    return True


def test_real_to_pdf_points():
    """Test real units to PDF points conversion."""
    scale_factor = 96  # 96 points per foot

    # 1 foot should equal 96 points
    result = real_to_pdf_points(1.0, scale_factor, "feet")
    assert abs(result - 96.0) < 0.001, f"Expected 96.0 points, got {result}"

    # 12 inches should equal 96 points
    result = real_to_pdf_points(12.0, scale_factor, "inches")
    assert abs(result - 96.0) < 0.001, f"Expected 96.0 points, got {result}"

    print("  [PASS] Real to PDF points conversion")
    return True


def test_area_conversion():
    """Test area unit conversions."""
    scale_factor = 96  # 96 points per foot

    # 96*96 sq points = 1 sq foot
    sq_points = 96 * 96
    result = area_points_to_real(sq_points, scale_factor, "sqft")
    assert abs(result - 1.0) < 0.001, f"Expected 1.0 sqft, got {result}"

    # Reverse conversion
    result = real_area_to_points(1.0, scale_factor, "sqft")
    assert abs(result - sq_points) < 0.001, f"Expected {sq_points} sq points, got {result}"

    print("  [PASS] Area conversion")
    return True


def test_format_imperial_length():
    """Test imperial length formatting."""
    assert format_imperial_length(10.0) == "10'-0\"", "10 feet"
    assert format_imperial_length(10.5) == "10'-6\"", "10 feet 6 inches"
    assert format_imperial_length(10.25) == "10'-3\"", "10 feet 3 inches"

    print("  [PASS] Imperial length formatting")
    return True


def test_format_area():
    """Test area formatting."""
    assert format_area(150.0, "sqft") == "150.0 SF"
    assert format_area(14.0, "sqm") == "14.0 m²"

    print("  [PASS] Area formatting")
    return True


def test_calibration_from_points():
    """Test scale factor calculation from two-point calibration."""
    # Two points 200 pts apart, representing 10 feet
    point1 = (100.0, 100.0)
    point2 = (300.0, 100.0)  # 200 pts horizontal distance
    real_length = 10.0

    result = calculate_scale_factor_from_calibration(point1, point2, real_length, "feet")
    expected = 200.0 / 10.0  # 20 points per foot

    assert abs(result - expected) < 0.001, f"Expected {expected}, got {result}"

    # Test with inches
    result = calculate_scale_factor_from_calibration(point1, point2, 120.0, "inches")  # 10 feet = 120 inches
    assert abs(result - expected) < 0.001, f"Expected {expected}, got {result}"

    print("  [PASS] Calibration from points")
    return True


def test_parse_calibration_string():
    """Test calibration string parsing."""
    calib = "100,100:300,100=10ft"
    point1, point2, length, unit = parse_calibration_string(calib)

    assert point1 == (100.0, 100.0), f"Expected (100,100), got {point1}"
    assert point2 == (300.0, 100.0), f"Expected (300,100), got {point2}"
    assert length == 10.0, f"Expected 10.0, got {length}"
    assert unit == "feet", f"Expected 'feet', got {unit}"

    # Test with meters
    calib = "0,0:100,0=5m"
    point1, point2, length, unit = parse_calibration_string(calib)
    assert unit == "meters", f"Expected 'meters', got {unit}"

    print("  [PASS] Calibration string parsing")
    return True


def run_all_tests():
    """Run all Phase 3 tests."""
    print("\n" + "=" * 60)
    print("Phase 3 Tests: Calibration and Scale Detection")
    print("=" * 60)

    results = []

    # Dimension pattern tests
    print("\nDimension Pattern Matching Tests:")
    results.append(test_dimension_pattern_imperial_feet_inches())
    results.append(test_dimension_pattern_imperial_feet_only())
    results.append(test_dimension_pattern_imperial_inches_only())
    results.append(test_dimension_pattern_metric())

    # Scale detection tests
    print("\nScale Detection Tests:")
    results.append(test_scale_fractional_pattern())
    results.append(test_scale_ratio_pattern())
    results.append(test_parse_manual_scale())
    results.append(test_scale_conflict_detection())

    # Unit conversion tests
    print("\nUnit Conversion Tests:")
    results.append(test_pdf_points_to_real())
    results.append(test_real_to_pdf_points())
    results.append(test_area_conversion())

    # Formatting tests
    print("\nFormatting Tests:")
    results.append(test_format_imperial_length())
    results.append(test_format_area())

    # Calibration tests
    print("\nCalibration Tests:")
    results.append(test_calibration_from_points())
    results.append(test_parse_calibration_string())

    # Summary
    passed = sum(results)
    total = len(results)
    print("\n" + "=" * 60)
    print(f"Phase 3 Results: {passed}/{total} tests passed")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
