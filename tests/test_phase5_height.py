"""
Phase 5 Tests: Ceiling Height Extraction

Tests for height pattern matching, RCP extraction, and cross-sheet matching.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.text import (
    HeightAnnotation,
    extract_height_from_text,
    match_heights_to_rooms,
    normalize_room_name,
    extract_room_number_from_name,
)
from src.constants import Confidence, DEFAULT_CEILING_HEIGHT_FT


def test_height_pattern_imperial_clg():
    """Test CLG pattern parsing."""
    test_cases = [
        ("CLG 9'-0\"", 9.0),
        ("CLG: 10'0\"", 10.0),
        ("CLG @ 12 FT", 12.0),
        ("CLG 9'-6\"", 9.5),
    ]

    passed = 0
    for text, expected in test_cases:
        result = extract_height_from_text(text)
        if result and abs(result[0] - expected) < 0.1:
            passed += 1
        else:
            print(f"  FAIL: '{text}' expected {expected}, got {result}")

    print(f"  [{'PASS' if passed == len(test_cases) else 'FAIL'}] CLG patterns: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_height_pattern_imperial_ceil():
    """Test CEIL/CEILING pattern parsing."""
    test_cases = [
        ("CEIL 10'-0\"", 10.0),
        ("CEILING: 12'", 12.0),
        ("CEILING 8'-6\"", 8.5),
    ]

    passed = 0
    for text, expected in test_cases:
        result = extract_height_from_text(text)
        if result and abs(result[0] - expected) < 0.1:
            passed += 1
        else:
            print(f"  FAIL: '{text}' expected {expected}, got {result}")

    print(f"  [{'PASS' if passed == len(test_cases) else 'FAIL'}] CEIL patterns: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_height_pattern_imperial_aff():
    """Test A.F.F. pattern parsing."""
    test_cases = [
        ("A.F.F. 9'-6\"", 9.5),
        ("AFF 10'0\"", 10.0),
        ("AFF: 12'", 12.0),
    ]

    passed = 0
    for text, expected in test_cases:
        result = extract_height_from_text(text)
        if result and abs(result[0] - expected) < 0.1:
            passed += 1
        else:
            print(f"  FAIL: '{text}' expected {expected}, got {result}")

    print(f"  [{'PASS' if passed == len(test_cases) else 'FAIL'}] AFF patterns: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_height_pattern_imperial_trailing():
    """Test patterns with height at end."""
    test_cases = [
        ("9'-0\" CLG", 9.0),
        ("10 FT CEILING", 10.0),
        ("+9'-0\"", 9.0),
    ]

    passed = 0
    for text, expected in test_cases:
        result = extract_height_from_text(text)
        if result and abs(result[0] - expected) < 0.1:
            passed += 1
        else:
            print(f"  FAIL: '{text}' expected {expected}, got {result}")

    print(f"  [{'PASS' if passed == len(test_cases) else 'FAIL'}] Trailing patterns: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_height_pattern_metric():
    """Test metric height patterns."""
    test_cases = [
        ("CLG 2700mm", 2.7 * 3.28084),   # 2700mm -> ~8.86 feet
        ("CLG 3000 MM", 3.0 * 3.28084),  # 3000mm -> ~9.84 feet
        ("CEILING 2.7m", 2.7 * 3.28084), # 2.7m -> ~8.86 feet
        ("CLG 3.0M", 3.0 * 3.28084),     # 3.0m -> ~9.84 feet
    ]

    passed = 0
    for text, expected in test_cases:
        result = extract_height_from_text(text)
        if result and abs(result[0] - expected) < 0.1:
            passed += 1
        else:
            print(f"  FAIL: '{text}' expected {expected:.2f}, got {result}")

    print(f"  [{'PASS' if passed == len(test_cases) else 'FAIL'}] Metric patterns: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_height_validation():
    """Test height validation (min/max)."""
    # Valid heights
    result = extract_height_from_text("CLG 9'-0\"")
    assert result is not None, "9' should be valid"

    result = extract_height_from_text("CLG 50'-0\"")
    assert result is not None, "50' should be valid (warehouse)"

    # Invalid heights (outside range)
    result = extract_height_from_text("CLG 5'-0\"")
    assert result is None, "5' should be invalid (too low)"

    result = extract_height_from_text("CLG 100'-0\"")
    assert result is None, "100' should be invalid (too high)"

    print("  [PASS] Height validation")
    return True


def test_normalize_room_name():
    """Test room name normalization."""
    assert normalize_room_name("OFFICE 101") == "OFFICE101"
    assert normalize_room_name("office-101") == "OFFICE101"
    assert normalize_room_name("Office_101") == "OFFICE101"
    assert normalize_room_name("  OFFICE  101  ") == "OFFICE101"

    print("  [PASS] Room name normalization")
    return True


def test_extract_room_number():
    """Test room number extraction."""
    assert extract_room_number_from_name("OFFICE 101") == "101"
    assert extract_room_number_from_name("A-205") == "205"
    assert extract_room_number_from_name("CONFERENCE") is None
    assert extract_room_number_from_name("1001A") == "1001"

    print("  [PASS] Room number extraction")
    return True


def test_match_heights_to_rooms_by_name():
    """Test height matching by room name."""
    rooms = [
        {"room_name": "OFFICE 101", "vertices": [(0, 0), (100, 0), (100, 100), (0, 100)]},
        {"room_name": "CONFERENCE", "vertices": [(200, 0), (300, 0), (300, 100), (200, 100)]},
    ]

    heights = [
        HeightAnnotation(
            location=(50, 50),
            height_feet=9.0,
            raw_text="CLG 9'",
            confidence=1.0,
            source="embedded",
            room_name="OFFICE 101"
        ),
        HeightAnnotation(
            location=(250, 50),
            height_feet=10.0,
            raw_text="CLG 10'",
            confidence=1.0,
            source="embedded",
            room_name="CONFERENCE"
        ),
    ]

    result = match_heights_to_rooms(rooms, heights, page_width=612)

    assert result[0]["ceiling_height_ft"] == 9.0
    assert result[0]["height_confidence"] == Confidence.HIGH
    assert result[1]["ceiling_height_ft"] == 10.0
    assert result[1]["height_confidence"] == Confidence.HIGH

    print("  [PASS] Height matching by name")
    return True


def test_match_heights_to_rooms_by_number():
    """Test height matching by room number."""
    rooms = [
        {"room_name": "OFFICE 101", "vertices": [(0, 0), (100, 0), (100, 100), (0, 100)]},
    ]

    heights = [
        HeightAnnotation(
            location=(50, 50),
            height_feet=9.0,
            raw_text="CLG 9'",
            confidence=1.0,
            source="embedded",
            room_name="101"  # Just the number
        ),
    ]

    result = match_heights_to_rooms(rooms, heights, page_width=612)

    assert result[0]["ceiling_height_ft"] == 9.0
    assert result[0]["height_confidence"] == Confidence.MEDIUM
    assert result[0]["height_source"] == "rcp_number_match"

    print("  [PASS] Height matching by number")
    return True


def test_match_heights_to_rooms_default():
    """Test default height assignment."""
    rooms = [
        {"room_name": "UNKNOWN", "vertices": [(0, 0), (100, 0), (100, 100), (0, 100)]},
    ]

    heights = []  # No heights available

    result = match_heights_to_rooms(rooms, heights, page_width=612)

    assert result[0]["ceiling_height_ft"] == DEFAULT_CEILING_HEIGHT_FT
    assert result[0]["height_confidence"] == Confidence.NONE
    assert result[0]["height_source"] == "default"

    print("  [PASS] Default height assignment")
    return True


def test_height_annotation_dataclass():
    """Test HeightAnnotation dataclass."""
    annotation = HeightAnnotation(
        location=(100, 200),
        height_feet=9.0,
        raw_text="CLG 9'-0\"",
        confidence=0.95,
        source="ocr",
        room_name="101"
    )

    assert annotation.location == (100, 200)
    assert annotation.height_feet == 9.0
    assert annotation.room_name == "101"

    print("  [PASS] HeightAnnotation dataclass")
    return True


def run_all_tests():
    """Run all Phase 5 tests."""
    print("\n" + "=" * 60)
    print("Phase 5 Tests: Ceiling Height Extraction")
    print("=" * 60)

    results = []

    # Height pattern tests
    print("\nHeight Pattern Matching Tests:")
    results.append(test_height_pattern_imperial_clg())
    results.append(test_height_pattern_imperial_ceil())
    results.append(test_height_pattern_imperial_aff())
    results.append(test_height_pattern_imperial_trailing())
    results.append(test_height_pattern_metric())
    results.append(test_height_validation())

    # Room name/number tests
    print("\nRoom Name Processing Tests:")
    results.append(test_normalize_room_name())
    results.append(test_extract_room_number())

    # Height-to-room matching tests
    print("\nHeight-to-Room Matching Tests:")
    results.append(test_match_heights_to_rooms_by_name())
    results.append(test_match_heights_to_rooms_by_number())
    results.append(test_match_heights_to_rooms_default())

    # Dataclass tests
    print("\nDataclass Tests:")
    results.append(test_height_annotation_dataclass())

    # Summary
    passed = sum(results)
    total = len(results)
    print("\n" + "=" * 60)
    print(f"Phase 5 Results: {passed}/{total} tests passed")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
