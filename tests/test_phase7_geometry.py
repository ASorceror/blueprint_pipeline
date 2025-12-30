"""
Phase 7 Tests: Geometry Calculations

Tests for room data structure and measurement calculations.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from shapely.geometry import Polygon

from src.geometry import (
    Room,
    extract_floor_level,
    calculate_floor_area,
    calculate_perimeter,
    calculate_wall_area,
    calculate_ceiling_area,
    convert_polygon_to_real_units,
    validate_room_measurements,
    calculate_all_room_measurements,
    create_room_from_polygon,
)
from src.constants import Confidence


def test_room_dataclass():
    """Test Room dataclass creation."""
    room = Room(
        room_id="room_1",
        room_name="OFFICE 101",
        sheet_number=0,
        sheet_name="A1.1 - LEVEL 1",
        floor_level="LEVEL 1"
    )

    assert room.room_id == "room_1"
    assert room.room_name == "OFFICE 101"
    assert room.sheet_number == 0
    assert room.floor_area_sqft == 0.0  # Default

    print("  [PASS] Room dataclass")
    return True


def test_room_to_dict():
    """Test Room to dictionary conversion."""
    room = Room(
        room_id="room_1",
        room_name="OFFICE",
        sheet_number=0,
        floor_area_sqft=150.5,
        perimeter_ft=50.0,
    )

    d = room.to_dict()

    assert d["room_id"] == "room_1"
    assert d["room_name"] == "OFFICE"
    assert d["floor_area_sqft"] == 150.5
    assert isinstance(d["metadata"], dict)

    print("  [PASS] Room to_dict")
    return True


def test_room_to_csv_row():
    """Test Room to CSV row conversion."""
    room = Room(
        room_id="room_1",
        room_name="OFFICE",
        sheet_number=0,
        floor_area_sqft=150.5555,
    )

    row = room.to_csv_row()
    header = Room.csv_header()

    assert len(row) == len(header)
    assert row[0] == "room_1"
    assert row[5] == 150.56  # Rounded to 2 decimal places

    print("  [PASS] Room to_csv_row")
    return True


def test_extract_floor_level():
    """Test floor level extraction from sheet name."""
    test_cases = [
        ("A1.1 - LEVEL 1 FLOOR PLAN", "LEVEL 1"),
        ("FIRST FLOOR PLAN", "FIRST FLOOR"),
        ("2ND FLOOR", "2ND FLOOR"),
        ("BASEMENT PLAN", "BASEMENT"),
        ("MEZZANINE LEVEL", "MEZZANINE"),
        ("RANDOM SHEET", ""),  # No match
    ]

    passed = 0
    for sheet_name, expected in test_cases:
        result = extract_floor_level(sheet_name)
        if result == expected:
            passed += 1
        else:
            print(f"  FAIL: '{sheet_name}' expected '{expected}', got '{result}'")

    print(f"  [{'PASS' if passed == len(test_cases) else 'FAIL'}] Floor level extraction: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_calculate_floor_area():
    """Test floor area calculation."""
    # Create a 100x100 point square
    polygon = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    # Area = 10000 sq points

    # At scale factor 10 (10 points = 1 foot), 1 foot = 10 points
    # 100 points = 10 feet, so room is 10x10 = 100 sqft
    scale_factor = 10

    area = calculate_floor_area(polygon, scale_factor)

    assert abs(area - 100.0) < 0.01, f"Expected 100.0 sqft, got {area}"

    # Test with zero scale factor
    area_zero = calculate_floor_area(polygon, 0)
    assert area_zero == 0.0, "Zero scale should return 0"

    print("  [PASS] Floor area calculation")
    return True


def test_calculate_perimeter():
    """Test perimeter calculation."""
    # Create a 100x100 point square
    polygon = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    # Perimeter = 400 points

    # At scale factor 10 (10 points = 1 foot)
    # 400 points / 10 = 40 feet
    scale_factor = 10

    perimeter = calculate_perimeter(polygon, scale_factor)

    assert abs(perimeter - 40.0) < 0.01, f"Expected 40.0 ft, got {perimeter}"

    print("  [PASS] Perimeter calculation")
    return True


def test_calculate_wall_area():
    """Test wall area calculation."""
    perimeter = 40.0  # feet
    ceiling_height = 10.0  # feet

    wall_area = calculate_wall_area(perimeter, ceiling_height)

    assert abs(wall_area - 400.0) < 0.01, f"Expected 400.0 sqft, got {wall_area}"

    print("  [PASS] Wall area calculation")
    return True


def test_calculate_ceiling_area():
    """Test ceiling area calculation."""
    floor_area = 150.0

    ceiling_area = calculate_ceiling_area(floor_area)

    assert ceiling_area == floor_area, "Ceiling should equal floor for flat ceiling"

    print("  [PASS] Ceiling area calculation")
    return True


def test_convert_polygon_to_real_units():
    """Test coordinate conversion to real units."""
    vertices = [(0, 0), (100, 0), (100, 100), (0, 100)]
    scale_factor = 10  # 10 points = 1 foot

    real_vertices = convert_polygon_to_real_units(vertices, scale_factor)

    expected = [(0, 0), (10, 0), (10, 10), (0, 10)]

    for (rx, ry), (ex, ey) in zip(real_vertices, expected):
        assert abs(rx - ex) < 0.01 and abs(ry - ey) < 0.01

    print("  [PASS] Coordinate conversion")
    return True


def test_validate_room_measurements():
    """Test room measurement validation."""
    # Valid room
    valid_room = Room(
        room_id="r1",
        room_name="TEST",
        sheet_number=0,
        floor_area_sqft=150.0,
        perimeter_ft=50.0,
        ceiling_height_ft=10.0,
        wall_area_sqft=500.0,
    )
    warnings = validate_room_measurements(valid_room)
    assert len(warnings) == 0, "Valid room should have no warnings"

    # Room too small
    small_room = Room(
        room_id="r2",
        room_name="SMALL",
        sheet_number=0,
        floor_area_sqft=10.0,  # Below MIN_AREA_REAL_SQFT (25)
        perimeter_ft=13.0,
        ceiling_height_ft=9.0,
    )
    warnings = validate_room_measurements(small_room)
    assert any("below minimum" in w for w in warnings), "Should warn about small area"

    # Ceiling too low
    low_ceiling_room = Room(
        room_id="r3",
        room_name="LOW",
        sheet_number=0,
        floor_area_sqft=100.0,
        perimeter_ft=40.0,
        ceiling_height_ft=5.0,  # Below MIN_CEILING_HEIGHT_FT (7)
    )
    warnings = validate_room_measurements(low_ceiling_room)
    assert any("Ceiling height" in w for w in warnings), "Should warn about low ceiling"

    print("  [PASS] Room validation")
    return True


def test_calculate_all_room_measurements():
    """Test complete room measurement calculation."""
    room = Room(
        room_id="room_test",
        room_name="TEST ROOM",
        sheet_number=0,
    )

    # 100x100 point square at scale 10 = 10x10 ft = 100 sqft
    polygon = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    scale_factor = 10
    ceiling_height = 9.0

    result = calculate_all_room_measurements(
        room, polygon, scale_factor, ceiling_height
    )

    assert abs(result.floor_area_sqft - 100.0) < 0.1
    assert abs(result.perimeter_ft - 40.0) < 0.1
    assert result.ceiling_height_ft == 9.0
    assert abs(result.wall_area_sqft - 360.0) < 0.1  # 40 * 9
    assert abs(result.ceiling_area_sqft - 100.0) < 0.1
    assert result.scale_factor == 10
    assert len(result.polygon_pdf_points) == 4
    assert len(result.polygon_real_units) == 4

    print("  [PASS] Complete room calculation")
    return True


def test_create_room_from_polygon():
    """Test room creation from polygon."""
    polygon = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    vertices = [(0, 0), (100, 0), (100, 100), (0, 100)]

    room = create_room_from_polygon(
        polygon_id="poly_1",
        polygon=polygon,
        vertices=vertices,
        room_name="OFFICE 101",
        sheet_number=0,
        scale_factor=10,
        ceiling_height_ft=9.0,
        source="vector",
        confidence=Confidence.HIGH
    )

    assert room.room_id == "poly_1"
    assert room.room_name == "OFFICE 101"
    assert room.source == "vector"
    assert room.confidence == Confidence.HIGH
    assert room.floor_area_sqft > 0

    print("  [PASS] Create room from polygon")
    return True


def test_room_with_realistic_values():
    """Test with realistic room values."""
    # Simulate a 15' x 20' office at 1/8" = 1' scale
    # Scale factor = 9 points per foot (1/8 inch = 9 points)

    scale_factor = 9  # 1/8" = 1' scale
    width_ft = 15
    height_ft = 20

    # Convert to PDF points
    width_pts = width_ft * scale_factor  # 135 pts
    height_pts = height_ft * scale_factor  # 180 pts

    polygon = Polygon([
        (0, 0),
        (width_pts, 0),
        (width_pts, height_pts),
        (0, height_pts)
    ])

    room = create_room_from_polygon(
        polygon_id="office_101",
        polygon=polygon,
        vertices=list(polygon.exterior.coords)[:-1],
        room_name="OFFICE 101",
        sheet_number=0,
        scale_factor=scale_factor,
        ceiling_height_ft=10.0
    )

    # Expected: 15 x 20 = 300 sqft
    assert abs(room.floor_area_sqft - 300.0) < 1.0, f"Expected ~300 sqft, got {room.floor_area_sqft}"

    # Expected perimeter: 2*(15+20) = 70 ft
    assert abs(room.perimeter_ft - 70.0) < 1.0, f"Expected ~70 ft, got {room.perimeter_ft}"

    # Expected wall area: 70 * 10 = 700 sqft
    assert abs(room.wall_area_sqft - 700.0) < 5.0, f"Expected ~700 sqft, got {room.wall_area_sqft}"

    print("  [PASS] Realistic room values")
    return True


def run_all_tests():
    """Run all Phase 7 tests."""
    print("\n" + "=" * 60)
    print("Phase 7 Tests: Geometry Calculations")
    print("=" * 60)

    results = []

    # Room dataclass tests
    print("\nRoom Dataclass Tests:")
    results.append(test_room_dataclass())
    results.append(test_room_to_dict())
    results.append(test_room_to_csv_row())
    results.append(test_extract_floor_level())

    # Calculation tests
    print("\nCalculation Tests:")
    results.append(test_calculate_floor_area())
    results.append(test_calculate_perimeter())
    results.append(test_calculate_wall_area())
    results.append(test_calculate_ceiling_area())
    results.append(test_convert_polygon_to_real_units())

    # Validation tests
    print("\nValidation Tests:")
    results.append(test_validate_room_measurements())

    # Integration tests
    print("\nIntegration Tests:")
    results.append(test_calculate_all_room_measurements())
    results.append(test_create_room_from_polygon())
    results.append(test_room_with_realistic_values())

    # Summary
    passed = sum(results)
    total = len(results)
    print("\n" + "=" * 60)
    print(f"Phase 7 Results: {passed}/{total} tests passed")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
