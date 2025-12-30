"""
Phase 8 Tests: Output Generation

Tests for CSV, JSON, and PDF output generation.
"""

import sys
import os
import json
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from shapely.geometry import Polygon

from src.geometry.room import Room
from src.output import (
    write_rooms_to_csv,
    generate_csv_filename,
    build_room_json,
    build_output_json,
    write_rooms_to_json,
    generate_json_filename,
    get_polygon_centroid,
    generate_annotated_pdf_filename,
    PIPELINE_VERSION,
    DEFAULT_LAYER_NAME,
)
from src.constants import Confidence


def create_test_rooms():
    """Create test room objects."""
    rooms = []

    # Room 1
    room1 = Room(
        room_id="room_001",
        room_name="OFFICE 101",
        sheet_number=0,
        sheet_name="A1.1 - LEVEL 1",
        floor_level="LEVEL 1",
        floor_area_sqft=150.5,
        perimeter_ft=50.0,
        ceiling_height_ft=10.0,
        wall_area_sqft=500.0,
        ceiling_area_sqft=150.5,
        source="vector",
        confidence=Confidence.HIGH,
        polygon_pdf_points=[(100, 100), (200, 100), (200, 200), (100, 200)],
    )
    rooms.append(room1)

    # Room 2
    room2 = Room(
        room_id="room_002",
        room_name="CONFERENCE",
        sheet_number=0,
        sheet_name="A1.1 - LEVEL 1",
        floor_level="LEVEL 1",
        floor_area_sqft=300.0,
        perimeter_ft=70.0,
        ceiling_height_ft=12.0,
        wall_area_sqft=840.0,
        ceiling_area_sqft=300.0,
        source="vector",
        confidence=Confidence.MEDIUM,
        warnings=["Large room"],
    )
    rooms.append(room2)

    return rooms


def test_generate_csv_filename():
    """Test CSV filename generation."""
    result = generate_csv_filename("C:/Project/Plans.pdf", "C:/output")
    expected = str(Path("C:/output/Plans_rooms.csv"))
    assert result == expected, f"Expected {expected}, got {result}"

    print("  [PASS] CSV filename generation")
    return True


def test_write_rooms_to_csv():
    """Test CSV writing."""
    rooms = create_test_rooms()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_rooms.csv"

        result = write_rooms_to_csv(rooms, str(output_path))

        # Check file created
        assert output_path.exists(), "CSV file should be created"

        # Read and verify contents
        with open(output_path, "r") as f:
            lines = f.readlines()

        # Should have header + 2 rooms
        assert len(lines) == 3, f"Expected 3 lines, got {len(lines)}"

        # Check header
        header = lines[0].strip()
        assert "room_id" in header
        assert "floor_area_sqft" in header

        # Check first room
        data = lines[1].strip()
        assert "room_001" in data
        assert "OFFICE 101" in data

    print("  [PASS] CSV writing")
    return True


def test_generate_json_filename():
    """Test JSON filename generation."""
    result = generate_json_filename("C:/Project/Plans.pdf", "C:/output")
    expected = str(Path("C:/output/Plans_rooms.json"))
    assert result == expected

    print("  [PASS] JSON filename generation")
    return True


def test_build_room_json():
    """Test room to JSON conversion."""
    rooms = create_test_rooms()
    room = rooms[0]

    # Without geometry
    result = build_room_json(room, include_geometry=False)
    assert result["room_id"] == "room_001"
    assert result["floor_area_sqft"] == 150.5
    assert "geometry" not in result

    # With geometry
    result = build_room_json(room, include_geometry=True)
    assert "geometry" in result
    assert result["geometry"]["type"] == "polygon"

    print("  [PASS] Room to JSON")
    return True


def test_build_output_json():
    """Test complete output JSON structure."""
    rooms = create_test_rooms()

    result = build_output_json(
        rooms,
        input_file="Plans.pdf",
        scale_used="1/8 inch = 1 foot",
        units="imperial",
        include_geometry=True
    )

    # Check metadata
    assert "metadata" in result
    assert result["metadata"]["input_file"] == "Plans.pdf"
    assert result["metadata"]["pipeline_version"] == PIPELINE_VERSION
    assert result["metadata"]["total_rooms"] == 2
    assert result["metadata"]["total_floor_area_sqft"] == 450.5

    # Check rooms array
    assert "rooms" in result
    assert len(result["rooms"]) == 2

    # Check warnings collected
    assert len(result["metadata"]["warnings"]) == 1
    assert "Large room" in result["metadata"]["warnings"][0]

    print("  [PASS] Build output JSON")
    return True


def test_write_rooms_to_json():
    """Test JSON writing."""
    rooms = create_test_rooms()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_rooms.json"

        result = write_rooms_to_json(
            rooms, str(output_path),
            input_file="test.pdf",
            include_geometry=True
        )

        # Check file created
        assert output_path.exists(), "JSON file should be created"

        # Read and parse JSON
        with open(output_path, "r") as f:
            data = json.load(f)

        assert "metadata" in data
        assert "rooms" in data
        assert len(data["rooms"]) == 2

    print("  [PASS] JSON writing")
    return True


def test_get_polygon_centroid():
    """Test polygon centroid calculation."""
    # Square 0-100 in both dimensions
    vertices = [(0, 0), (100, 0), (100, 100), (0, 100)]
    cx, cy = get_polygon_centroid(vertices)

    assert abs(cx - 50) < 0.1, f"Expected cx=50, got {cx}"
    assert abs(cy - 50) < 0.1, f"Expected cy=50, got {cy}"

    # Empty list
    cx, cy = get_polygon_centroid([])
    assert cx == 0 and cy == 0

    print("  [PASS] Polygon centroid calculation")
    return True


def test_generate_annotated_pdf_filename():
    """Test annotated PDF filename generation."""
    result = generate_annotated_pdf_filename("C:/Project/Plans.pdf", "C:/output")
    expected = str(Path("C:/output/Plans_annotated.pdf"))
    assert result == expected

    print("  [PASS] Annotated PDF filename generation")
    return True


def test_csv_round_trip():
    """Test CSV write and read back."""
    rooms = create_test_rooms()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.csv"
        write_rooms_to_csv(rooms, str(output_path))

        import csv
        with open(output_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["room_id"] == "room_001"
        assert float(rows[0]["floor_area_sqft"]) == 150.5

    print("  [PASS] CSV round trip")
    return True


def test_json_round_trip():
    """Test JSON write and read back."""
    rooms = create_test_rooms()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.json"
        write_rooms_to_json(
            rooms, str(output_path),
            input_file="test.pdf",
            include_geometry=True
        )

        with open(output_path, "r") as f:
            data = json.load(f)

        assert data["metadata"]["total_rooms"] == 2
        assert data["rooms"][0]["room_id"] == "room_001"
        assert "geometry" in data["rooms"][0]

    print("  [PASS] JSON round trip")
    return True


def test_default_layer_name():
    """Test default layer name constant."""
    assert DEFAULT_LAYER_NAME == "Extracted Rooms"

    print("  [PASS] Default layer name")
    return True


def run_all_tests():
    """Run all Phase 8 tests."""
    print("\n" + "=" * 60)
    print("Phase 8 Tests: Output Generation")
    print("=" * 60)

    results = []

    # Filename generation tests
    print("\nFilename Generation Tests:")
    results.append(test_generate_csv_filename())
    results.append(test_generate_json_filename())
    results.append(test_generate_annotated_pdf_filename())

    # CSV tests
    print("\nCSV Output Tests:")
    results.append(test_write_rooms_to_csv())
    results.append(test_csv_round_trip())

    # JSON tests
    print("\nJSON Output Tests:")
    results.append(test_build_room_json())
    results.append(test_build_output_json())
    results.append(test_write_rooms_to_json())
    results.append(test_json_round_trip())

    # PDF annotation tests
    print("\nPDF Annotation Tests:")
    results.append(test_get_polygon_centroid())
    results.append(test_default_layer_name())

    # Summary
    passed = sum(results)
    total = len(results)
    print("\n" + "=" * 60)
    print(f"Phase 8 Results: {passed}/{total} tests passed")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
