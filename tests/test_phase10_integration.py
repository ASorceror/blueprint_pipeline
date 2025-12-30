"""
Phase 10 Tests: Integration Testing and Final Validation

End-to-end integration tests for the blueprint pipeline.
Tests require sample PDF files to run full pipeline tests.
"""

import sys
import os
import tempfile
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from shapely.geometry import Polygon
import pymupdf

from src.constants import (
    Confidence,
    PageType,
    ProcessingPath,
    DEFAULT_CEILING_HEIGHT_FT,
    DEFAULT_RENDER_DPI,
    POINTS_PER_INCH,
)
from src.geometry.room import Room
from src.pipeline import (
    PipelineConfig,
    PipelineResult,
    get_pages_to_process,
    detect_scale,
)


# =============================================================================
# Unit Tests (No PDF required)
# =============================================================================

def test_pipeline_config_defaults():
    """Test PipelineConfig default values."""
    config = PipelineConfig(
        input_pdf="test.pdf",
        output_dir="./output"
    )

    assert config.pages == "all"
    assert config.units == "imperial"
    assert config.dpi == DEFAULT_RENDER_DPI
    assert config.default_height == DEFAULT_CEILING_HEIGHT_FT
    assert config.no_ocr is False
    assert config.no_annotate is False
    assert config.verbose is False
    assert config.debug is False

    print("  [PASS] Pipeline config defaults")
    return True


def test_pipeline_config_custom():
    """Test PipelineConfig with custom values."""
    config = PipelineConfig(
        input_pdf="test.pdf",
        output_dir="./output",
        pages="1-5",
        units="metric",
        manual_scale="1/4 inch = 1 foot",
        dpi=200,
        no_ocr=True,
        verbose=True
    )

    assert config.pages == "1-5"
    assert config.units == "metric"
    assert config.manual_scale == "1/4 inch = 1 foot"
    assert config.dpi == 200
    assert config.no_ocr is True
    assert config.verbose is True

    print("  [PASS] Pipeline config custom")
    return True


def test_pages_to_process_all():
    """Test get_pages_to_process with 'all'."""
    result = get_pages_to_process("all", 10)
    assert result == list(range(10))

    print("  [PASS] Pages to process 'all'")
    return True


def test_pages_to_process_range():
    """Test get_pages_to_process with range."""
    result = get_pages_to_process("3-7", 10)
    assert result == [2, 3, 4, 5, 6]  # 0-indexed

    print("  [PASS] Pages to process range")
    return True


def test_pages_to_process_list():
    """Test get_pages_to_process with comma-separated list."""
    result = get_pages_to_process("1,5,10", 15)
    assert result == [0, 4, 9]  # 0-indexed

    print("  [PASS] Pages to process list")
    return True


def test_pages_to_process_mixed():
    """Test get_pages_to_process with mixed specification."""
    result = get_pages_to_process("1-3,5,8-10", 15)
    assert result == [0, 1, 2, 4, 7, 8, 9]

    print("  [PASS] Pages to process mixed")
    return True


def test_room_to_dict():
    """Test Room to_dict method."""
    room = Room(
        room_id="room_001",
        room_name="OFFICE 101",
        sheet_number=0,
        sheet_name="A1.1",
        floor_level="LEVEL 1",
        floor_area_sqft=150.5,
        perimeter_ft=50.0,
        ceiling_height_ft=10.0,
        wall_area_sqft=500.0,
        ceiling_area_sqft=150.5,
        source="vector",
        confidence=Confidence.HIGH,
    )

    d = room.to_dict()

    assert d["room_id"] == "room_001"
    assert d["room_name"] == "OFFICE 101"
    assert d["floor_area_sqft"] == 150.5
    assert d["source"] == "vector"

    print("  [PASS] Room to_dict")
    return True


def test_room_to_csv_row():
    """Test Room to_csv_row method."""
    room = Room(
        room_id="room_001",
        room_name="OFFICE 101",
        sheet_number=0,
        floor_area_sqft=150.5,
        perimeter_ft=50.0,
        ceiling_height_ft=10.0,
        source="vector",
        confidence=Confidence.HIGH,
    )

    row = room.to_csv_row()

    assert row[0] == "room_001"
    assert row[1] == "OFFICE 101"
    assert row[5] == 150.5  # floor_area_sqft
    assert row[10] == "vector"  # source

    print("  [PASS] Room to_csv_row")
    return True


def test_pipeline_result_structure():
    """Test PipelineResult structure."""
    result = PipelineResult(
        input_file="test.pdf",
        output_dir="./output",
        total_pages=10,
        pages_processed=5,
        total_rooms=15,
        rooms=[],
        scale_used="1/8 inch = 1 foot",
        units="imperial",
        warnings=["test warning"],
        csv_path="test.csv",
        json_path="test.json",
        annotated_pdf_path="test_annotated.pdf",
        processing_time=5.5
    )

    assert result.input_file == "test.pdf"
    assert result.total_pages == 10
    assert result.pages_processed == 5
    assert result.total_rooms == 15
    assert len(result.warnings) == 1
    assert result.processing_time == 5.5

    print("  [PASS] Pipeline result structure")
    return True


# =============================================================================
# Module Integration Tests (Test component interactions)
# =============================================================================

def test_vector_extraction_pipeline():
    """Test vector extraction module integration."""
    from src.vector.extractor import Segment
    from src.vector.wall_merger import detect_and_merge_double_walls
    from src.vector.polygonizer import bridge_gaps, segments_to_polygons

    # Create test segments forming a rectangle
    segments = [
        Segment(start=(0, 0), end=(100, 0), width=1.0),
        Segment(start=(100, 0), end=(100, 100), width=1.0),
        Segment(start=(100, 100), end=(0, 100), width=1.0),
        Segment(start=(0, 100), end=(0, 0), width=1.0),
    ]

    # Merge (no double walls in this case)
    merged, merge_count = detect_and_merge_double_walls(segments)
    assert len(merged) == 4

    # Bridge gaps (none needed)
    bridged, bridge_count = bridge_gaps(merged, 36)
    assert len(bridged) == 4

    # Polygonize
    room_polygons, debug = segments_to_polygons(bridged, 200, 200)
    assert len(room_polygons) >= 1

    print("  [PASS] Vector extraction pipeline")
    return True


def test_calibration_pipeline():
    """Test calibration module integration."""
    from src.calibration.unit_converter import (
        pdf_points_to_real,
        area_points_to_real,
        calculate_scale_factor_from_calibration,
    )

    # Test scale factor calculation
    # 100 points distance = 10 feet
    scale = calculate_scale_factor_from_calibration(
        (0, 0), (100, 0), 10.0
    )
    assert abs(scale - 10.0) < 0.1  # 10 pts per foot

    # Test unit conversion
    real_dist = pdf_points_to_real(100, scale)
    assert abs(real_dist - 10.0) < 0.1

    # Test area conversion
    real_area = area_points_to_real(10000, scale)  # 100x100 pts
    assert abs(real_area - 100.0) < 1.0  # 10x10 ft

    print("  [PASS] Calibration pipeline")
    return True


def test_geometry_pipeline():
    """Test geometry module integration."""
    from src.geometry.calculator import (
        calculate_floor_area,
        calculate_perimeter,
        calculate_wall_area,
        create_room_from_polygon,
    )

    # Create a 10x10 foot room (at scale 72 pts/ft)
    poly = Polygon([
        (0, 0), (720, 0), (720, 720), (0, 720)
    ])

    scale_factor = 72.0  # 72 pts per foot

    # Test calculations
    area = calculate_floor_area(poly, scale_factor)
    assert abs(area - 100.0) < 0.1  # 10x10 = 100 sqft

    perimeter = calculate_perimeter(poly, scale_factor)
    assert abs(perimeter - 40.0) < 0.1  # 4*10 = 40 ft

    wall_area = calculate_wall_area(perimeter, 10.0)
    assert abs(wall_area - 400.0) < 0.1  # 40*10 = 400 sqft

    # Test room creation
    vertices = list(poly.exterior.coords)[:-1]
    room = create_room_from_polygon(
        polygon_id="test_001",
        polygon=poly,
        vertices=vertices,
        room_name="TEST ROOM",
        sheet_number=0,
        scale_factor=scale_factor,
        ceiling_height_ft=10.0,
    )

    assert room.room_id == "test_001"
    assert abs(room.floor_area_sqft - 100.0) < 0.1

    print("  [PASS] Geometry pipeline")
    return True


def test_output_pipeline():
    """Test output module integration."""
    from src.output.csv_writer import write_rooms_to_csv
    from src.output.json_writer import write_rooms_to_json, build_output_json

    rooms = [
        Room(
            room_id="room_001",
            room_name="OFFICE",
            sheet_number=0,
            floor_area_sqft=100.0,
            perimeter_ft=40.0,
            ceiling_height_ft=10.0,
            source="vector",
            confidence=Confidence.HIGH,
        )
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test CSV output
        csv_path = Path(tmpdir) / "test.csv"
        write_rooms_to_csv(rooms, str(csv_path))
        assert csv_path.exists()

        with open(csv_path) as f:
            lines = f.readlines()
        assert len(lines) == 2  # header + 1 room

        # Test JSON output
        json_path = Path(tmpdir) / "test.json"
        write_rooms_to_json(rooms, str(json_path), input_file="test.pdf")
        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)
        assert "metadata" in data
        assert "rooms" in data
        assert len(data["rooms"]) == 1

    print("  [PASS] Output pipeline")
    return True


# =============================================================================
# Full Pipeline Tests (Require sample PDFs)
# =============================================================================

def create_test_pdf(output_path: str) -> str:
    """Create a simple test PDF with vector graphics."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)  # Letter size

    # Draw a simple room (rectangle)
    shape = page.new_shape()

    # Room 1: 100x100 pts at position (100, 100)
    shape.draw_rect(pymupdf.Rect(100, 100, 200, 200))
    shape.finish(width=1, color=(0, 0, 0))

    # Room 2: 150x100 pts at position (250, 100)
    shape.draw_rect(pymupdf.Rect(250, 100, 400, 200))
    shape.finish(width=1, color=(0, 0, 0))

    shape.commit()

    # Add some text labels
    page.insert_text((125, 150), "OFFICE 101", fontsize=8)
    page.insert_text((290, 150), "CONF 102", fontsize=8)

    # Add scale text
    page.insert_text((50, 750), "SCALE: 1/8\" = 1'-0\"", fontsize=6)

    doc.save(output_path)
    doc.close()

    return output_path


def test_full_pipeline_with_test_pdf():
    """Test full pipeline with a generated test PDF."""
    from src.pipeline import run_pipeline
    import gc
    import time

    tmpdir = tempfile.mkdtemp()
    try:
        # Create test PDF
        pdf_path = str(Path(tmpdir) / "test_blueprint.pdf")
        create_test_pdf(pdf_path)

        # Create mock args
        @dataclass
        class MockArgs:
            input: str = pdf_path
            output: str = tmpdir
            pages: str = "all"
            units: str = "imperial"
            scale: Optional[str] = None
            calib: Optional[str] = None
            default_height: float = 10.0
            door_gap: float = 0.5
            dpi: int = 150
            no_ocr: bool = True  # Skip OCR for faster test
            no_annotate: bool = True  # Skip annotation to avoid file lock
            verbose: bool = False
            debug: bool = False

        args = MockArgs()

        # Run pipeline
        result = run_pipeline(args)

        # Verify result structure
        assert result.input_file == pdf_path
        assert result.total_pages == 1
        assert result.processing_time > 0

        # Check output files
        if result.csv_path:
            assert Path(result.csv_path).exists()
        if result.json_path:
            assert Path(result.json_path).exists()

        print("  [PASS] Full pipeline with test PDF")
        return True

    except Exception as e:
        print(f"  [FAIL] Full pipeline with test PDF: {e}")
        return False
    finally:
        # Cleanup - force garbage collection and wait
        gc.collect()
        time.sleep(0.1)
        try:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
        except:
            pass


def test_module_imports():
    """Test that all modules can be imported."""
    try:
        from src.pdf import reader, classifier
        from src.vector import extractor, wall_merger, polygonizer
        from src.calibration import scale_detector, unit_converter, dimension_associator
        from src.text import ocr_engine, label_matcher, height_parser
        from src.raster import preprocessor, room_detector
        from src.geometry import room, calculator
        from src.output import csv_writer, json_writer, pdf_annotator
        from src import cli, pipeline
        print("  [PASS] All module imports")
        return True
    except ImportError as e:
        print(f"  [FAIL] Module import error: {e}")
        return False


def test_constants_accessibility():
    """Test that all constants are accessible."""
    from src.constants import (
        # Vector constants
        MIN_DRAWINGS_FOR_VECTOR,
        MIN_SEGMENT_LENGTH_POINTS,
        MAX_SEGMENT_LENGTH_RATIO,
        DOUBLE_LINE_DISTANCE_MAX,
        # Gap bridging
        DEFAULT_GAP_TOLERANCE_POINTS,
        # Polygon filtering
        MIN_AREA_SQ_POINTS,
        MIN_VERTICES,
        MAX_AREA_PAGE_RATIO,
        # Text
        ROOM_LABEL_MIN_FONTSIZE,
        OCR_CONFIDENCE_THRESHOLD,
        # Raster
        DEFAULT_RENDER_DPI,
        MAX_SKEW_CORRECTION_DEG,
        # Scale
        SCALE_CONFLICT_THRESHOLD_PERCENT,
        DEFAULT_COMMERCIAL_SCALE,
        # Heights
        DEFAULT_CEILING_HEIGHT_FT,
        MIN_CEILING_HEIGHT_FT,
        MAX_CEILING_HEIGHT_FT,
        # Enums
        Confidence,
        ProcessingPath,
        PageType,
    )

    # Verify some key values
    assert DEFAULT_RENDER_DPI == 300
    assert DEFAULT_CEILING_HEIGHT_FT == 10.0
    assert MIN_CEILING_HEIGHT_FT == 7
    assert MAX_CEILING_HEIGHT_FT == 60

    print("  [PASS] Constants accessibility")
    return True


# =============================================================================
# Run All Tests
# =============================================================================

def run_all_tests():
    """Run all Phase 10 tests."""
    print("\n" + "=" * 60)
    print("Phase 10 Tests: Integration Testing and Final Validation")
    print("=" * 60)

    results = []

    # Unit tests
    print("\nUnit Tests:")
    results.append(test_pipeline_config_defaults())
    results.append(test_pipeline_config_custom())
    results.append(test_pages_to_process_all())
    results.append(test_pages_to_process_range())
    results.append(test_pages_to_process_list())
    results.append(test_pages_to_process_mixed())
    results.append(test_room_to_dict())
    results.append(test_room_to_csv_row())
    results.append(test_pipeline_result_structure())

    # Module integration tests
    print("\nModule Integration Tests:")
    results.append(test_vector_extraction_pipeline())
    results.append(test_calibration_pipeline())
    results.append(test_geometry_pipeline())
    results.append(test_output_pipeline())

    # Full pipeline tests
    print("\nFull Pipeline Tests:")
    results.append(test_module_imports())
    results.append(test_constants_accessibility())
    results.append(test_full_pipeline_with_test_pdf())

    # Summary
    passed = sum(results)
    total = len(results)
    print("\n" + "=" * 60)
    print(f"Phase 10 Results: {passed}/{total} tests passed")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
