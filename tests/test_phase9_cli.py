"""
Phase 9 Tests: CLI and Pipeline Orchestration

Tests for command-line interface and pipeline integration.
"""

import sys
import os
import tempfile
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cli import (
    parse_page_range,
    create_parser,
    validate_args,
)
from src.pipeline import (
    PipelineConfig,
    PageResult,
    PipelineResult,
    get_pages_to_process,
)
from src.constants import (
    DEFAULT_RENDER_DPI,
    DEFAULT_GAP_TOLERANCE_INCHES,
    DEFAULT_CEILING_HEIGHT_FT,
)


def test_parse_page_range_all():
    """Test parsing 'all' page range."""
    result = parse_page_range("all", 10)
    expected = list(range(10))
    assert result == expected, f"Expected {expected}, got {result}"

    print("  [PASS] Parse page range 'all'")
    return True


def test_parse_page_range_single():
    """Test parsing single page."""
    result = parse_page_range("3", 10)
    expected = [2]  # 0-indexed
    assert result == expected, f"Expected {expected}, got {result}"

    print("  [PASS] Parse page range single")
    return True


def test_parse_page_range_range():
    """Test parsing page range."""
    result = parse_page_range("2-5", 10)
    expected = [1, 2, 3, 4]  # 0-indexed
    assert result == expected, f"Expected {expected}, got {result}"

    print("  [PASS] Parse page range range")
    return True


def test_parse_page_range_list():
    """Test parsing comma-separated list."""
    result = parse_page_range("1,3,5", 10)
    expected = [0, 2, 4]  # 0-indexed
    assert result == expected, f"Expected {expected}, got {result}"

    print("  [PASS] Parse page range list")
    return True


def test_parse_page_range_mixed():
    """Test parsing mixed range and list."""
    result = parse_page_range("1-3,5,7-8", 10)
    expected = [0, 1, 2, 4, 6, 7]  # 0-indexed
    assert result == expected, f"Expected {expected}, got {result}"

    print("  [PASS] Parse page range mixed")
    return True


def test_parse_page_range_out_of_bounds():
    """Test page range filtering for out of bounds."""
    result = parse_page_range("1,5,15", 10)
    expected = [0, 4]  # 15 is out of bounds
    assert result == expected, f"Expected {expected}, got {result}"

    print("  [PASS] Parse page range out of bounds")
    return True


def test_create_parser():
    """Test parser creation."""
    parser = create_parser()

    assert parser is not None
    assert parser.prog == "blueprint_pipeline"

    print("  [PASS] Create parser")
    return True


def test_parser_required_args():
    """Test parser required arguments."""
    parser = create_parser()

    # Should fail without required args
    try:
        parser.parse_args([])
        assert False, "Should have raised error"
    except SystemExit:
        pass  # Expected

    print("  [PASS] Parser required args")
    return True


def test_parser_full_args():
    """Test parser with all arguments."""
    parser = create_parser()

    args = parser.parse_args([
        "-i", "test.pdf",
        "-o", "./output",
        "--pages", "1-5",
        "--units", "metric",
        "--scale", "1/4 inch = 1 foot",
        "--default-height", "12.0",
        "--door-gap", "0.75",
        "--dpi", "200",
        "--no-ocr",
        "--no-annotate",
        "-v",
        "--debug"
    ])

    assert args.input == "test.pdf"
    assert args.output == "./output"
    assert args.pages == "1-5"
    assert args.units == "metric"
    assert args.scale == "1/4 inch = 1 foot"
    assert args.default_height == 12.0
    assert args.door_gap == 0.75
    assert args.dpi == 200
    assert args.no_ocr is True
    assert args.no_annotate is True
    assert args.verbose is True
    assert args.debug is True

    print("  [PASS] Parser full args")
    return True


def test_parser_defaults():
    """Test parser default values."""
    parser = create_parser()

    args = parser.parse_args(["-i", "test.pdf", "-o", "./output"])

    assert args.pages == "all"
    assert args.units == "imperial"
    assert args.scale is None
    assert args.default_height == DEFAULT_CEILING_HEIGHT_FT
    assert args.door_gap == DEFAULT_GAP_TOLERANCE_INCHES
    assert args.dpi == DEFAULT_RENDER_DPI
    assert args.no_ocr is False
    assert args.no_annotate is False
    assert args.verbose is False
    assert args.debug is False

    print("  [PASS] Parser defaults")
    return True


def test_validate_args_missing_input():
    """Test validation catches missing input file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        args = argparse.Namespace(
            input="nonexistent.pdf",
            output=tmpdir,
            dpi=300,
            default_height=10.0
        )

        is_valid, error = validate_args(args)
        assert is_valid is False
        assert "not found" in error.lower()

    print("  [PASS] Validate missing input")
    return True


def test_validate_args_bad_extension():
    """Test validation catches non-PDF input."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a non-PDF file
        test_file = Path(tmpdir) / "test.txt"
        test_file.write_text("not a pdf")

        args = argparse.Namespace(
            input=str(test_file),
            output=tmpdir,
            dpi=300,
            default_height=10.0
        )

        is_valid, error = validate_args(args)
        assert is_valid is False
        assert "pdf" in error.lower()

    print("  [PASS] Validate bad extension")
    return True


def test_validate_args_dpi_range():
    """Test validation catches invalid DPI."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test PDF
        test_file = Path(tmpdir) / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 test")

        # Too low
        args = argparse.Namespace(
            input=str(test_file),
            output=tmpdir,
            dpi=50,
            default_height=10.0
        )
        is_valid, error = validate_args(args)
        assert is_valid is False
        assert "dpi" in error.lower()

        # Too high
        args.dpi = 1000
        is_valid, error = validate_args(args)
        assert is_valid is False
        assert "dpi" in error.lower()

    print("  [PASS] Validate DPI range")
    return True


def test_validate_args_height_range():
    """Test validation catches invalid height."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 test")

        # Too low
        args = argparse.Namespace(
            input=str(test_file),
            output=tmpdir,
            dpi=300,
            default_height=3.0
        )
        is_valid, error = validate_args(args)
        assert is_valid is False
        assert "height" in error.lower()

        # Too high
        args.default_height = 150.0
        is_valid, error = validate_args(args)
        assert is_valid is False
        assert "height" in error.lower()

    print("  [PASS] Validate height range")
    return True


def test_pipeline_config():
    """Test PipelineConfig dataclass."""
    config = PipelineConfig(
        input_pdf="test.pdf",
        output_dir="./output"
    )

    assert config.input_pdf == "test.pdf"
    assert config.output_dir == "./output"
    assert config.pages == "all"
    assert config.dpi == DEFAULT_RENDER_DPI
    assert config.no_ocr is False

    print("  [PASS] PipelineConfig dataclass")
    return True


def test_pipeline_result():
    """Test PipelineResult dataclass."""
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

    assert result.total_pages == 10
    assert result.pages_processed == 5
    assert result.total_rooms == 15
    assert result.processing_time == 5.5

    print("  [PASS] PipelineResult dataclass")
    return True


def test_get_pages_to_process():
    """Test get_pages_to_process function."""
    # All pages
    result = get_pages_to_process("all", 5)
    assert result == [0, 1, 2, 3, 4]

    # Range
    result = get_pages_to_process("2-4", 10)
    assert result == [1, 2, 3]

    # List
    result = get_pages_to_process("1,3,5", 10)
    assert result == [0, 2, 4]

    print("  [PASS] get_pages_to_process")
    return True


def run_all_tests():
    """Run all Phase 9 tests."""
    print("\n" + "=" * 60)
    print("Phase 9 Tests: CLI and Pipeline Orchestration")
    print("=" * 60)

    results = []

    # Page range parsing tests
    print("\nPage Range Parsing Tests:")
    results.append(test_parse_page_range_all())
    results.append(test_parse_page_range_single())
    results.append(test_parse_page_range_range())
    results.append(test_parse_page_range_list())
    results.append(test_parse_page_range_mixed())
    results.append(test_parse_page_range_out_of_bounds())

    # Parser tests
    print("\nArgument Parser Tests:")
    results.append(test_create_parser())
    results.append(test_parser_required_args())
    results.append(test_parser_full_args())
    results.append(test_parser_defaults())

    # Validation tests
    print("\nValidation Tests:")
    results.append(test_validate_args_missing_input())
    results.append(test_validate_args_bad_extension())
    results.append(test_validate_args_dpi_range())
    results.append(test_validate_args_height_range())

    # Pipeline dataclass tests
    print("\nPipeline Dataclass Tests:")
    results.append(test_pipeline_config())
    results.append(test_pipeline_result())
    results.append(test_get_pages_to_process())

    # Summary
    passed = sum(results)
    total = len(results)
    print("\n" + "=" * 60)
    print(f"Phase 9 Results: {passed}/{total} tests passed")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
