#!/usr/bin/env python
"""
Phase 1 Tests: PDF Reading and Page Classification

Tests for:
- Opening and iterating PDF pages
- Page dimensions extraction
- Page type classification
- Processing path decision
- Page rendering to image
"""

import sys
import tempfile
from pathlib import Path

# Add project root to path so we can import src modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pymupdf

from src.pdf import (
    open_pdf,
    get_page_count,
    get_page,
    get_page_dimensions,
    render_page_to_image,
    classify_page,
    classify_document,
    determine_processing_path,
    PDFReadError,
    PDFPasswordProtectedError,
)
from src.constants import ProcessingPath, PageType


def create_test_pdf(filepath: str, page_type: str = "floor_plan") -> None:
    """Create a simple test PDF with specified characteristics."""
    doc = pymupdf.open()

    # Create a page (letter size: 612 x 792 points)
    page = doc.new_page(width=612, height=792)

    # Add text based on page type
    if page_type == "floor_plan":
        text = "LEVEL 1 FLOOR PLAN\n\nOFFICE 101\nCONFERENCE\nLOBBY"
    elif page_type == "rcp":
        text = "REFLECTED CEILING PLAN\n\nCLG 9'-0\"\nCLG 10'-0\""
    elif page_type == "elevation":
        text = "NORTH ELEVATION\n\nScale: 1/4\" = 1'-0\""
    elif page_type == "schedule":
        text = "ROOM SCHEDULE\n\nRoom | Area | Height"
    else:
        text = "GENERAL NOTES"

    # Insert text
    page.insert_text((72, 100), text, fontsize=12)

    # Add some drawing elements (lines) for vector detection
    shape = page.new_shape()
    # Draw a rectangle (4 lines)
    shape.draw_rect(pymupdf.Rect(100, 200, 500, 600))
    # Draw some internal lines (walls)
    for i in range(5):
        y = 200 + i * 80
        shape.draw_line((100, y), (500, y))
    for i in range(4):
        x = 100 + i * 100
        shape.draw_line((x, 200), (x, 600))

    shape.finish(width=1, color=(0, 0, 0))
    shape.commit()

    # Add more drawing elements to exceed MIN_DRAWINGS_FOR_VECTOR (50)
    for i in range(50):
        shape = page.new_shape()
        shape.draw_line((50 + i, 650), (50 + i, 680))
        shape.finish(width=0.5, color=(0.5, 0.5, 0.5))
        shape.commit()

    doc.save(filepath)
    doc.close()


def create_raster_test_pdf(filepath: str) -> None:
    """Create a test PDF with an embedded image (simulating scanned PDF)."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)

    # Create a simple image
    import io
    from PIL import Image

    img = Image.new('RGB', (400, 300), color='white')
    # Draw some lines on the image
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 350, 250], outline='black', width=2)
    draw.line([50, 150, 350, 150], fill='black', width=2)
    draw.line([200, 50, 200, 250], fill='black', width=2)

    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    # Insert image into PDF
    page.insert_image(pymupdf.Rect(100, 200, 500, 500), stream=img_bytes.read())

    # Add minimal text
    page.insert_text((72, 100), "SCANNED FLOOR PLAN", fontsize=12)

    doc.save(filepath)
    doc.close()


class TestPDFReader:
    """Tests for PDF reader functions."""

    def test_open_pdf_success(self):
        """Test opening a valid PDF."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            filepath = f.name

        try:
            create_test_pdf(filepath)
            doc = open_pdf(filepath)
            assert doc is not None
            assert doc.page_count > 0
            doc.close()
            print("  [PASS] open_pdf with valid file")
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_open_pdf_not_found(self):
        """Test opening a non-existent file."""
        try:
            open_pdf("/nonexistent/path/to/file.pdf")
            print("  [FAIL] Should have raised PDFReadError")
            return False
        except PDFReadError:
            print("  [PASS] open_pdf raises PDFReadError for missing file")
            return True

    def test_get_page_count(self):
        """Test getting page count."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            filepath = f.name

        try:
            create_test_pdf(filepath)
            doc = open_pdf(filepath)
            count = get_page_count(doc)
            assert count == 1, f"Expected 1 page, got {count}"
            doc.close()
            print("  [PASS] get_page_count")
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_get_page_dimensions(self):
        """Test getting page dimensions."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            filepath = f.name

        try:
            create_test_pdf(filepath)
            doc = open_pdf(filepath)
            page = get_page(doc, 0)
            width, height = get_page_dimensions(page)

            # Letter size: 612 x 792 points
            assert abs(width - 612) < 1, f"Expected width ~612, got {width}"
            assert abs(height - 792) < 1, f"Expected height ~792, got {height}"
            doc.close()
            print(f"  [PASS] get_page_dimensions: {width:.1f} x {height:.1f} points")
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_render_page_to_image(self):
        """Test rendering page to image."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            filepath = f.name

        try:
            create_test_pdf(filepath)
            doc = open_pdf(filepath)
            page = get_page(doc, 0)
            img = render_page_to_image(page, dpi=150)

            assert isinstance(img, np.ndarray), "Image should be numpy array"
            assert len(img.shape) == 3, "Image should have 3 dimensions"
            assert img.shape[2] == 3, "Image should have 3 channels (BGR)"
            assert img.shape[0] > 0 and img.shape[1] > 0, "Image should have non-zero size"

            # At 150 DPI, letter size should be roughly 1275 x 1650 pixels
            expected_width = int(612 * 150 / 72)
            expected_height = int(792 * 150 / 72)
            assert abs(img.shape[1] - expected_width) < 5, f"Width mismatch: {img.shape[1]} vs {expected_width}"
            assert abs(img.shape[0] - expected_height) < 5, f"Height mismatch: {img.shape[0]} vs {expected_height}"

            doc.close()
            print(f"  [PASS] render_page_to_image: {img.shape}")
        finally:
            Path(filepath).unlink(missing_ok=True)


class TestPageClassification:
    """Tests for page classification functions."""

    def test_classify_floor_plan(self):
        """Test classification of floor plan page."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            filepath = f.name

        try:
            create_test_pdf(filepath, "floor_plan")
            doc = open_pdf(filepath)
            page = get_page(doc, 0)
            classification = classify_page(page, 0)

            assert classification.page_type == PageType.FLOOR_PLAN, \
                f"Expected FLOOR_PLAN, got {classification.page_type}"
            doc.close()
            print(f"  [PASS] classify floor plan: {classification.page_type}")
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_classify_rcp(self):
        """Test classification of RCP page."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            filepath = f.name

        try:
            create_test_pdf(filepath, "rcp")
            doc = open_pdf(filepath)
            page = get_page(doc, 0)
            classification = classify_page(page, 0)

            assert classification.page_type == PageType.RCP, \
                f"Expected RCP, got {classification.page_type}"
            doc.close()
            print(f"  [PASS] classify RCP: {classification.page_type}")
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_classify_elevation(self):
        """Test classification of elevation page."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            filepath = f.name

        try:
            create_test_pdf(filepath, "elevation")
            doc = open_pdf(filepath)
            page = get_page(doc, 0)
            classification = classify_page(page, 0)

            assert classification.page_type == PageType.ELEVATION, \
                f"Expected ELEVATION, got {classification.page_type}"
            doc.close()
            print(f"  [PASS] classify elevation: {classification.page_type}")
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_classify_schedule(self):
        """Test classification of schedule page."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            filepath = f.name

        try:
            create_test_pdf(filepath, "schedule")
            doc = open_pdf(filepath)
            page = get_page(doc, 0)
            classification = classify_page(page, 0)

            assert classification.page_type == PageType.SCHEDULE, \
                f"Expected SCHEDULE, got {classification.page_type}"
            doc.close()
            print(f"  [PASS] classify schedule: {classification.page_type}")
        finally:
            Path(filepath).unlink(missing_ok=True)


class TestProcessingPath:
    """Tests for processing path determination."""

    def test_vector_path(self):
        """Test that vector path is determined for pages with many drawings."""
        # Many drawings, few images
        path = determine_processing_path(
            drawing_count=100,
            image_count=1,
            image_coverage_percent=5.0
        )
        assert path == ProcessingPath.VECTOR, f"Expected VECTOR, got {path}"
        print(f"  [PASS] vector path determination: {path}")

    def test_raster_path(self):
        """Test that raster path is determined for image-heavy pages."""
        # Few drawings, has images
        path = determine_processing_path(
            drawing_count=10,
            image_count=1,
            image_coverage_percent=50.0
        )
        assert path == ProcessingPath.RASTER, f"Expected RASTER, got {path}"
        print(f"  [PASS] raster path determination: {path}")

    def test_hybrid_path(self):
        """Test that hybrid path is determined for mixed content."""
        # Many drawings AND significant images
        path = determine_processing_path(
            drawing_count=100,
            image_count=5,
            image_coverage_percent=40.0
        )
        assert path == ProcessingPath.HYBRID, f"Expected HYBRID, got {path}"
        print(f"  [PASS] hybrid path determination: {path}")

    def test_skip_path(self):
        """Test that skip path is determined for text-only pages."""
        # No drawings, no images
        path = determine_processing_path(
            drawing_count=5,
            image_count=0,
            image_coverage_percent=0.0
        )
        assert path == ProcessingPath.SKIP, f"Expected SKIP, got {path}"
        print(f"  [PASS] skip path determination: {path}")

    def test_vector_page_classification(self):
        """Test that a vector PDF gets VECTOR processing path."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            filepath = f.name

        try:
            create_test_pdf(filepath, "floor_plan")
            doc = open_pdf(filepath)
            page = get_page(doc, 0)
            classification = classify_page(page, 0)

            # Our test PDF has 50+ drawings
            assert classification.processing_path == ProcessingPath.VECTOR, \
                f"Expected VECTOR, got {classification.processing_path} (drawings: {classification.drawing_count})"
            doc.close()
            print(f"  [PASS] vector page processing path: {classification.processing_path} "
                  f"({classification.drawing_count} drawings)")
        finally:
            Path(filepath).unlink(missing_ok=True)


def run_all_tests():
    """Run all Phase 1 tests."""
    print("=" * 60)
    print("Phase 1 Tests: PDF Reading and Page Classification")
    print("=" * 60)
    print()

    all_passed = True

    # PDF Reader Tests
    print("PDF Reader Tests:")
    print("-" * 40)
    reader_tests = TestPDFReader()
    try:
        reader_tests.test_open_pdf_success()
        reader_tests.test_open_pdf_not_found()
        reader_tests.test_get_page_count()
        reader_tests.test_get_page_dimensions()
        reader_tests.test_render_page_to_image()
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        all_passed = False
    except Exception as e:
        print(f"  [ERROR] {e}")
        all_passed = False

    print()

    # Page Classification Tests
    print("Page Classification Tests:")
    print("-" * 40)
    class_tests = TestPageClassification()
    try:
        class_tests.test_classify_floor_plan()
        class_tests.test_classify_rcp()
        class_tests.test_classify_elevation()
        class_tests.test_classify_schedule()
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        all_passed = False
    except Exception as e:
        print(f"  [ERROR] {e}")
        all_passed = False

    print()

    # Processing Path Tests
    print("Processing Path Tests:")
    print("-" * 40)
    path_tests = TestProcessingPath()
    try:
        path_tests.test_vector_path()
        path_tests.test_raster_path()
        path_tests.test_hybrid_path()
        path_tests.test_skip_path()
        path_tests.test_vector_page_classification()
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        all_passed = False
    except Exception as e:
        print(f"  [ERROR] {e}")
        all_passed = False

    print()
    print("=" * 60)
    if all_passed:
        print("ALL PHASE 1 TESTS PASSED")
        return 0
    else:
        print("SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
