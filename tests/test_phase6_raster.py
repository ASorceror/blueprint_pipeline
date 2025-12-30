"""
Phase 6 Tests: Raster Processing (Scanned PDFs)

Tests for image preprocessing and room detection from raster images.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import cv2

from src.raster import (
    PreprocessingResult,
    to_grayscale,
    enhance_contrast,
    remove_noise,
    binarize,
    morphological_cleanup,
    detect_skew_angle,
    preprocess_image,
    RasterDetectionResult,
    transform_image_to_pdf_coords,
    simplify_contour,
    detect_rooms_connected_components,
    detect_rooms_contours,
    assess_quality,
    detect_rooms_from_raster,
)
from src.constants import Confidence


def create_test_image(width: int = 1000, height: int = 800) -> np.ndarray:
    """Create a test image with simple room-like rectangles."""
    # White background
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Draw black wall lines
    # Outer boundary
    cv2.rectangle(image, (50, 50), (950, 750), (0, 0, 0), 3)

    # Vertical wall in middle
    cv2.line(image, (500, 50), (500, 750), (0, 0, 0), 3)

    # Horizontal wall (partial for door)
    cv2.line(image, (50, 400), (450, 400), (0, 0, 0), 3)
    cv2.line(image, (550, 400), (950, 400), (0, 0, 0), 3)

    return image


def create_binary_test_image(width: int = 400, height: int = 300) -> np.ndarray:
    """Create a simple binary test image with rooms."""
    # Start with all white (rooms)
    binary = np.ones((height, width), dtype=np.uint8) * 255

    # Draw black walls
    # Outer walls
    cv2.rectangle(binary, (20, 20), (380, 280), 0, 5)

    # Vertical divider
    cv2.line(binary, (200, 20), (200, 280), 0, 5)

    return binary


def test_to_grayscale():
    """Test grayscale conversion."""
    # Create color image
    color = np.zeros((100, 100, 3), dtype=np.uint8)
    color[:, :, 0] = 128  # Blue
    color[:, :, 1] = 64   # Green
    color[:, :, 2] = 192  # Red

    gray = to_grayscale(color)

    assert len(gray.shape) == 2, "Should be 2D array"
    assert gray.shape == (100, 100), "Shape should match input"
    assert gray.dtype == np.uint8, "Should be uint8"

    # Test with already grayscale
    gray2 = to_grayscale(gray)
    assert np.array_equal(gray, gray2), "Should return same if already gray"

    print("  [PASS] Grayscale conversion")
    return True


def test_enhance_contrast():
    """Test CLAHE contrast enhancement."""
    # Create low contrast image with some variation
    gray = np.zeros((100, 100), dtype=np.uint8)
    gray[0:50, :] = 100
    gray[50:100, :] = 150  # Two distinct regions

    enhanced = enhance_contrast(gray)

    # Enhanced image should have same shape
    assert enhanced.shape == gray.shape, "Shape should match"
    assert enhanced.dtype == gray.dtype, "Dtype should match"

    # CLAHE should produce a valid image
    assert enhanced.min() >= 0, "Min should be >= 0"
    assert enhanced.max() <= 255, "Max should be <= 255"

    print("  [PASS] Contrast enhancement")
    return True


def test_remove_noise():
    """Test noise removal."""
    # Create noisy image
    gray = np.random.randint(100, 150, (100, 100), dtype=np.uint8)

    # Gaussian
    denoised = remove_noise(gray, method="gaussian")
    assert denoised.shape == gray.shape
    assert denoised.std() < gray.std(), "Denoised should have lower variance"

    # Bilateral
    denoised2 = remove_noise(gray, method="bilateral")
    assert denoised2.shape == gray.shape

    print("  [PASS] Noise removal")
    return True


def test_binarize():
    """Test adaptive binarization."""
    # Create gradient image
    gray = np.zeros((100, 100), dtype=np.uint8)
    for i in range(100):
        gray[i, :] = int(i * 2.5)

    binary = binarize(gray)

    assert binary.shape == gray.shape
    unique = np.unique(binary)
    assert len(unique) == 2, f"Should be binary, got {len(unique)} values"
    assert 0 in unique and 255 in unique, "Should have 0 and 255"

    print("  [PASS] Binarization")
    return True


def test_morphological_cleanup():
    """Test morphological operations."""
    # Create binary with small noise
    binary = np.zeros((100, 100), dtype=np.uint8)
    binary[30:70, 30:70] = 255  # Main square
    binary[10:12, 10:12] = 255  # Small noise

    cleaned = morphological_cleanup(binary, kernel_size=3)

    # Small noise should be removed by opening
    assert cleaned[10:12, 10:12].sum() < binary[10:12, 10:12].sum()
    # Main area should mostly remain
    assert cleaned[40:60, 40:60].sum() > 0

    print("  [PASS] Morphological cleanup")
    return True


def test_preprocess_image():
    """Test full preprocessing pipeline."""
    image = create_test_image()

    result = preprocess_image(image, dpi=150, correct_skew_flag=False)

    assert isinstance(result, PreprocessingResult)
    assert len(result.binary_image.shape) == 2, "Should be 2D binary"
    assert "grayscale" in result.steps_applied
    assert "clahe_contrast" in result.steps_applied
    assert "adaptive_threshold" in result.steps_applied
    assert result.dpi == 150

    print("  [PASS] Full preprocessing pipeline")
    return True


def test_transform_coords():
    """Test image to PDF coordinate transformation."""
    image_points = [(0, 0), (1000, 0), (1000, 800), (0, 800)]

    pdf_points = transform_image_to_pdf_coords(
        image_points,
        image_width=1000,
        image_height=800,
        page_width=612.0,
        page_height=792.0
    )

    # (0, 0) image -> (0, 792) PDF (top-left -> top in PDF coords)
    assert abs(pdf_points[0][0] - 0) < 0.1
    assert abs(pdf_points[0][1] - 792) < 0.1

    # (1000, 800) image -> (612, 0) PDF (bottom-right -> bottom-right in PDF)
    assert abs(pdf_points[2][0] - 612) < 0.1
    assert abs(pdf_points[2][1] - 0) < 0.1

    print("  [PASS] Coordinate transformation")
    return True


def test_simplify_contour():
    """Test contour simplification."""
    # Create a contour with many points (circle approximation)
    contour = np.array([[[int(50 + 30 * np.cos(t)), int(50 + 30 * np.sin(t))]]
                        for t in np.linspace(0, 2 * np.pi, 100)])
    contour = contour.astype(np.int32)

    simplified = simplify_contour(contour, dpi=300)

    assert len(simplified) < len(contour), "Should have fewer points"
    assert len(simplified) >= 4, "Should have at least 4 points"

    print("  [PASS] Contour simplification")
    return True


def test_detect_rooms_connected_components():
    """Test connected components room detection."""
    binary = create_binary_test_image()

    rooms = detect_rooms_connected_components(
        binary,
        page_width=612.0,
        page_height=792.0,
        dpi=150
    )

    # Should detect 2 rooms (left and right of divider)
    assert len(rooms) >= 1, f"Should detect at least 1 room, got {len(rooms)}"
    assert all(r.source == "raster" for r in rooms)
    assert all(r.confidence == Confidence.LOW for r in rooms)

    print(f"  [PASS] Connected components detection ({len(rooms)} rooms)")
    return True


def test_detect_rooms_contours():
    """Test contour-based room detection."""
    binary = create_binary_test_image()

    rooms = detect_rooms_contours(
        binary,
        page_width=612.0,
        page_height=792.0,
        dpi=150
    )

    assert len(rooms) >= 0, "Should not crash"
    assert all(r.source == "raster" for r in rooms)

    print(f"  [PASS] Contour detection ({len(rooms)} rooms)")
    return True


def test_assess_quality():
    """Test quality assessment."""
    from src.vector.polygonizer import RoomPolygon
    from shapely.geometry import Polygon

    page_area = 612 * 792

    # Test POOR (< 3 rooms)
    rooms_few = []
    assert assess_quality(rooms_few, page_area) == "POOR"

    # Test with good rooms
    good_rooms = []
    for i in range(5):
        poly = Polygon([(100 * i, 100), (100 * i + 50, 100), (100 * i + 50, 200), (100 * i, 200)])
        good_rooms.append(RoomPolygon(
            polygon_id=f"r{i}",
            vertices=list(poly.exterior.coords),
            area_sq_points=5000 + i * 100,
            shapely_polygon=poly
        ))

    quality = assess_quality(good_rooms, page_area)
    assert quality in ["GOOD", "SUSPECT"], f"Expected GOOD or SUSPECT, got {quality}"

    print("  [PASS] Quality assessment")
    return True


def test_detect_rooms_from_raster():
    """Test full raster detection pipeline."""
    binary = create_binary_test_image()

    result = detect_rooms_from_raster(
        binary,
        page_width=612.0,
        page_height=792.0,
        dpi=150
    )

    assert isinstance(result, RasterDetectionResult)
    assert result.method in ["flood_fill", "connected_components", "contours"]
    assert result.quality in ["GOOD", "SUSPECT", "POOR"]
    assert "room_count" in result.details

    print(f"  [PASS] Full raster pipeline (method={result.method}, quality={result.quality})")
    return True


def test_preprocessing_result_dataclass():
    """Test PreprocessingResult dataclass."""
    binary = np.zeros((100, 100), dtype=np.uint8)
    original = np.zeros((100, 100, 3), dtype=np.uint8)

    result = PreprocessingResult(
        binary_image=binary,
        original_image=original,
        steps_applied=["step1", "step2"],
        skew_angle=0.5,
        dpi=300
    )

    assert result.skew_angle == 0.5
    assert len(result.steps_applied) == 2

    print("  [PASS] PreprocessingResult dataclass")
    return True


def run_all_tests():
    """Run all Phase 6 tests."""
    print("\n" + "=" * 60)
    print("Phase 6 Tests: Raster Processing (Scanned PDFs)")
    print("=" * 60)

    results = []

    # Preprocessing tests
    print("\nImage Preprocessing Tests:")
    results.append(test_to_grayscale())
    results.append(test_enhance_contrast())
    results.append(test_remove_noise())
    results.append(test_binarize())
    results.append(test_morphological_cleanup())
    results.append(test_preprocess_image())

    # Room detection tests
    print("\nRoom Detection Tests:")
    results.append(test_transform_coords())
    results.append(test_simplify_contour())
    results.append(test_detect_rooms_connected_components())
    results.append(test_detect_rooms_contours())
    results.append(test_assess_quality())
    results.append(test_detect_rooms_from_raster())

    # Dataclass tests
    print("\nDataclass Tests:")
    results.append(test_preprocessing_result_dataclass())

    # Summary
    passed = sum(results)
    total = len(results)
    print("\n" + "=" * 60)
    print(f"Phase 6 Results: {passed}/{total} tests passed")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
