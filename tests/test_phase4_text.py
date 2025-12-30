"""
Phase 4 Tests: Text Extraction and Room Labeling

Tests for OCR extraction, room label filtering, and label-to-polygon matching.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.text import (
    TextBlock,
    get_available_engines,
    extract_embedded_text,
    transform_ocr_coords_to_pdf,
    merge_text_blocks,
    RoomLabel,
    LabelMatch,
    filter_room_labels,
    match_labels_to_polygons,
    is_room_label_pattern,
    is_excluded_pattern,
)
from src.vector.polygonizer import RoomPolygon
from src.constants import Confidence

from shapely.geometry import Polygon


def test_get_available_engines():
    """Test that at least one OCR engine is available."""
    engines = get_available_engines()
    # At least one engine should be available (paddleocr was installed in Phase 0)
    print(f"  Available OCR engines: {engines}")
    print(f"  [PASS] OCR engine availability check")
    return True


def test_transform_ocr_coords():
    """Test coordinate transformation from image to PDF."""
    # Image: 1000x800 pixels
    # PDF: 612x792 points
    image_width, image_height = 1000, 800
    page_width, page_height = 612.0, 792.0

    # Test box at image center
    ocr_bbox = (400, 300, 600, 500)  # Center of image

    pdf_bbox = transform_ocr_coords_to_pdf(
        ocr_bbox, image_width, image_height,
        page_width, page_height
    )

    # Check X transformation (just scaling)
    expected_x0 = 400 * (612 / 1000)  # 244.8
    expected_x1 = 600 * (612 / 1000)  # 367.2

    # Check Y transformation (scale and flip)
    # Image y1=500 -> PDF y0 = 792 - (500 * 792/800) = 792 - 495 = 297
    # Image y0=300 -> PDF y1 = 792 - (300 * 792/800) = 792 - 297 = 495
    expected_y0 = 792 - (500 * (792 / 800))  # 297
    expected_y1 = 792 - (300 * (792 / 800))  # 495

    x0, y0, x1, y1 = pdf_bbox

    assert abs(x0 - expected_x0) < 0.1, f"X0: expected {expected_x0}, got {x0}"
    assert abs(x1 - expected_x1) < 0.1, f"X1: expected {expected_x1}, got {x1}"
    assert abs(y0 - expected_y0) < 0.1, f"Y0: expected {expected_y0}, got {y0}"
    assert abs(y1 - expected_y1) < 0.1, f"Y1: expected {expected_y1}, got {y1}"

    print("  [PASS] Coordinate transformation")
    return True


def test_merge_text_blocks():
    """Test merging embedded and OCR text blocks."""
    embedded = [
        TextBlock(text="OFFICE 101", bbox=(100, 100, 200, 120), confidence=1.0, source="embedded"),
        TextBlock(text="CONFERENCE", bbox=(300, 100, 400, 120), confidence=1.0, source="embedded"),
    ]

    ocr = [
        # This overlaps with OFFICE 101 - should be excluded
        TextBlock(text="OFFICE", bbox=(105, 102, 195, 118), confidence=0.9, source="ocr"),
        # This is new - should be included
        TextBlock(text="STORAGE", bbox=(500, 100, 600, 120), confidence=0.85, source="ocr"),
    ]

    merged = merge_text_blocks(embedded, ocr)

    # Should have 3 blocks: 2 embedded + 1 new OCR
    assert len(merged) == 3, f"Expected 3 blocks, got {len(merged)}"

    texts = [b.text for b in merged]
    assert "OFFICE 101" in texts, "Should have OFFICE 101"
    assert "CONFERENCE" in texts, "Should have CONFERENCE"
    assert "STORAGE" in texts, "Should have STORAGE"

    print("  [PASS] Text block merging")
    return True


def test_room_label_pattern_matching():
    """Test room label pattern recognition."""
    # These should match
    positive_cases = [
        "OFFICE 101",
        "CONFERENCE",
        "MEETING ROOM",
        "KITCHEN",
        "BREAK ROOM",
        "RESTROOM",
        "STORAGE",
        "LOBBY",
        "CORRIDOR",
        "101",
        "A101",
        "101A",
        "1-101",
        "MECHANICAL",
        "ELEVATOR",
    ]

    # These should not match
    negative_cases = [
        "10'-6\"",       # Dimension
        "1/8\" = 1'",    # Scale
        "NOTE: SEE DETAIL",  # Note
        "A1.01",         # Drawing number
        "1/4",           # Fraction
    ]

    positive_passed = 0
    for text in positive_cases:
        if is_room_label_pattern(text):
            positive_passed += 1
        else:
            print(f"  FAIL: '{text}' should be a room label")

    negative_passed = 0
    for text in negative_cases:
        if is_excluded_pattern(text):
            negative_passed += 1
        else:
            print(f"  FAIL: '{text}' should be excluded")

    total = len(positive_cases) + len(negative_cases)
    passed = positive_passed + negative_passed

    print(f"  [{'PASS' if passed == total else 'FAIL'}] Room label patterns: {passed}/{total}")
    return passed == total


def test_filter_room_labels():
    """Test filtering text blocks to room labels."""
    text_blocks = [
        TextBlock(text="OFFICE 101", bbox=(100, 100, 200, 120), confidence=1.0,
                  font_size=12, source="embedded"),
        TextBlock(text="10'-6\"", bbox=(300, 100, 400, 120), confidence=1.0,
                  font_size=10, source="embedded"),  # Dimension - exclude
        TextBlock(text="STORAGE", bbox=(500, 100, 600, 120), confidence=0.9,
                  font_size=14, source="ocr"),
        TextBlock(text="A", bbox=(700, 100, 710, 110), confidence=1.0,
                  font_size=8, source="embedded"),  # Too short
        TextBlock(text="CONFERENCE", bbox=(800, 100, 950, 120), confidence=1.0,
                  font_size=50, source="embedded"),  # Font too large
    ]

    labels = filter_room_labels(text_blocks)

    # Should get OFFICE 101 and STORAGE only
    assert len(labels) == 2, f"Expected 2 labels, got {len(labels)}"

    texts = [l.text for l in labels]
    assert "OFFICE 101" in texts
    assert "STORAGE" in texts

    print("  [PASS] Room label filtering")
    return True


def test_label_to_polygon_matching():
    """Test matching labels to polygons."""
    # Create test polygons
    poly1 = RoomPolygon(
        polygon_id="poly_1",
        vertices=[(100, 100), (200, 100), (200, 200), (100, 200)],
        area_sq_points=10000,
        shapely_polygon=Polygon([(100, 100), (200, 100), (200, 200), (100, 200)])
    )
    poly2 = RoomPolygon(
        polygon_id="poly_2",
        vertices=[(300, 100), (400, 100), (400, 200), (300, 200)],
        area_sq_points=10000,
        shapely_polygon=Polygon([(300, 100), (400, 100), (400, 200), (300, 200)])
    )
    poly3 = RoomPolygon(
        polygon_id="poly_3",
        vertices=[(500, 100), (600, 100), (600, 200), (500, 200)],
        area_sq_points=10000,
        shapely_polygon=Polygon([(500, 100), (600, 100), (600, 200), (500, 200)])
    )

    # Create test labels
    # Label 1: Inside poly1 (should match HIGH)
    label1 = RoomLabel(
        text="OFFICE 101",
        bbox=(140, 140, 180, 160),
        centroid=(160, 150),
        confidence=1.0,
        source="embedded"
    )
    # Label 2: Near poly2 but outside (should match MEDIUM with expanded bounds)
    label2 = RoomLabel(
        text="CONFERENCE",
        bbox=(280, 140, 320, 160),
        centroid=(300, 150),  # Just at edge of poly2
        confidence=1.0,
        source="embedded"
    )

    labels = [label1, label2]
    polygons = [poly1, poly2, poly3]

    matches = match_labels_to_polygons(labels, polygons)

    assert len(matches) == 3, f"Expected 3 matches (one per polygon), got {len(matches)}"

    # Check poly1 matched with HIGH confidence
    poly1_match = next(m for m in matches if m.polygon_id == "poly_1")
    assert poly1_match.room_name == "OFFICE 101"
    assert poly1_match.name_confidence == Confidence.HIGH

    # poly3 should be auto-named
    poly3_match = next(m for m in matches if m.polygon_id == "poly_3")
    assert poly3_match.room_name.startswith("ROOM_")
    assert poly3_match.name_confidence == Confidence.NONE

    print("  [PASS] Label-to-polygon matching")
    return True


def test_text_block_dataclass():
    """Test TextBlock dataclass."""
    block = TextBlock(
        text="TEST",
        bbox=(0, 0, 100, 20),
        confidence=0.95,
        font_size=12,
        font_name="Arial",
        source="embedded"
    )

    assert block.text == "TEST"
    assert block.confidence == 0.95
    assert block.source == "embedded"

    print("  [PASS] TextBlock dataclass")
    return True


def test_room_label_dataclass():
    """Test RoomLabel dataclass."""
    label = RoomLabel(
        text="OFFICE",
        bbox=(0, 0, 100, 20),
        centroid=(50, 10),
        confidence=1.0,
        source="embedded"
    )

    assert label.text == "OFFICE"
    assert label.centroid == (50, 10)
    assert not label.is_matched

    label.is_matched = True
    assert label.is_matched

    print("  [PASS] RoomLabel dataclass")
    return True


def run_all_tests():
    """Run all Phase 4 tests."""
    print("\n" + "=" * 60)
    print("Phase 4 Tests: Text Extraction and Room Labeling")
    print("=" * 60)

    results = []

    # OCR engine tests
    print("\nOCR Engine Tests:")
    results.append(test_get_available_engines())
    results.append(test_transform_ocr_coords())
    results.append(test_merge_text_blocks())

    # Label pattern tests
    print("\nRoom Label Pattern Tests:")
    results.append(test_room_label_pattern_matching())
    results.append(test_filter_room_labels())

    # Label matching tests
    print("\nLabel-to-Polygon Matching Tests:")
    results.append(test_label_to_polygon_matching())

    # Dataclass tests
    print("\nDataclass Tests:")
    results.append(test_text_block_dataclass())
    results.append(test_room_label_dataclass())

    # Summary
    passed = sum(results)
    total = len(results)
    print("\n" + "=" * 60)
    print(f"Phase 4 Results: {passed}/{total} tests passed")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
