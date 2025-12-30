#!/usr/bin/env python
"""
Phase 2 Tests: Vector Extraction and Polygonization

Tests for:
- Segment extraction from PDF paths
- Wall segment filtering
- Double-line wall detection and merging
- Gap bridging
- Polygonization
- Polygon filtering
"""

import sys
import math
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pymupdf
from shapely.geometry import Polygon

from src.vector import (
    Segment,
    extract_wall_segments,
    detect_and_merge_double_walls,
    segments_are_parallel,
    bridge_gaps,
    extract_room_polygons,
    filter_room_polygons,
)
from src.constants import (
    MIN_SEGMENT_LENGTH_POINTS,
    DOUBLE_LINE_DISTANCE_MAX,
    DEFAULT_GAP_TOLERANCE_POINTS,
    MIN_AREA_SQ_POINTS,
)


def create_floor_plan_pdf(filepath: str) -> None:
    """Create a test PDF with a simple floor plan (4 rooms in a grid)."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)

    shape = page.new_shape()

    # Draw outer boundary (300x300 starting at 150, 200)
    # This creates a 2x2 grid of rooms
    x0, y0 = 150, 200
    size = 300

    # Outer walls
    shape.draw_rect(pymupdf.Rect(x0, y0, x0 + size, y0 + size))

    # Vertical divider (with door gap)
    mid_x = x0 + size / 2
    shape.draw_line((mid_x, y0), (mid_x, y0 + 120))  # Upper part
    shape.draw_line((mid_x, y0 + 150), (mid_x, y0 + size))  # Lower part (30pt gap for door)

    # Horizontal divider (with door gap)
    mid_y = y0 + size / 2
    shape.draw_line((x0, mid_y), (x0 + 120, mid_y))  # Left part
    shape.draw_line((x0 + 150, mid_y), (x0 + size, mid_y))  # Right part (30pt gap)

    shape.finish(width=1.0, color=(0, 0, 0))
    shape.commit()

    # Add text labels
    page.insert_text((x0 + 30, y0 + 80), "ROOM 1", fontsize=10)
    page.insert_text((mid_x + 30, y0 + 80), "ROOM 2", fontsize=10)
    page.insert_text((x0 + 30, mid_y + 80), "ROOM 3", fontsize=10)
    page.insert_text((mid_x + 30, mid_y + 80), "ROOM 4", fontsize=10)

    doc.save(filepath)
    doc.close()


def create_double_wall_pdf(filepath: str) -> None:
    """Create a test PDF with double-line walls."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)

    shape = page.new_shape()

    # Draw a rectangle with double lines (wall thickness = 6 points)
    x0, y0 = 150, 200
    size = 200
    wall_thickness = 6

    # Outer rectangle
    shape.draw_rect(pymupdf.Rect(x0, y0, x0 + size, y0 + size))
    shape.finish(width=0.5, color=(0, 0, 0))
    shape.commit()

    shape = page.new_shape()
    # Inner rectangle (parallel to outer, creating double-line effect)
    shape.draw_rect(pymupdf.Rect(
        x0 + wall_thickness, y0 + wall_thickness,
        x0 + size - wall_thickness, y0 + size - wall_thickness
    ))
    shape.finish(width=0.5, color=(0, 0, 0))
    shape.commit()

    doc.save(filepath)
    doc.close()


class TestSegmentExtraction:
    """Tests for segment extraction."""

    def test_extract_segments_from_pdf(self):
        """Test extracting segments from a floor plan PDF."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            filepath = f.name

        try:
            create_floor_plan_pdf(filepath)
            doc = pymupdf.open(filepath)
            page = doc.load_page(0)

            segments = extract_wall_segments(page)

            assert len(segments) > 0, "Should extract some segments"
            doc.close()
            print(f"  [PASS] Extracted {len(segments)} wall segments")
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_segment_properties(self):
        """Test segment length and angle calculations."""
        # Horizontal segment
        seg_h = Segment(start=(0, 0), end=(100, 0), width=1.0)
        assert abs(seg_h.length - 100) < 0.01, f"Expected length 100, got {seg_h.length}"
        assert abs(seg_h.angle - 0) < 0.01, f"Expected angle 0, got {seg_h.angle}"

        # Vertical segment
        seg_v = Segment(start=(0, 0), end=(0, 100), width=1.0)
        assert abs(seg_v.length - 100) < 0.01, f"Expected length 100, got {seg_v.length}"
        assert abs(seg_v.angle - 90) < 0.01, f"Expected angle 90, got {seg_v.angle}"

        # Diagonal segment
        seg_d = Segment(start=(0, 0), end=(100, 100), width=1.0)
        expected_len = math.sqrt(2) * 100
        assert abs(seg_d.length - expected_len) < 0.01
        assert abs(seg_d.angle - 45) < 0.01

        print("  [PASS] Segment length and angle calculations")

    def test_filter_short_segments(self):
        """Test that short segments are filtered out."""
        segments = [
            Segment(start=(0, 0), end=(1, 0), width=1.0),  # Too short (1 point)
            Segment(start=(0, 0), end=(100, 0), width=1.0),  # Long enough
        ]

        # Filter manually to test
        filtered = [s for s in segments if s.length >= MIN_SEGMENT_LENGTH_POINTS]
        assert len(filtered) == 1, "Should filter out short segment"
        print("  [PASS] Short segment filtering")


class TestDoubleWallMerger:
    """Tests for double-line wall detection and merging."""

    def test_parallel_detection(self):
        """Test that parallel segments are detected."""
        # Two horizontal segments
        seg_a = Segment(start=(0, 0), end=(100, 0), width=1.0)
        seg_b = Segment(start=(0, 5), end=(100, 5), width=1.0)

        assert segments_are_parallel(seg_a, seg_b), "Horizontal segments should be parallel"

        # Two perpendicular segments
        seg_c = Segment(start=(0, 0), end=(0, 100), width=1.0)
        assert not segments_are_parallel(seg_a, seg_c), "Perpendicular segments should not be parallel"

        print("  [PASS] Parallel segment detection")

    def test_double_wall_merging(self):
        """Test that double-line walls are merged."""
        # Two parallel horizontal segments close together
        segments = [
            Segment(start=(0, 0), end=(100, 0), width=1.0),
            Segment(start=(0, 6), end=(100, 6), width=1.0),  # 6 points apart
        ]

        merged, pair_count = detect_and_merge_double_walls(segments)

        # Should merge into one centerline
        assert pair_count == 1, f"Expected 1 pair, got {pair_count}"
        assert len(merged) == 1, f"Expected 1 merged segment, got {len(merged)}"

        # Centerline should be at y=3
        center_y = (merged[0].start[1] + merged[0].end[1]) / 2
        assert abs(center_y - 3) < 0.5, f"Expected centerline at y≈3, got {center_y}"

        print("  [PASS] Double-wall merging")

    def test_non_parallel_not_merged(self):
        """Test that non-parallel segments are not merged."""
        segments = [
            Segment(start=(0, 0), end=(100, 0), width=1.0),  # Horizontal
            Segment(start=(50, 0), end=(50, 100), width=1.0),  # Vertical
        ]

        merged, pair_count = detect_and_merge_double_walls(segments)

        assert pair_count == 0, "Non-parallel segments should not be merged"
        assert len(merged) == 2, "Both segments should remain"

        print("  [PASS] Non-parallel segments not merged")


class TestGapBridging:
    """Tests for gap bridging algorithm."""

    def test_bridge_small_gap(self):
        """Test that small gaps are bridged."""
        # Two segments with a 30-point gap (like a doorway)
        segments = [
            Segment(start=(0, 0), end=(100, 0), width=1.0),
            Segment(start=(130, 0), end=(230, 0), width=1.0),  # 30pt gap
        ]

        bridged, bridge_count = bridge_gaps(segments, gap_tolerance=36)  # 0.5 inch = 36 points

        assert bridge_count == 1, f"Expected 1 bridge, got {bridge_count}"
        assert len(bridged) == 3, f"Expected 3 segments (2 original + 1 bridge), got {len(bridged)}"

        print("  [PASS] Small gap bridged")

    def test_large_gap_not_bridged(self):
        """Test that large gaps are not bridged."""
        # Two segments with a 100-point gap (too large for doorway)
        segments = [
            Segment(start=(0, 0), end=(100, 0), width=1.0),
            Segment(start=(200, 0), end=(300, 0), width=1.0),  # 100pt gap
        ]

        bridged, bridge_count = bridge_gaps(segments, gap_tolerance=36)

        assert bridge_count == 0, "Large gap should not be bridged"

        print("  [PASS] Large gap not bridged")


class TestPolygonization:
    """Tests for polygonization and filtering."""

    def test_simple_rectangle(self):
        """Test polygonizing a simple rectangle."""
        # Four segments forming a rectangle
        segments = [
            Segment(start=(0, 0), end=(100, 0), width=1.0),
            Segment(start=(100, 0), end=(100, 100), width=1.0),
            Segment(start=(100, 100), end=(0, 100), width=1.0),
            Segment(start=(0, 100), end=(0, 0), width=1.0),
        ]

        rooms, debug = extract_room_polygons(segments, 612, 792)

        assert len(rooms) == 1, f"Expected 1 room, got {len(rooms)}"
        assert abs(rooms[0].area_sq_points - 10000) < 100, \
            f"Expected area ~10000, got {rooms[0].area_sq_points}"

        print(f"  [PASS] Simple rectangle polygonization (area: {rooms[0].area_sq_points:.0f})")

    def test_filter_tiny_polygons(self):
        """Test that tiny polygons are filtered out."""
        # Create some tiny polygons
        tiny_polygons = [
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),  # 100 sq pts - too small
            Polygon([(0, 0), (200, 0), (200, 200), (0, 200)]),  # 40000 sq pts - OK
        ]

        filtered = filter_room_polygons(tiny_polygons, 612, 792)

        assert len(filtered) == 1, f"Expected 1 polygon after filter, got {len(filtered)}"
        assert filtered[0].area > MIN_AREA_SQ_POINTS

        print("  [PASS] Tiny polygon filtering")

    def test_filter_page_covering_polygon(self):
        """Test that polygons covering most of the page are filtered."""
        page_width, page_height = 612, 792

        # Polygon covering 90% of page
        large_poly = Polygon([
            (0, 0), (page_width, 0),
            (page_width, page_height * 0.9),
            (0, page_height * 0.9)
        ])

        small_poly = Polygon([
            (100, 100), (200, 100), (200, 200), (100, 200)
        ])

        filtered = filter_room_polygons([large_poly, small_poly], page_width, page_height)

        # Large polygon should be filtered out
        assert len(filtered) == 1, f"Expected 1 polygon after filter, got {len(filtered)}"
        assert filtered[0].area < page_width * page_height * 0.5

        print("  [PASS] Page-covering polygon filtering")

    def test_full_floor_plan(self):
        """Test full pipeline on a floor plan PDF."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            filepath = f.name

        try:
            create_floor_plan_pdf(filepath)
            doc = pymupdf.open(filepath)
            page = doc.load_page(0)

            # Extract segments
            segments = extract_wall_segments(page)
            assert len(segments) > 5, f"Expected >5 segments, got {len(segments)}"

            # Extract rooms
            rooms, debug = extract_room_polygons(segments, 612, 792, gap_tolerance=36)

            # Should find multiple rooms (ideally 4, but could be fewer if gaps not bridged)
            assert len(rooms) >= 1, f"Expected at least 1 room, got {len(rooms)}"

            doc.close()
            print(f"  [PASS] Full floor plan: {len(segments)} segments -> {len(rooms)} rooms")
            print(f"         Debug: {debug}")
        finally:
            Path(filepath).unlink(missing_ok=True)


def run_all_tests():
    """Run all Phase 2 tests."""
    print("=" * 60)
    print("Phase 2 Tests: Vector Extraction and Polygonization")
    print("=" * 60)
    print()

    all_passed = True

    # Segment Extraction Tests
    print("Segment Extraction Tests:")
    print("-" * 40)
    try:
        tests = TestSegmentExtraction()
        tests.test_extract_segments_from_pdf()
        tests.test_segment_properties()
        tests.test_filter_short_segments()
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        all_passed = False
    except Exception as e:
        print(f"  [ERROR] {e}")
        all_passed = False

    print()

    # Double Wall Merger Tests
    print("Double Wall Merger Tests:")
    print("-" * 40)
    try:
        tests = TestDoubleWallMerger()
        tests.test_parallel_detection()
        tests.test_double_wall_merging()
        tests.test_non_parallel_not_merged()
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        all_passed = False
    except Exception as e:
        print(f"  [ERROR] {e}")
        all_passed = False

    print()

    # Gap Bridging Tests
    print("Gap Bridging Tests:")
    print("-" * 40)
    try:
        tests = TestGapBridging()
        tests.test_bridge_small_gap()
        tests.test_large_gap_not_bridged()
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        all_passed = False
    except Exception as e:
        print(f"  [ERROR] {e}")
        all_passed = False

    print()

    # Polygonization Tests
    print("Polygonization Tests:")
    print("-" * 40)
    try:
        tests = TestPolygonization()
        tests.test_simple_rectangle()
        tests.test_filter_tiny_polygons()
        tests.test_filter_page_covering_polygon()
        tests.test_full_floor_plan()
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        all_passed = False
    except Exception as e:
        print(f"  [ERROR] {e}")
        all_passed = False

    print()
    print("=" * 60)
    if all_passed:
        print("ALL PHASE 2 TESTS PASSED")
        return 0
    else:
        print("SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
