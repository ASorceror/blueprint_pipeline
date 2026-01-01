"""
Test Hybrid Room Detector on Ellinwood floor plan.

This validates the complete label-driven detection pipeline.
"""

import sys
from pathlib import Path

# Add src to path for proper imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path.parent))

import pymupdf
import logging
from typing import List

from src.text.ocr_engine import extract_all_text
from src.pdf.reader import render_page_to_image
from src.vector.extractor import extract_wall_segments
from src.detection.hybrid_detector import (
    HybridRoomDetector,
    create_rooms_from_detection,
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_hybrid_detection():
    """Test hybrid detection on Ellinwood floor plan."""

    # Path to test file
    test_file = Path(r"C:\tb\blueprint_processor\output\classified_sheets\floor_plans\2018-1203_Ellinwood_GMP_Permit_A1.1E_LEVEL_01_FLOOR_PLAN.pdf")

    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return

    print(f"\n{'='*70}")
    print(f"TESTING HYBRID ROOM DETECTOR")
    print(f"File: {test_file.name}")
    print(f"{'='*70}\n")

    # Open PDF
    doc = pymupdf.open(test_file)
    page = doc[0]
    page_width = page.rect.width
    page_height = page.rect.height

    print(f"Page dimensions: {page_width:.1f} x {page_height:.1f} points")

    # Step 1: Render page image for OCR
    print("\n[Step 1] Rendering page for OCR...")
    page_image = render_page_to_image(page, dpi=150)
    print(f"Image shape: {page_image.shape}")

    # Step 2: Extract all text
    print("\n[Step 2] Extracting text (embedded + OCR)...")
    text_blocks = extract_all_text(page, page_image=page_image, force_ocr=True)
    print(f"Total text blocks: {len(text_blocks)}")

    # Step 3: Extract wall segments
    print("\n[Step 3] Extracting wall segments...")
    segments_raw = extract_wall_segments(page)
    print(f"Wall segments extracted: {len(segments_raw)}")

    # Convert to tuples format (Segment uses start/end tuples)
    wall_segments = []
    for seg in segments_raw:
        wall_segments.append((
            seg.start[0], seg.start[1], seg.end[0], seg.end[1],
            seg.width if hasattr(seg, 'width') else 1.0
        ))

    # Step 4: Initialize hybrid detector
    print("\n[Step 4] Initializing hybrid detector...")
    # Exclude right 15% for title block
    drawing_bounds = (0, 0, page_width * 0.85, page_height)

    detector = HybridRoomDetector(
        drawing_bounds=drawing_bounds,
        min_label_confidence=0.5,
        min_boundary_completeness=0.5,
        scale_factor=1.0  # Will be set properly later
    )

    # Step 5: Run detection
    print("\n[Step 5] Running hybrid detection...")
    print("-"*70)

    result = detector.detect_rooms(
        text_blocks=text_blocks,
        wall_segments=wall_segments,
        page_width=page_width,
        page_height=page_height
    )

    # Step 6: Display results
    print("\n" + "="*70)
    print("DETECTION RESULTS")
    print("="*70)

    print(f"\nSummary:")
    print(f"  Labels found: {result.labels_found}")
    print(f"  Labels with boundaries: {result.labels_with_boundaries}")
    print(f"  Rooms detected: {len(result.rooms)}")
    print(f"  Total area: {result.total_area_sqft:,.0f} SF (unscaled)")

    print(f"\nBoundary methods used:")
    for method, count in result.stats.get("boundary_methods", {}).items():
        print(f"  {method}: {count}")

    print(f"\nDetected Rooms:")
    print("-"*70)

    for i, room in enumerate(result.rooms):
        boundary_info = ""
        if room.boundary_result:
            boundary_info = f"completeness={room.boundary_result.completeness:.0%}"

        print(f"  {i+1:3d}. {room.room_name:15s}  "
              f"conf={room.confidence:.2f}  "
              f"area={room.area_sqft:,.0f} SF  "
              f"{boundary_info}")

    # Step 7: Convert to Room objects
    print("\n" + "="*70)
    print("CONVERTING TO ROOM OBJECTS")
    print("="*70)

    rooms = create_rooms_from_detection(
        detection_result=result,
        page_num=0,
        scale_factor=1.0,
        ceiling_height_ft=10.0
    )

    print(f"\nCreated {len(rooms)} Room objects")

    # Step 8: Statistics
    print("\n" + "="*70)
    print("COMPARISON STATISTICS")
    print("="*70)

    print(f"\nOld geometry-first approach:")
    print(f"  Polygons detected: 53")
    print(f"  Rooms with names: 22")
    print(f"  Garbage polygons: 31 (58%)")

    print(f"\nNew label-driven approach:")
    print(f"  Labels detected: {result.labels_found}")
    print(f"  Rooms created: {len(result.rooms)}")
    print(f"  High confidence (>=0.85): {sum(1 for r in result.rooms if r.confidence >= 0.85)}")
    print(f"  Medium confidence (0.6-0.85): {sum(1 for r in result.rooms if 0.6 <= r.confidence < 0.85)}")
    print(f"  Low confidence (<0.6): {sum(1 for r in result.rooms if r.confidence < 0.6)}")

    if result.warnings:
        print(f"\nWarnings:")
        for w in result.warnings[:10]:
            print(f"  - {w}")

    doc.close()

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)

    return result


if __name__ == "__main__":
    test_hybrid_detection()
