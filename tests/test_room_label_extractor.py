"""
Test Room Label Extractor on Ellinwood floor plan.

This validates Phase 4 of the label-driven detection pipeline.
"""

import sys
from pathlib import Path

# Add src to path for proper imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path.parent))

import pymupdf
import logging
from typing import List

from src.text.ocr_engine import extract_all_text, TextBlock
from src.text.room_label_extractor import (
    RoomLabelExtractor,
    ExtractedLabel,
    LabelType,
    extract_room_labels,
)
from src.pdf.reader import render_page_to_image

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_ellinwood_floor_plan():
    """Test label extraction on Ellinwood floor plan."""

    # Path to test file
    test_file = Path(r"C:\tb\blueprint_processor\output\classified_sheets\floor_plans\2018-1203_Ellinwood_GMP_Permit_A1.1E_LEVEL_01_FLOOR_PLAN.pdf")

    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return

    print(f"\n{'='*60}")
    print(f"TESTING ROOM LABEL EXTRACTOR")
    print(f"File: {test_file.name}")
    print(f"{'='*60}\n")

    # Open PDF
    doc = pymupdf.open(test_file)
    page = doc[0]
    page_width = page.rect.width
    page_height = page.rect.height

    print(f"Page dimensions: {page_width:.1f} x {page_height:.1f} points")

    # Render page image for OCR
    print("\nRendering page for OCR...")
    page_image = render_page_to_image(page, dpi=150)
    print(f"Image shape: {page_image.shape}")

    # Extract all text
    print("\nExtracting text (embedded + OCR)...")
    text_blocks = extract_all_text(page, page_image=page_image, force_ocr=True)
    print(f"Total text blocks: {len(text_blocks)}")

    # Sample of text blocks
    print("\nSample text blocks (first 20):")
    for i, block in enumerate(text_blocks[:20]):
        print(f"  {i+1:3d}. '{block.text}' at ({block.bbox[0]:.0f}, {block.bbox[1]:.0f}) conf={block.confidence:.2f}")

    # Initialize label extractor
    # Exclude right 15% for title block (based on Ellinwood layout)
    drawing_bounds = (0, 0, page_width * 0.85, page_height)
    extractor = RoomLabelExtractor(drawing_bounds=drawing_bounds, min_confidence=0.5)

    # Extract room labels
    print("\n" + "="*60)
    print("EXTRACTING ROOM LABELS")
    print("="*60)

    labels = extractor.extract(text_blocks, page_width, page_height)

    print(f"\nExtracted {len(labels)} room labels:")
    print("-"*60)

    # Group by type
    by_type = {}
    for label in labels:
        type_name = label.label_type.value
        if type_name not in by_type:
            by_type[type_name] = []
        by_type[type_name].append(label)

    for type_name, type_labels in sorted(by_type.items()):
        print(f"\n{type_name.upper()} ({len(type_labels)}):")
        for label in sorted(type_labels, key=lambda x: -x.confidence):
            print(f"  - {label.text:15s} conf={label.confidence:.2f} at ({label.centroid[0]:.0f}, {label.centroid[1]:.0f})")

    # Cluster labels
    print("\n" + "="*60)
    print("CLUSTERING LABELS")
    print("="*60)

    clusters = extractor.cluster_labels(labels, distance_threshold=100)
    print(f"\nFound {len(clusters)} label clusters:")

    for i, cluster in enumerate(clusters):
        print(f"\n  Cluster {i+1}: '{cluster.combined_text}'")
        print(f"    Primary: {cluster.primary_label.text} ({cluster.primary_label.label_type.value})")
        print(f"    Centroid: ({cluster.centroid[0]:.0f}, {cluster.centroid[1]:.0f})")
        if len(cluster.labels) > 1:
            print(f"    Labels: {[l.text for l in cluster.labels]}")

    # Statistics
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)

    print(f"\nText blocks extracted: {len(text_blocks)}")
    print(f"Room labels found: {len(labels)}")
    print(f"Label clusters: {len(clusters)}")
    print(f"Average confidence: {sum(l.confidence for l in labels)/len(labels) if labels else 0:.2f}")

    # By confidence
    high_conf = [l for l in labels if l.confidence >= 0.85]
    med_conf = [l for l in labels if 0.6 <= l.confidence < 0.85]
    low_conf = [l for l in labels if l.confidence < 0.6]

    print(f"\nConfidence distribution:")
    print(f"  HIGH (>=0.85): {len(high_conf)} labels")
    print(f"  MEDIUM (0.6-0.85): {len(med_conf)} labels")
    print(f"  LOW (<0.6): {len(low_conf)} labels")

    # Compare with old approach - get labels from old matcher
    print("\n" + "="*60)
    print("COMPARISON WITH GEOMETRY-FIRST APPROACH")
    print("="*60)

    # Old CSV showed 53 "rooms" with only 22 having actual room numbers
    # Let's count what we found
    room_numbers = [l for l in labels if l.label_type == LabelType.ROOM_NUMBER]
    area_tags = [l for l in labels if l.label_type == LabelType.AREA_TAG]
    room_names = [l for l in labels if l.label_type == LabelType.ROOM_NAME]

    print(f"\nOld approach: 53 polygons, 22 with room numbers, 31 garbage")
    print(f"New approach: {len(labels)} labels")
    print(f"  - Room numbers: {len(room_numbers)}")
    print(f"  - Area tags: {len(area_tags)}")
    print(f"  - Room names: {len(room_names)}")

    # List all found room numbers
    print(f"\nRoom numbers found:")
    for label in sorted(room_numbers, key=lambda x: x.text):
        print(f"  {label.text}")

    doc.close()

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

    return labels, clusters


if __name__ == "__main__":
    test_ellinwood_floor_plan()
