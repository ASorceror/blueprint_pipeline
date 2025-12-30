"""
PDF Page Classifier Module

Functions for classifying PDF page types and determining processing paths.
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pymupdf

from ..constants import (
    MIN_DRAWINGS_FOR_VECTOR,
    MAX_IMAGES_FOR_VECTOR,
    MAX_IMAGE_COVERAGE_PERCENT,
    PAGE_TYPE_KEYWORDS,
    PageType,
    ProcessingPath,
)

logger = logging.getLogger(__name__)


@dataclass
class PageClassification:
    """Result of page classification."""
    page_number: int
    page_type: str
    processing_path: str
    width_points: float
    height_points: float
    drawing_count: int
    image_count: int
    image_coverage_percent: float
    detected_keywords: List[str]


def count_drawing_objects(page: pymupdf.Page) -> int:
    """
    Count the number of drawing objects (paths) on a page.

    Args:
        page: pymupdf.Page object

    Returns:
        Count of drawing objects
    """
    try:
        drawings = page.get_drawings()
        return len(drawings)
    except Exception as e:
        logger.warning(f"Error counting drawings: {e}")
        return 0


def get_image_info(page: pymupdf.Page) -> Tuple[int, float]:
    """
    Get image count and coverage percentage for a page.

    Args:
        page: pymupdf.Page object

    Returns:
        Tuple of (image_count, coverage_percent)
    """
    try:
        images = page.get_images()
        image_count = len(images)

        if image_count == 0:
            return 0, 0.0

        # Calculate total image area
        page_area = page.rect.width * page.rect.height
        total_image_area = 0.0

        for img in images:
            # Get image bounding box
            try:
                img_rects = page.get_image_rects(img[0])
                for rect in img_rects:
                    total_image_area += rect.width * rect.height
            except Exception:
                # If we can't get rect, estimate from image dimensions
                # img[2] = width, img[3] = height in pixels
                # This is an approximation
                pass

        coverage_percent = (total_image_area / page_area * 100) if page_area > 0 else 0
        return image_count, min(coverage_percent, 100.0)

    except Exception as e:
        logger.warning(f"Error getting image info: {e}")
        return 0, 0.0


def determine_processing_path(
    drawing_count: int,
    image_count: int,
    image_coverage_percent: float
) -> str:
    """
    Determine the processing path for a page based on content analysis.

    Decision tree from spec section 1.3:
    - VECTOR: Many drawings, few/small images
    - HYBRID: Many drawings AND significant images
    - RASTER: Few drawings but has images
    - SKIP: No drawings and no images (text-only)

    Args:
        drawing_count: Number of drawing objects
        image_count: Number of images
        image_coverage_percent: Percentage of page covered by images

    Returns:
        ProcessingPath value (VECTOR, RASTER, HYBRID, or SKIP)
    """
    has_drawings = drawing_count >= MIN_DRAWINGS_FOR_VECTOR
    has_few_images = image_count <= MAX_IMAGES_FOR_VECTOR
    low_image_coverage = image_coverage_percent < MAX_IMAGE_COVERAGE_PERCENT

    if has_drawings:
        if has_few_images and low_image_coverage:
            return ProcessingPath.VECTOR
        else:
            return ProcessingPath.HYBRID
    else:
        if image_count > 0:
            return ProcessingPath.RASTER
        else:
            return ProcessingPath.SKIP


def get_title_block_text(page: pymupdf.Page) -> str:
    """
    Extract text from the title block area (bottom 15% or right 15% of page).

    Args:
        page: pymupdf.Page object

    Returns:
        Text from title block area
    """
    rect = page.rect
    width = rect.width
    height = rect.height

    # Bottom 15% of page
    bottom_rect = pymupdf.Rect(0, height * 0.85, width, height)

    # Right 15% of page
    right_rect = pymupdf.Rect(width * 0.85, 0, width, height)

    bottom_text = page.get_text(clip=bottom_rect)
    right_text = page.get_text(clip=right_rect)

    return f"{bottom_text} {right_text}"


def classify_page_type(page: pymupdf.Page) -> Tuple[str, List[str]]:
    """
    Classify the page type based on keyword detection.

    Searches title block first, then entire page.
    Priority order: FLOOR_PLAN > RCP > ELEVATION > SECTION > SCHEDULE > OTHER

    Args:
        page: pymupdf.Page object

    Returns:
        Tuple of (page_type, list of detected keywords)
    """
    # Get text from title block area first (higher confidence)
    title_block_text = get_title_block_text(page).lower()

    # Get all text from page
    all_text = page.get_text().lower()

    detected_keywords = []

    # Check keywords in priority order
    # Priority defined in spec: FLOOR_PLAN=1, RCP=2, ELEVATION=3, SECTION=4, SCHEDULE=5

    priority_order = [
        (PageType.FLOOR_PLAN, PAGE_TYPE_KEYWORDS["FLOOR_PLAN"]),
        (PageType.RCP, PAGE_TYPE_KEYWORDS["RCP"]),
        (PageType.ELEVATION, PAGE_TYPE_KEYWORDS["ELEVATION"]),
        (PageType.SECTION, PAGE_TYPE_KEYWORDS["SECTION"]),
        (PageType.SCHEDULE, PAGE_TYPE_KEYWORDS["SCHEDULE"]),
    ]

    # First check title block (higher priority)
    for page_type, keywords in priority_order:
        for keyword in keywords:
            if keyword.lower() in title_block_text:
                detected_keywords.append(keyword)
                logger.debug(f"Found '{keyword}' in title block -> {page_type}")
                return page_type, detected_keywords

    # Then check entire page
    for page_type, keywords in priority_order:
        for keyword in keywords:
            if keyword.lower() in all_text:
                detected_keywords.append(keyword)
                logger.debug(f"Found '{keyword}' on page -> {page_type}")
                return page_type, detected_keywords

    return PageType.OTHER, []


def classify_page(page: pymupdf.Page, page_number: int) -> PageClassification:
    """
    Fully classify a page: type, processing path, and metrics.

    Args:
        page: pymupdf.Page object
        page_number: 0-indexed page number

    Returns:
        PageClassification object with all classification results
    """
    # Get page dimensions
    rect = page.rect
    width = rect.width
    height = rect.height

    # Count drawings and images
    drawing_count = count_drawing_objects(page)
    image_count, image_coverage = get_image_info(page)

    # Determine processing path
    processing_path = determine_processing_path(
        drawing_count, image_count, image_coverage
    )

    # Classify page type
    page_type, detected_keywords = classify_page_type(page)

    classification = PageClassification(
        page_number=page_number,
        page_type=page_type,
        processing_path=processing_path,
        width_points=width,
        height_points=height,
        drawing_count=drawing_count,
        image_count=image_count,
        image_coverage_percent=image_coverage,
        detected_keywords=detected_keywords,
    )

    logger.info(
        f"Page {page_number}: {page_type} ({processing_path}) - "
        f"{drawing_count} drawings, {image_count} images ({image_coverage:.1f}% coverage)"
    )

    return classification


def classify_document(doc: pymupdf.Document) -> List[PageClassification]:
    """
    Classify all pages in a document.

    Args:
        doc: pymupdf.Document object

    Returns:
        List of PageClassification objects, one per page
    """
    classifications = []

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        classification = classify_page(page, page_num)
        classifications.append(classification)

    # Log summary
    floor_plans = sum(1 for c in classifications if c.page_type == PageType.FLOOR_PLAN)
    rcps = sum(1 for c in classifications if c.page_type == PageType.RCP)
    vector_pages = sum(1 for c in classifications if c.processing_path == ProcessingPath.VECTOR)
    raster_pages = sum(1 for c in classifications if c.processing_path == ProcessingPath.RASTER)

    logger.info(
        f"Document summary: {doc.page_count} pages, "
        f"{floor_plans} floor plans, {rcps} RCPs, "
        f"{vector_pages} vector, {raster_pages} raster"
    )

    return classifications
