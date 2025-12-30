"""
OCR Engine Module

Provides OCR functionality using PaddleOCR (primary) or Tesseract (fallback).
Handles coordinate transformation from image to PDF coordinates.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pymupdf

from ..constants import (
    OCR_CONFIDENCE_THRESHOLD,
    DEFAULT_RENDER_DPI,
)

logger = logging.getLogger(__name__)

# Try to import OCR engines
PADDLE_AVAILABLE = False
TESSERACT_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    logger.debug("PaddleOCR not available")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    logger.debug("Tesseract not available")


@dataclass
class TextBlock:
    """A text block extracted from a page."""
    text: str
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1) in PDF coordinates
    confidence: float
    font_size: Optional[float] = None
    font_name: Optional[str] = None
    source: str = "embedded"  # "embedded" or "ocr"


def get_available_engines() -> List[str]:
    """Return list of available OCR engines."""
    engines = []
    if PADDLE_AVAILABLE:
        engines.append("paddleocr")
    if TESSERACT_AVAILABLE:
        engines.append("tesseract")
    return engines


def extract_embedded_text(page: pymupdf.Page) -> List[TextBlock]:
    """
    Extract embedded text from a PDF page.

    Args:
        page: pymupdf.Page object

    Returns:
        List of TextBlock objects with embedded text
    """
    text_blocks = []

    try:
        # Get text as dictionary with block details
        text_dict = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)

        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:  # Type 0 is text
                continue

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue

                    bbox = span.get("bbox")
                    if not bbox:
                        continue

                    font_size = span.get("size", 0)
                    font_name = span.get("font", "")

                    text_blocks.append(TextBlock(
                        text=text,
                        bbox=tuple(bbox),
                        confidence=1.0,  # Embedded text has perfect confidence
                        font_size=font_size,
                        font_name=font_name,
                        source="embedded"
                    ))

    except Exception as e:
        logger.error(f"Error extracting embedded text: {e}")

    logger.debug(f"Extracted {len(text_blocks)} embedded text blocks")
    return text_blocks


def transform_ocr_coords_to_pdf(
    ocr_bbox: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    page_width: float,
    page_height: float
) -> Tuple[float, float, float, float]:
    """
    Transform OCR coordinates (image pixels) to PDF coordinates.

    OCR uses image coordinates (origin top-left).
    PDF uses points (origin bottom-left).

    Args:
        ocr_bbox: (x0, y0, x1, y1) in image pixels
        image_width: Width of rendered image
        image_height: Height of rendered image
        page_width: Width of page in PDF points
        page_height: Height of page in PDF points

    Returns:
        (x0, y0, x1, y1) in PDF coordinates
    """
    x0, y0, x1, y1 = ocr_bbox

    # Scale factors
    scale_x = page_width / image_width
    scale_y = page_height / image_height

    # Transform X coordinates (just scale)
    pdf_x0 = x0 * scale_x
    pdf_x1 = x1 * scale_x

    # Transform Y coordinates (scale and flip)
    # Image Y increases downward, PDF Y increases upward
    pdf_y0 = page_height - (y1 * scale_y)  # Note: y1 becomes pdf_y0 (flip)
    pdf_y1 = page_height - (y0 * scale_y)  # Note: y0 becomes pdf_y1 (flip)

    return (pdf_x0, pdf_y0, pdf_x1, pdf_y1)


def run_paddleocr(
    image: np.ndarray,
    page_width: float,
    page_height: float,
    use_gpu: bool = False,
    lang: str = "en"
) -> List[TextBlock]:
    """
    Run PaddleOCR on an image.

    Args:
        image: Numpy array (BGR format from OpenCV)
        page_width: Page width in PDF points
        page_height: Page height in PDF points
        use_gpu: Whether to use GPU acceleration
        lang: Language code

    Returns:
        List of TextBlock objects
    """
    if not PADDLE_AVAILABLE:
        logger.warning("PaddleOCR not available")
        return []

    text_blocks = []

    try:
        # Initialize PaddleOCR
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            use_gpu=use_gpu,
            show_log=False
        )

        # Run OCR
        result = ocr.ocr(image, cls=True)

        if not result or not result[0]:
            logger.debug("PaddleOCR returned no results")
            return []

        image_height, image_width = image.shape[:2]

        for line in result[0]:
            if len(line) < 2:
                continue

            # line[0] is the bounding box points (4 corners)
            # line[1] is (text, confidence)
            box_points = line[0]
            text_info = line[1]

            text = text_info[0]
            confidence = text_info[1]

            if confidence < OCR_CONFIDENCE_THRESHOLD:
                continue

            # Convert polygon to bounding box
            xs = [p[0] for p in box_points]
            ys = [p[1] for p in box_points]
            ocr_bbox = (min(xs), min(ys), max(xs), max(ys))

            # Transform to PDF coordinates
            pdf_bbox = transform_ocr_coords_to_pdf(
                ocr_bbox, image_width, image_height,
                page_width, page_height
            )

            text_blocks.append(TextBlock(
                text=text.strip(),
                bbox=pdf_bbox,
                confidence=confidence,
                source="ocr"
            ))

    except Exception as e:
        logger.error(f"PaddleOCR error: {e}")

    logger.debug(f"PaddleOCR extracted {len(text_blocks)} text blocks")
    return text_blocks


def run_tesseract(
    image: np.ndarray,
    page_width: float,
    page_height: float,
    lang: str = "eng"
) -> List[TextBlock]:
    """
    Run Tesseract OCR on an image.

    Args:
        image: Numpy array (BGR format from OpenCV)
        page_width: Page width in PDF points
        page_height: Page height in PDF points
        lang: Language code

    Returns:
        List of TextBlock objects
    """
    if not TESSERACT_AVAILABLE:
        logger.warning("Tesseract not available")
        return []

    text_blocks = []

    try:
        import cv2

        # Convert BGR to RGB for Tesseract
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        # Get detailed output
        data = pytesseract.image_to_data(
            image_rgb,
            lang=lang,
            output_type=pytesseract.Output.DICT
        )

        image_height, image_width = image.shape[:2]

        n_boxes = len(data['text'])
        for i in range(n_boxes):
            text = data['text'][i].strip()
            if not text:
                continue

            conf = data['conf'][i]
            if conf == -1:  # No confidence available
                conf = 0.5
            else:
                conf = conf / 100.0  # Convert to 0-1 range

            if conf < OCR_CONFIDENCE_THRESHOLD:
                continue

            x = data['left'][i]
            y = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]

            ocr_bbox = (x, y, x + w, y + h)

            # Transform to PDF coordinates
            pdf_bbox = transform_ocr_coords_to_pdf(
                ocr_bbox, image_width, image_height,
                page_width, page_height
            )

            text_blocks.append(TextBlock(
                text=text,
                bbox=pdf_bbox,
                confidence=conf,
                source="ocr"
            ))

    except Exception as e:
        logger.error(f"Tesseract error: {e}")

    logger.debug(f"Tesseract extracted {len(text_blocks)} text blocks")
    return text_blocks


def run_ocr(
    image: np.ndarray,
    page_width: float,
    page_height: float,
    engine: str = "paddleocr",
    use_gpu: bool = False,
    lang: str = "en"
) -> List[TextBlock]:
    """
    Run OCR on an image using specified engine.

    Args:
        image: Numpy array (BGR format)
        page_width: Page width in PDF points
        page_height: Page height in PDF points
        engine: "paddleocr" or "tesseract"
        use_gpu: Whether to use GPU (PaddleOCR only)
        lang: Language code

    Returns:
        List of TextBlock objects
    """
    if engine == "paddleocr" and PADDLE_AVAILABLE:
        return run_paddleocr(image, page_width, page_height, use_gpu, lang)
    elif engine == "tesseract" and TESSERACT_AVAILABLE:
        tess_lang = "eng" if lang == "en" else lang
        return run_tesseract(image, page_width, page_height, tess_lang)
    elif PADDLE_AVAILABLE:
        logger.info(f"Engine '{engine}' not available, using PaddleOCR")
        return run_paddleocr(image, page_width, page_height, use_gpu, lang)
    elif TESSERACT_AVAILABLE:
        logger.info(f"Engine '{engine}' not available, using Tesseract")
        return run_tesseract(image, page_width, page_height, "eng")
    else:
        logger.error("No OCR engine available")
        return []


def merge_text_blocks(
    embedded: List[TextBlock],
    ocr: List[TextBlock],
    overlap_threshold: float = 0.5
) -> List[TextBlock]:
    """
    Merge embedded text and OCR results.

    Prefers embedded text where both exist at the same location.

    Args:
        embedded: List of embedded text blocks
        ocr: List of OCR text blocks
        overlap_threshold: Minimum overlap ratio to consider same text

    Returns:
        Merged list of text blocks
    """
    merged = list(embedded)  # Start with all embedded text
    used_ocr = set()

    def bbox_overlap(bbox1, bbox2) -> float:
        """Calculate overlap ratio between two bboxes."""
        x0_1, y0_1, x1_1, y1_1 = bbox1
        x0_2, y0_2, x1_2, y1_2 = bbox2

        # Calculate intersection
        x0_i = max(x0_1, x0_2)
        y0_i = max(y0_1, y0_2)
        x1_i = min(x1_1, x1_2)
        y1_i = min(y1_1, y1_2)

        if x0_i >= x1_i or y0_i >= y1_i:
            return 0.0

        intersection = (x1_i - x0_i) * (y1_i - y0_i)
        area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
        area2 = (x1_2 - x0_2) * (y1_2 - y0_2)

        if min(area1, area2) == 0:
            return 0.0

        return intersection / min(area1, area2)

    # Mark OCR results that overlap with embedded text
    for i, ocr_block in enumerate(ocr):
        for emb_block in embedded:
            if bbox_overlap(ocr_block.bbox, emb_block.bbox) > overlap_threshold:
                used_ocr.add(i)
                break

    # Add non-overlapping OCR results
    for i, ocr_block in enumerate(ocr):
        if i not in used_ocr:
            merged.append(ocr_block)

    logger.debug(
        f"Merged text: {len(embedded)} embedded + "
        f"{len(ocr) - len(used_ocr)} new OCR = {len(merged)} total"
    )

    return merged


def extract_all_text(
    page: pymupdf.Page,
    page_image: Optional[np.ndarray] = None,
    force_ocr: bool = False,
    ocr_engine: str = "paddleocr",
    use_gpu: bool = False
) -> List[TextBlock]:
    """
    Extract all text from a page (embedded + OCR if needed).

    Args:
        page: pymupdf.Page object
        page_image: Optional rendered page image for OCR
        force_ocr: Force OCR even if embedded text exists
        ocr_engine: OCR engine to use
        use_gpu: Whether to use GPU for OCR

    Returns:
        List of all text blocks
    """
    # Always extract embedded text first
    embedded = extract_embedded_text(page)

    # Decide whether to run OCR
    run_ocr_flag = force_ocr or len(embedded) < 20

    if not run_ocr_flag:
        logger.debug("Sufficient embedded text, skipping OCR")
        return embedded

    if page_image is None:
        logger.debug("No page image provided for OCR")
        return embedded

    # Run OCR
    page_width = page.rect.width
    page_height = page.rect.height

    ocr_results = run_ocr(
        page_image,
        page_width,
        page_height,
        engine=ocr_engine,
        use_gpu=use_gpu
    )

    # Merge results
    return merge_text_blocks(embedded, ocr_results)
