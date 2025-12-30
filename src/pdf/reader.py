"""
PDF Reader Module

Functions for opening, reading, and extracting information from PDF files.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pymupdf

from ..constants import (
    DEFAULT_RENDER_DPI,
    LOW_MEMORY_DPI,
    LARGE_FILE_MB_THRESHOLD,
    MAX_PAGE_DIMENSION_POINTS,
)

logger = logging.getLogger(__name__)


class PDFReadError(Exception):
    """Raised when a PDF cannot be read."""
    pass


class PDFPasswordProtectedError(PDFReadError):
    """Raised when a PDF is password protected."""
    pass


class PDFCorruptedError(PDFReadError):
    """Raised when a PDF is corrupted."""
    pass


def open_pdf(filepath: str) -> pymupdf.Document:
    """
    Open a PDF file and return a document object.

    Args:
        filepath: Path to the PDF file

    Returns:
        pymupdf.Document object

    Raises:
        PDFReadError: If file not found
        PDFPasswordProtectedError: If PDF is password protected
        PDFCorruptedError: If PDF is corrupted
    """
    path = Path(filepath)

    if not path.exists():
        raise PDFReadError(f"File not found: {filepath}")

    if not path.is_file():
        raise PDFReadError(f"Path is not a file: {filepath}")

    try:
        doc = pymupdf.open(filepath)
    except Exception as e:
        error_msg = str(e).lower()
        if "password" in error_msg or "encrypted" in error_msg:
            raise PDFPasswordProtectedError(
                f"PDF is password protected: {filepath}. "
                "Please provide an unprotected version."
            )
        raise PDFCorruptedError(f"Cannot open PDF (may be corrupted): {filepath}. Error: {e}")

    # Check if document needs password
    if doc.needs_pass:
        doc.close()
        raise PDFPasswordProtectedError(
            f"PDF is password protected: {filepath}. "
            "Please provide an unprotected version."
        )

    # Check if document is valid
    if doc.page_count == 0:
        doc.close()
        raise PDFCorruptedError(f"PDF has no pages: {filepath}")

    logger.info(f"Opened PDF: {filepath} ({doc.page_count} pages)")
    return doc


def get_page_count(doc: pymupdf.Document) -> int:
    """
    Get the number of pages in a PDF document.

    Args:
        doc: pymupdf.Document object

    Returns:
        Integer count of pages
    """
    return doc.page_count


def get_page(doc: pymupdf.Document, page_number: int) -> pymupdf.Page:
    """
    Get a specific page from a PDF document.

    Args:
        doc: pymupdf.Document object
        page_number: 0-indexed page number

    Returns:
        pymupdf.Page object

    Raises:
        PDFReadError: If page number is invalid
    """
    if page_number < 0 or page_number >= doc.page_count:
        raise PDFReadError(
            f"Invalid page number: {page_number}. "
            f"Document has {doc.page_count} pages (0-{doc.page_count - 1})."
        )

    return doc.load_page(page_number)


def get_page_dimensions(page: pymupdf.Page) -> Tuple[float, float]:
    """
    Get the dimensions of a page in PDF points.

    Note: 72 points = 1 inch

    Args:
        page: pymupdf.Page object

    Returns:
        Tuple of (width, height) in PDF points
    """
    rect = page.rect
    return (rect.width, rect.height)


def get_file_size_mb(filepath: str) -> float:
    """
    Get the file size in megabytes.

    Args:
        filepath: Path to the file

    Returns:
        File size in MB
    """
    return Path(filepath).stat().st_size / (1024 * 1024)


def should_use_low_memory_mode(
    filepath: str,
    page_count: int,
    page_width: float,
    page_height: float
) -> bool:
    """
    Determine if low-memory mode should be used.

    Args:
        filepath: Path to the PDF file
        page_count: Number of pages
        page_width: Width in PDF points
        page_height: Height in PDF points

    Returns:
        True if low-memory mode should be used
    """
    # Check file size
    file_size_mb = get_file_size_mb(filepath)
    if file_size_mb > LARGE_FILE_MB_THRESHOLD:
        logger.info(f"Large file ({file_size_mb:.1f} MB) - using low-memory mode")
        return True

    # Check page count
    if page_count > 100:  # LARGE_PAGE_COUNT_THRESHOLD
        logger.info(f"Many pages ({page_count}) - using low-memory mode")
        return True

    # Check page dimensions
    if page_width > MAX_PAGE_DIMENSION_POINTS or page_height > MAX_PAGE_DIMENSION_POINTS:
        logger.info(f"Large page ({page_width:.0f}x{page_height:.0f} pts) - using low-memory mode")
        return True

    return False


def render_page_to_image(
    page: pymupdf.Page,
    dpi: int = DEFAULT_RENDER_DPI,
    low_memory: bool = False
) -> np.ndarray:
    """
    Render a PDF page to an image as a numpy array in BGR format.

    Args:
        page: pymupdf.Page object
        dpi: Resolution in dots per inch (default: 300)
        low_memory: If True, use LOW_MEMORY_DPI instead

    Returns:
        Image as numpy array in BGR format (OpenCV standard)
    """
    if low_memory:
        dpi = LOW_MEMORY_DPI
        logger.debug(f"Using low-memory DPI: {dpi}")

    # Calculate zoom factor (72 points per inch is PDF standard)
    zoom = dpi / 72.0
    mat = pymupdf.Matrix(zoom, zoom)

    # Render page to pixmap
    try:
        pix = page.get_pixmap(matrix=mat, alpha=False)
    except Exception as e:
        logger.warning(f"Render failed at DPI {dpi}, trying lower DPI: {e}")
        # Fall back to lower DPI
        zoom = LOW_MEMORY_DPI / 72.0
        mat = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

    # Convert to numpy array
    # pymupdf returns RGB format
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

    # Convert RGB to BGR (OpenCV standard)
    img_bgr = img[:, :, ::-1].copy()

    logger.debug(f"Rendered page to image: {img_bgr.shape} at {dpi} DPI")
    return img_bgr


def get_page_text(page: pymupdf.Page) -> str:
    """
    Extract all text from a page.

    Args:
        page: pymupdf.Page object

    Returns:
        All text on the page as a single string
    """
    return page.get_text()


def get_page_text_blocks(page: pymupdf.Page) -> list:
    """
    Extract text blocks with position information.

    Args:
        page: pymupdf.Page object

    Returns:
        List of text blocks, each with (x0, y0, x1, y1, text, block_no, block_type)
    """
    return page.get_text("blocks")
