# PDF reading and classification module

from .reader import (
    open_pdf,
    get_page_count,
    get_page,
    get_page_dimensions,
    render_page_to_image,
    get_page_text,
    get_page_text_blocks,
    should_use_low_memory_mode,
    PDFReadError,
    PDFPasswordProtectedError,
    PDFCorruptedError,
)

from .classifier import (
    classify_page,
    classify_document,
    classify_page_type,
    determine_processing_path,
    count_drawing_objects,
    get_image_info,
    PageClassification,
)

__all__ = [
    # Reader functions
    "open_pdf",
    "get_page_count",
    "get_page",
    "get_page_dimensions",
    "render_page_to_image",
    "get_page_text",
    "get_page_text_blocks",
    "should_use_low_memory_mode",
    # Reader exceptions
    "PDFReadError",
    "PDFPasswordProtectedError",
    "PDFCorruptedError",
    # Classifier functions
    "classify_page",
    "classify_document",
    "classify_page_type",
    "determine_processing_path",
    "count_drawing_objects",
    "get_image_info",
    "PageClassification",
]
