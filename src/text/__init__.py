# Text extraction and OCR module

from .ocr_engine import (
    TextBlock,
    get_available_engines,
    extract_embedded_text,
    run_ocr,
    run_paddleocr,
    run_tesseract,
    merge_text_blocks,
    extract_all_text,
    transform_ocr_coords_to_pdf,
)

from .label_matcher import (
    RoomLabel,
    LabelMatch,
    filter_room_labels,
    match_labels_to_polygons,
    apply_labels_to_polygons,
    is_room_label_pattern,
    is_excluded_pattern,
)

from .height_parser import (
    HeightAnnotation,
    extract_height_from_text,
    extract_heights_from_page,
    match_heights_to_rooms,
    normalize_room_name,
    extract_room_number_from_name,
)

__all__ = [
    # OCR Engine
    "TextBlock",
    "get_available_engines",
    "extract_embedded_text",
    "run_ocr",
    "run_paddleocr",
    "run_tesseract",
    "merge_text_blocks",
    "extract_all_text",
    "transform_ocr_coords_to_pdf",
    # Label Matcher
    "RoomLabel",
    "LabelMatch",
    "filter_room_labels",
    "match_labels_to_polygons",
    "apply_labels_to_polygons",
    "is_room_label_pattern",
    "is_excluded_pattern",
    # Height Parser
    "HeightAnnotation",
    "extract_height_from_text",
    "extract_heights_from_page",
    "match_heights_to_rooms",
    "normalize_room_name",
    "extract_room_number_from_name",
]
