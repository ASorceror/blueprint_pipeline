# Scale detection and calibration module

from .scale_detector import (
    ScaleResult,
    DimensionMatch,
    find_dimension_matches,
    find_scale_from_text,
    extract_scale_from_page,
    parse_manual_scale,
    check_scale_conflict,
)

from .unit_converter import (
    pdf_points_to_real,
    real_to_pdf_points,
    area_points_to_real,
    real_area_to_points,
    format_imperial_length,
    format_area,
    calculate_scale_factor_from_calibration,
    parse_calibration_string,
)

from .dimension_associator import (
    DimensionAssociation,
    associate_dimensions_to_segments,
    find_associated_segment,
)

__all__ = [
    # Scale Detector
    "ScaleResult",
    "DimensionMatch",
    "find_dimension_matches",
    "find_scale_from_text",
    "extract_scale_from_page",
    "parse_manual_scale",
    "check_scale_conflict",
    # Unit Converter
    "pdf_points_to_real",
    "real_to_pdf_points",
    "area_points_to_real",
    "real_area_to_points",
    "format_imperial_length",
    "format_area",
    "calculate_scale_factor_from_calibration",
    "parse_calibration_string",
    # Dimension Associator
    "DimensionAssociation",
    "associate_dimensions_to_segments",
    "find_associated_segment",
]
