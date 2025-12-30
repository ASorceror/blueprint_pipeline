# Raster/scanned PDF processing module

from .preprocessor import (
    PreprocessingResult,
    to_grayscale,
    enhance_contrast,
    remove_noise,
    binarize,
    morphological_cleanup,
    detect_skew_angle,
    correct_skew,
    preprocess_image,
)

from .room_detector import (
    RasterDetectionResult,
    transform_image_to_pdf_coords,
    simplify_contour,
    contour_to_polygon,
    detect_rooms_flood_fill,
    detect_rooms_connected_components,
    detect_rooms_contours,
    assess_quality,
    detect_rooms_from_raster,
)

__all__ = [
    # Preprocessor
    "PreprocessingResult",
    "to_grayscale",
    "enhance_contrast",
    "remove_noise",
    "binarize",
    "morphological_cleanup",
    "detect_skew_angle",
    "correct_skew",
    "preprocess_image",
    # Room Detector
    "RasterDetectionResult",
    "transform_image_to_pdf_coords",
    "simplify_contour",
    "contour_to_polygon",
    "detect_rooms_flood_fill",
    "detect_rooms_connected_components",
    "detect_rooms_contours",
    "assess_quality",
    "detect_rooms_from_raster",
]
