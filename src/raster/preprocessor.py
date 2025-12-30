"""
Image Preprocessor Module

Preprocessing pipeline for scanned PDF pages.
"""

import logging
from dataclasses import dataclass
from typing import Tuple, List, Optional

import cv2
import numpy as np

from ..constants import (
    DEFAULT_RENDER_DPI,
    LOW_MEMORY_DPI,
    KERNEL_SIZE_DIVISOR,
    MAX_SKEW_CORRECTION_DEG,
)

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    """Result of image preprocessing."""
    binary_image: np.ndarray
    original_image: np.ndarray
    steps_applied: List[str]
    skew_angle: float
    dpi: int


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale.

    Args:
        image: BGR or grayscale image

    Returns:
        Grayscale image
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """
    Enhance contrast using CLAHE.

    Args:
        gray: Grayscale image

    Returns:
        Contrast-enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def remove_noise(image: np.ndarray, method: str = "gaussian") -> np.ndarray:
    """
    Remove noise from image.

    Args:
        image: Input image
        method: "gaussian" or "bilateral"

    Returns:
        Denoised image
    """
    if method == "bilateral":
        return cv2.bilateralFilter(image, 9, 75, 75)
    else:
        return cv2.GaussianBlur(image, (3, 3), 0)


def binarize(gray: np.ndarray, block_size: int = 11, c: int = 2) -> np.ndarray:
    """
    Binarize image using adaptive thresholding.

    Args:
        gray: Grayscale image
        block_size: Block size for adaptive threshold
        c: Constant subtracted from mean

    Returns:
        Binary image (walls = black/0, rooms = white/255)
    """
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c
    )
    return binary


def morphological_cleanup(
    binary: np.ndarray,
    kernel_size: int = 3
) -> np.ndarray:
    """
    Apply morphological operations to clean up binary image.

    Args:
        binary: Binary image
        kernel_size: Size of morphological kernel

    Returns:
        Cleaned binary image
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_size, kernel_size)
    )

    # Close: fills small gaps in walls
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Open: removes small noise
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    return opened


def detect_skew_angle(binary: np.ndarray) -> float:
    """
    Detect skew angle using Hough lines.

    Args:
        binary: Binary image

    Returns:
        Detected skew angle in degrees
    """
    # Detect edges
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)

    # Detect lines using Hough transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    if lines is None or len(lines) == 0:
        return 0.0

    # Collect angles
    angles = []
    for line in lines:
        rho, theta = line[0]
        # Convert to degrees and normalize to [-45, 45]
        angle = np.degrees(theta) - 90
        if -45 < angle < 45:
            angles.append(angle)

    if not angles:
        return 0.0

    # Return median angle
    return float(np.median(angles))


def correct_skew(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate image to correct skew.

    Args:
        image: Input image
        angle: Skew angle in degrees

    Returns:
        Rotated image
    """
    if abs(angle) < 0.1:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new image size to avoid cropping
    cos = abs(rotation_matrix[0, 0])
    sin = abs(rotation_matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # Adjust rotation matrix for new size
    rotation_matrix[0, 2] += (new_w - w) / 2
    rotation_matrix[1, 2] += (new_h - h) / 2

    # Rotate image
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


def preprocess_image(
    image: np.ndarray,
    dpi: int = DEFAULT_RENDER_DPI,
    correct_skew_flag: bool = True,
    denoise_method: str = "gaussian"
) -> PreprocessingResult:
    """
    Full preprocessing pipeline for scanned blueprint image.

    Steps:
    1. Convert to grayscale
    2. Enhance contrast (CLAHE)
    3. Remove noise
    4. Binarize (adaptive threshold)
    5. Morphological cleanup
    6. Skew correction (optional)

    Args:
        image: BGR numpy array
        dpi: Rendering DPI (affects kernel sizes)
        correct_skew_flag: Whether to attempt skew correction
        denoise_method: "gaussian" or "bilateral"

    Returns:
        PreprocessingResult with binary image and metadata
    """
    steps = []
    original = image.copy()

    # Step 1: Grayscale
    gray = to_grayscale(image)
    steps.append("grayscale")

    # Step 2: Contrast enhancement
    enhanced = enhance_contrast(gray)
    steps.append("clahe_contrast")

    # Step 3: Noise removal
    denoised = remove_noise(enhanced, method=denoise_method)
    steps.append(f"denoise_{denoise_method}")

    # Step 4: Binarization
    binary = binarize(denoised)
    steps.append("adaptive_threshold")

    # Step 5: Morphological cleanup
    kernel_size = max(3, dpi // KERNEL_SIZE_DIVISOR)
    cleaned = morphological_cleanup(binary, kernel_size)
    steps.append(f"morphology_k{kernel_size}")

    # Step 6: Skew correction
    skew_angle = 0.0
    if correct_skew_flag:
        skew_angle = detect_skew_angle(cleaned)
        if abs(skew_angle) <= MAX_SKEW_CORRECTION_DEG:
            cleaned = correct_skew(cleaned, skew_angle)
            steps.append(f"skew_correction_{skew_angle:.2f}deg")
        else:
            logger.debug(f"Skew angle {skew_angle:.2f} too large, not correcting")

    logger.info(f"Preprocessing complete: {', '.join(steps)}")

    return PreprocessingResult(
        binary_image=cleaned,
        original_image=original,
        steps_applied=steps,
        skew_angle=skew_angle,
        dpi=dpi
    )
