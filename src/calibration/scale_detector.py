"""
Scale Detector Module

Detects scale from dimension strings and scale text notations.
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional

import pymupdf

from ..constants import (
    SCALE_FACTORS,
    METRIC_SCALE_FACTORS,
    DEFAULT_COMMERCIAL_SCALE,
    DEFAULT_COMMERCIAL_SCALE_FACTOR,
    SCALE_CONFLICT_THRESHOLD_PERCENT,
    Confidence,
)

logger = logging.getLogger(__name__)


@dataclass
class DimensionMatch:
    """A matched dimension string with parsed value."""
    text: str
    value_inches: float  # Normalized to inches for comparison
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    confidence: str
    is_metric: bool = False


@dataclass
class ScaleResult:
    """Result of scale detection."""
    scale_factor: float  # PDF points per foot
    scale_source: str  # "manual", "dimension", "scale_text", "default"
    scale_confidence: str  # HIGH, MEDIUM, LOW
    detected_notation: str  # Human-readable, e.g., "1/8 inch = 1 foot"
    units: str  # "imperial" or "metric"


# =============================================================================
# DIMENSION PATTERN MATCHING (from Appendix A)
# =============================================================================

# Imperial - Feet and Inches: 10'-6", 10' 6", 10'6", 10'-6 1/2"
IMPERIAL_FEET_INCHES_PATTERN = re.compile(
    r"(\d+)['\'][-\s]?(\d+)?(?:[-\s]?(\d+/\d+))?[\"″]?",
    re.IGNORECASE
)

# Imperial - Feet Only: 10', 10 FT, 10.5'
IMPERIAL_FEET_ONLY_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:ft|feet|FT|FEET|['\'])",
    re.IGNORECASE
)

# Imperial - Inches Only: 126", 126 IN
IMPERIAL_INCHES_ONLY_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:in|inch|inches|IN|[\"″])",
    re.IGNORECASE
)

# Metric - Millimeters: 3000mm, 3000 mm
METRIC_MM_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:mm|MM)",
    re.IGNORECASE
)

# Metric - Centimeters: 300cm, 300 cm
METRIC_CM_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:cm|CM)",
    re.IGNORECASE
)

# Metric - Meters: 3.0m, 3 M (but not mm)
METRIC_M_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:m|M)(?!m|M)",
    re.IGNORECASE
)

# Scale notation - Fractional inch: 1/8" = 1'-0"
SCALE_FRACTIONAL_PATTERN = re.compile(
    r"(\d+)/(\d+)[\"″]?\s*=\s*1['\'][-\s]?0?[\"″]?",
    re.IGNORECASE
)

# Scale notation - Metric ratio: 1:100, 1 : 50
# Requires at least 2 digits AND not followed by AM/PM to avoid matching timestamps
SCALE_RATIO_PATTERN = re.compile(
    r"1\s*:\s*(\d{2,})(?!\s*[AP]M)",
    re.IGNORECASE
)

# Scale notation - Text description
SCALE_TEXT_PATTERNS = [
    (re.compile(r"QUARTER\s*INCH\s*SCALE", re.IGNORECASE), 18),  # 1/4" = 1'
    (re.compile(r"EIGHTH\s*INCH\s*SCALE", re.IGNORECASE), 9),   # 1/8" = 1'
    (re.compile(r"HALF\s*INCH\s*SCALE", re.IGNORECASE), 36),    # 1/2" = 1'
    (re.compile(r"FULL\s*SCALE", re.IGNORECASE), 72),           # 1" = 1'
]


def parse_fraction(frac_str: str) -> float:
    """Parse a fraction string like '1/2' to a float."""
    if '/' in frac_str:
        num, denom = frac_str.split('/')
        return float(num) / float(denom)
    return 0.0


def parse_feet_inches(match: re.Match) -> float:
    """Parse a feet-inches match to total inches."""
    feet = float(match.group(1)) if match.group(1) else 0
    inches = float(match.group(2)) if match.group(2) else 0
    fraction = parse_fraction(match.group(3)) if match.group(3) else 0

    return feet * 12 + inches + fraction


def find_dimension_matches(text: str) -> List[Tuple[str, float, bool]]:
    """
    Find all dimension matches in text.

    Returns list of (matched_text, value_in_inches, is_metric)
    """
    matches = []

    # Try imperial feet-inches first
    for match in IMPERIAL_FEET_INCHES_PATTERN.finditer(text):
        value = parse_feet_inches(match)
        if value > 0:
            matches.append((match.group(), value, False))

    # Imperial feet only
    for match in IMPERIAL_FEET_ONLY_PATTERN.finditer(text):
        feet = float(match.group(1))
        matches.append((match.group(), feet * 12, False))

    # Imperial inches only
    for match in IMPERIAL_INCHES_ONLY_PATTERN.finditer(text):
        inches = float(match.group(1))
        matches.append((match.group(), inches, False))

    # Metric millimeters
    for match in METRIC_MM_PATTERN.finditer(text):
        mm = float(match.group(1))
        inches = mm / 25.4
        matches.append((match.group(), inches, True))

    # Metric centimeters
    for match in METRIC_CM_PATTERN.finditer(text):
        cm = float(match.group(1))
        inches = cm / 2.54
        matches.append((match.group(), inches, True))

    # Metric meters
    for match in METRIC_M_PATTERN.finditer(text):
        m = float(match.group(1))
        inches = m * 39.37
        matches.append((match.group(), inches, True))

    return matches


def find_scale_from_text(text: str) -> Optional[Tuple[float, str]]:
    """
    Find scale notation in text.

    Returns (scale_factor, notation_string) or None
    """
    # Try fractional inch pattern: 1/8" = 1'-0"
    match = SCALE_FRACTIONAL_PATTERN.search(text)
    if match:
        numerator = int(match.group(1))
        denominator = int(match.group(2))
        # Scale factor = (numerator/denominator) * 72 points per inch / 12 inches per foot
        scale_factor = (numerator / denominator) * 72 / 12
        notation = f"{numerator}/{denominator} inch = 1 foot"
        return (scale_factor, notation)

    # Try metric ratio pattern: 1:100
    match = SCALE_RATIO_PATTERN.search(text)
    if match:
        ratio = int(match.group(1))
        # For 1:100, 1 unit on paper = 100 units real
        # In points: 72 points/inch * 39.37 inches/meter = 2835 points/meter
        # At 1:100, 1 real meter = 2835/100 = 28.35 points
        scale_factor = 2835 / ratio  # points per meter
        notation = f"1:{ratio}"
        return (scale_factor, notation)

    # Try text patterns
    for pattern, factor in SCALE_TEXT_PATTERNS:
        if pattern.search(text):
            notation = pattern.pattern.replace(r'\s*', ' ').replace('\\', '')
            return (factor, notation)

    return None


def extract_scale_from_page(page: pymupdf.Page) -> ScaleResult:
    """
    Extract scale from a PDF page.

    Tries multiple methods in priority order:
    1. Scale text detection
    2. Default scale

    Args:
        page: pymupdf.Page object

    Returns:
        ScaleResult with detected scale
    """
    # Get page text
    all_text = page.get_text()

    # Get title block text (bottom 15% or right 15%)
    rect = page.rect
    width, height = rect.width, rect.height

    bottom_rect = pymupdf.Rect(0, height * 0.85, width, height)
    right_rect = pymupdf.Rect(width * 0.85, 0, width, height)

    title_block_text = page.get_text(clip=bottom_rect) + " " + page.get_text(clip=right_rect)

    # Try to find scale in title block first (higher confidence)
    scale_result = find_scale_from_text(title_block_text)
    if scale_result:
        scale_factor, notation = scale_result
        logger.info(f"Scale found in title block: {notation} (factor: {scale_factor})")
        return ScaleResult(
            scale_factor=scale_factor,
            scale_source="scale_text",
            scale_confidence=Confidence.HIGH,
            detected_notation=notation,
            units="metric" if ":" in notation else "imperial"
        )

    # Try anywhere on page
    scale_result = find_scale_from_text(all_text)
    if scale_result:
        scale_factor, notation = scale_result
        logger.info(f"Scale found on page: {notation} (factor: {scale_factor})")
        return ScaleResult(
            scale_factor=scale_factor,
            scale_source="scale_text",
            scale_confidence=Confidence.MEDIUM,
            detected_notation=notation,
            units="metric" if ":" in notation else "imperial"
        )

    # Default to commercial scale
    logger.warning(f"No scale detected, using default: {DEFAULT_COMMERCIAL_SCALE}")
    return ScaleResult(
        scale_factor=DEFAULT_COMMERCIAL_SCALE_FACTOR,
        scale_source="default",
        scale_confidence=Confidence.LOW,
        detected_notation=DEFAULT_COMMERCIAL_SCALE,
        units="imperial"
    )


def parse_manual_scale(scale_string: str) -> Optional[ScaleResult]:
    """
    Parse a manually provided scale string.

    Accepts formats like:
    - "1/8 inch = 1 foot"
    - "1:100"
    - Direct factor like "9" (points per foot)

    Args:
        scale_string: User-provided scale string

    Returns:
        ScaleResult or None if parsing fails
    """
    scale_string = scale_string.strip()

    # Check for known scales
    for known_scale, factor in SCALE_FACTORS.items():
        if known_scale.lower() in scale_string.lower():
            return ScaleResult(
                scale_factor=factor,
                scale_source="manual",
                scale_confidence=Confidence.HIGH,
                detected_notation=known_scale,
                units="imperial"
            )

    # Check for metric scales
    for known_scale, factor in METRIC_SCALE_FACTORS.items():
        if known_scale.lower() in scale_string.lower():
            return ScaleResult(
                scale_factor=factor,
                scale_source="manual",
                scale_confidence=Confidence.HIGH,
                detected_notation=known_scale,
                units="metric"
            )

    # Try parsing as scale notation
    result = find_scale_from_text(scale_string)
    if result:
        scale_factor, notation = result
        return ScaleResult(
            scale_factor=scale_factor,
            scale_source="manual",
            scale_confidence=Confidence.HIGH,
            detected_notation=notation,
            units="metric" if ":" in notation else "imperial"
        )

    # Try parsing as direct factor
    try:
        factor = float(scale_string)
        return ScaleResult(
            scale_factor=factor,
            scale_source="manual",
            scale_confidence=Confidence.HIGH,
            detected_notation=f"Factor: {factor}",
            units="imperial"
        )
    except ValueError:
        pass

    return None


def check_scale_conflict(scale1: float, scale2: float) -> bool:
    """
    Check if two scale factors conflict (differ by more than threshold).

    Args:
        scale1: First scale factor
        scale2: Second scale factor

    Returns:
        True if scales conflict
    """
    if scale1 == 0 or scale2 == 0:
        return False

    diff_percent = abs(scale1 - scale2) / max(scale1, scale2) * 100
    return diff_percent > SCALE_CONFLICT_THRESHOLD_PERCENT
