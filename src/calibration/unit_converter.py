"""
Unit Converter Module

Functions for converting between PDF points and real-world units.
"""

import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# Constants
POINTS_PER_INCH = 72
INCHES_PER_FOOT = 12
FEET_PER_METER = 3.28084
MM_PER_INCH = 25.4


def pdf_points_to_real(
    length_in_points: float,
    scale_factor: float,
    output_unit: str = "feet"
) -> float:
    """
    Convert a length from PDF points to real-world units.

    Args:
        length_in_points: Length in PDF points
        scale_factor: Scale factor (PDF points per foot for imperial)
        output_unit: Target unit - "feet", "meters", "inches", "mm"

    Returns:
        Length in requested unit
    """
    if scale_factor == 0:
        logger.warning("Scale factor is 0, returning 0")
        return 0.0

    # Convert to feet first (assuming scale_factor is points per foot)
    length_feet = length_in_points / scale_factor

    # Convert to requested unit
    if output_unit == "feet":
        return length_feet
    elif output_unit == "inches":
        return length_feet * INCHES_PER_FOOT
    elif output_unit == "meters":
        return length_feet / FEET_PER_METER
    elif output_unit == "mm":
        return length_feet / FEET_PER_METER * 1000
    else:
        logger.warning(f"Unknown unit '{output_unit}', returning feet")
        return length_feet


def real_to_pdf_points(
    real_length: float,
    scale_factor: float,
    input_unit: str = "feet"
) -> float:
    """
    Convert a length from real-world units to PDF points.

    Args:
        real_length: Length in real-world units
        scale_factor: Scale factor (PDF points per foot for imperial)
        input_unit: Input unit - "feet", "meters", "inches", "mm"

    Returns:
        Length in PDF points
    """
    # Convert to feet first
    if input_unit == "feet":
        length_feet = real_length
    elif input_unit == "inches":
        length_feet = real_length / INCHES_PER_FOOT
    elif input_unit == "meters":
        length_feet = real_length * FEET_PER_METER
    elif input_unit == "mm":
        length_feet = real_length / 1000 * FEET_PER_METER
    else:
        logger.warning(f"Unknown unit '{input_unit}', assuming feet")
        length_feet = real_length

    # Convert to points
    return length_feet * scale_factor


def area_points_to_real(
    area_in_sq_points: float,
    scale_factor: float,
    output_unit: str = "sqft"
) -> float:
    """
    Convert an area from PDF points squared to real-world units.

    Note: Area conversion uses scale_factor squared.

    Args:
        area_in_sq_points: Area in PDF points squared
        scale_factor: Scale factor (PDF points per foot for imperial)
        output_unit: Target unit - "sqft", "sqm"

    Returns:
        Area in requested unit
    """
    if scale_factor == 0:
        logger.warning("Scale factor is 0, returning 0")
        return 0.0

    # Convert to square feet (scale_factor squared for area)
    area_sqft = area_in_sq_points / (scale_factor ** 2)

    # Convert to requested unit
    if output_unit == "sqft":
        return area_sqft
    elif output_unit == "sqm":
        return area_sqft / (FEET_PER_METER ** 2)
    else:
        logger.warning(f"Unknown unit '{output_unit}', returning sqft")
        return area_sqft


def real_area_to_points(
    real_area: float,
    scale_factor: float,
    input_unit: str = "sqft"
) -> float:
    """
    Convert an area from real-world units to PDF points squared.

    Args:
        real_area: Area in real-world units
        scale_factor: Scale factor (PDF points per foot for imperial)
        input_unit: Input unit - "sqft", "sqm"

    Returns:
        Area in PDF points squared
    """
    # Convert to square feet first
    if input_unit == "sqft":
        area_sqft = real_area
    elif input_unit == "sqm":
        area_sqft = real_area * (FEET_PER_METER ** 2)
    else:
        logger.warning(f"Unknown unit '{input_unit}', assuming sqft")
        area_sqft = real_area

    # Convert to points squared
    return area_sqft * (scale_factor ** 2)


def format_imperial_length(length_feet: float) -> str:
    """
    Format a length in feet as imperial string (e.g., 10'-6").

    Args:
        length_feet: Length in feet

    Returns:
        Formatted string like "10'-6""
    """
    feet = int(length_feet)
    remaining_inches = (length_feet - feet) * 12

    if remaining_inches < 0.1:
        return f"{feet}'-0\""
    elif abs(remaining_inches - round(remaining_inches)) < 0.1:
        return f"{feet}'-{int(round(remaining_inches))}\""
    else:
        return f"{feet}'-{remaining_inches:.1f}\""


def format_area(area: float, unit: str = "sqft") -> str:
    """
    Format an area with unit.

    Args:
        area: Area value
        unit: Unit string

    Returns:
        Formatted string like "150.5 SF" or "14.0 m²"
    """
    if unit == "sqft":
        return f"{area:.1f} SF"
    elif unit == "sqm":
        return f"{area:.1f} m²"
    else:
        return f"{area:.1f} {unit}"


def calculate_scale_factor_from_calibration(
    point1: Tuple[float, float],
    point2: Tuple[float, float],
    real_length: float,
    length_unit: str = "feet"
) -> float:
    """
    Calculate scale factor from two-point calibration.

    Args:
        point1: First point (x, y) in PDF coordinates
        point2: Second point (x, y) in PDF coordinates
        real_length: Real-world length between points
        length_unit: Unit of real_length

    Returns:
        Scale factor (PDF points per foot)
    """
    import math

    # Calculate PDF distance
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    pdf_length = math.sqrt(dx * dx + dy * dy)

    # Convert real length to feet
    if length_unit == "feet":
        real_feet = real_length
    elif length_unit == "inches":
        real_feet = real_length / 12
    elif length_unit == "meters":
        real_feet = real_length * FEET_PER_METER
    elif length_unit == "mm":
        real_feet = real_length / 1000 * FEET_PER_METER
    else:
        real_feet = real_length

    if real_feet == 0:
        logger.warning("Real length is 0, cannot calculate scale")
        return 0.0

    scale_factor = pdf_length / real_feet
    logger.info(f"Calibration: {pdf_length:.1f} pts / {real_feet:.2f} ft = {scale_factor:.2f} pts/ft")

    return scale_factor


def parse_calibration_string(calib_string: str) -> Tuple[Tuple[float, float], Tuple[float, float], float, str]:
    """
    Parse a calibration string in format "x1,y1:x2,y2=LENGTH UNIT".

    Args:
        calib_string: Calibration string like "100,200:300,200=10ft"

    Returns:
        Tuple of (point1, point2, length, unit)

    Raises:
        ValueError: If string format is invalid
    """
    import re

    # Pattern: x1,y1:x2,y2=LENGTHunit
    pattern = r"(\d+(?:\.\d+)?),(\d+(?:\.\d+)?):(\d+(?:\.\d+)?),(\d+(?:\.\d+)?)=(\d+(?:\.\d+)?)\s*(ft|feet|m|meters?|in|inches?)?"

    match = re.match(pattern, calib_string, re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid calibration format: {calib_string}")

    x1, y1, x2, y2, length, unit = match.groups()

    point1 = (float(x1), float(y1))
    point2 = (float(x2), float(y2))
    real_length = float(length)
    length_unit = (unit or "feet").lower()

    # Normalize unit names
    if length_unit in ("ft", "feet"):
        length_unit = "feet"
    elif length_unit in ("m", "meter", "meters"):
        length_unit = "meters"
    elif length_unit in ("in", "inch", "inches"):
        length_unit = "inches"

    return point1, point2, real_length, length_unit
