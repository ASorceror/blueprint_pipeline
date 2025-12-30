"""
Geometry Calculator Module

Functions for calculating room measurements from polygons.
"""

import logging
import math
from typing import List, Tuple

from shapely.geometry import Polygon

from ..constants import (
    MIN_AREA_REAL_SQFT,
    MAX_AREA_REAL_SQFT,
    MIN_CEILING_HEIGHT_FT,
    MAX_CEILING_HEIGHT_FT,
)
from .room import Room

logger = logging.getLogger(__name__)


def calculate_floor_area(
    polygon: Polygon,
    scale_factor: float
) -> float:
    """
    Calculate floor area in real-world units.

    Args:
        polygon: Shapely Polygon in PDF coordinates
        scale_factor: PDF points per foot

    Returns:
        Floor area in square feet
    """
    if scale_factor == 0:
        logger.warning("Scale factor is 0, cannot calculate area")
        return 0.0

    # Polygon area is in PDF points squared
    area_sq_points = polygon.area

    # Convert to square feet: divide by scale_factor squared
    floor_area_sqft = area_sq_points / (scale_factor ** 2)

    return floor_area_sqft


def calculate_perimeter(
    polygon: Polygon,
    scale_factor: float
) -> float:
    """
    Calculate perimeter in real-world units.

    Args:
        polygon: Shapely Polygon in PDF coordinates
        scale_factor: PDF points per foot

    Returns:
        Perimeter in feet
    """
    if scale_factor == 0:
        logger.warning("Scale factor is 0, cannot calculate perimeter")
        return 0.0

    # Polygon length is perimeter in PDF points
    perimeter_points = polygon.length

    # Convert to feet
    perimeter_ft = perimeter_points / scale_factor

    return perimeter_ft


def calculate_wall_area(
    perimeter_ft: float,
    ceiling_height_ft: float
) -> float:
    """
    Calculate gross wall area.

    Note: This is gross wall area without subtracting doors/windows.

    Args:
        perimeter_ft: Room perimeter in feet
        ceiling_height_ft: Ceiling height in feet

    Returns:
        Wall area in square feet
    """
    return perimeter_ft * ceiling_height_ft


def calculate_ceiling_area(floor_area_sqft: float) -> float:
    """
    Calculate ceiling area.

    For flat ceilings, ceiling area equals floor area.
    Complex ceilings not supported in v1.

    Args:
        floor_area_sqft: Floor area in square feet

    Returns:
        Ceiling area in square feet
    """
    return floor_area_sqft


def convert_polygon_to_real_units(
    vertices: List[Tuple[float, float]],
    scale_factor: float
) -> List[Tuple[float, float]]:
    """
    Convert polygon vertices from PDF points to real-world units.

    Args:
        vertices: List of (x, y) tuples in PDF points
        scale_factor: PDF points per foot

    Returns:
        List of (x, y) tuples in feet
    """
    if scale_factor == 0:
        return vertices

    return [(x / scale_factor, y / scale_factor) for x, y in vertices]


def validate_room_measurements(room: Room) -> List[str]:
    """
    Validate room measurements and return warnings.

    Args:
        room: Room object with measurements

    Returns:
        List of warning messages
    """
    warnings = []

    # Check floor area
    if room.floor_area_sqft < MIN_AREA_REAL_SQFT:
        warnings.append(
            f"Floor area {room.floor_area_sqft:.1f} SF is below minimum "
            f"({MIN_AREA_REAL_SQFT} SF)"
        )

    if room.floor_area_sqft > MAX_AREA_REAL_SQFT:
        warnings.append(
            f"Floor area {room.floor_area_sqft:.1f} SF exceeds maximum "
            f"({MAX_AREA_REAL_SQFT} SF)"
        )

    # Check ceiling height
    if room.ceiling_height_ft < MIN_CEILING_HEIGHT_FT:
        warnings.append(
            f"Ceiling height {room.ceiling_height_ft:.1f}' is below minimum "
            f"({MIN_CEILING_HEIGHT_FT}')"
        )

    if room.ceiling_height_ft > MAX_CEILING_HEIGHT_FT:
        warnings.append(
            f"Ceiling height {room.ceiling_height_ft:.1f}' exceeds maximum "
            f"({MAX_CEILING_HEIGHT_FT}')"
        )

    # Check perimeter vs area ratio
    if room.floor_area_sqft > 0:
        expected_perimeter = 4 * math.sqrt(room.floor_area_sqft)
        if room.perimeter_ft > expected_perimeter * 4:
            warnings.append(
                f"Perimeter {room.perimeter_ft:.1f}' is unusually high "
                f"for floor area {room.floor_area_sqft:.1f} SF (irregular shape)"
            )

    # Check wall area reasonableness
    if room.floor_area_sqft > 0 and room.wall_area_sqft > room.floor_area_sqft * 10:
        warnings.append(
            f"Wall area {room.wall_area_sqft:.1f} SF is unusually high "
            f"relative to floor area"
        )

    return warnings


def calculate_all_room_measurements(
    room: Room,
    polygon: Polygon,
    scale_factor: float,
    ceiling_height_ft: float
) -> Room:
    """
    Calculate all measurements for a room.

    Args:
        room: Room object to populate
        polygon: Shapely Polygon in PDF coordinates
        scale_factor: PDF points per foot
        ceiling_height_ft: Ceiling height in feet

    Returns:
        Room with all measurements populated
    """
    # Store polygon
    room.shapely_polygon = polygon
    room.polygon_pdf_points = list(polygon.exterior.coords)[:-1]

    # Store scale info
    room.scale_factor = scale_factor

    # Calculate measurements
    room.floor_area_sqft = calculate_floor_area(polygon, scale_factor)
    room.perimeter_ft = calculate_perimeter(polygon, scale_factor)
    room.ceiling_height_ft = ceiling_height_ft
    room.wall_area_sqft = calculate_wall_area(room.perimeter_ft, ceiling_height_ft)
    room.ceiling_area_sqft = calculate_ceiling_area(room.floor_area_sqft)

    # Convert vertices to real units
    room.polygon_real_units = convert_polygon_to_real_units(
        room.polygon_pdf_points, scale_factor
    )

    # Validate and add warnings
    warnings = validate_room_measurements(room)
    room.warnings = warnings

    if warnings:
        for warning in warnings:
            logger.warning(f"Room {room.room_id}: {warning}")

    logger.debug(
        f"Room {room.room_id}: {room.floor_area_sqft:.1f} SF, "
        f"{room.perimeter_ft:.1f}' perimeter, "
        f"{room.ceiling_height_ft:.1f}' height"
    )

    return room


def create_room_from_polygon(
    polygon_id: str,
    polygon: Polygon,
    vertices: List[Tuple[float, float]],
    room_name: str,
    sheet_number: int,
    scale_factor: float,
    ceiling_height_ft: float,
    source: str = "vector",
    confidence: str = "MEDIUM"
) -> Room:
    """
    Create a complete Room object from a polygon.

    Args:
        polygon_id: Unique room identifier
        polygon: Shapely Polygon
        vertices: Polygon vertices in PDF points
        room_name: Room name/label
        sheet_number: Page number (0-indexed)
        scale_factor: PDF points per foot
        ceiling_height_ft: Ceiling height in feet
        source: "vector" or "raster"
        confidence: Confidence level

    Returns:
        Complete Room object
    """
    room = Room(
        room_id=polygon_id,
        room_name=room_name,
        sheet_number=sheet_number,
        source=source,
        confidence=confidence,
    )

    return calculate_all_room_measurements(
        room, polygon, scale_factor, ceiling_height_ft
    )
