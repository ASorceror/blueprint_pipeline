"""
Room Detector Module

Detects rooms from preprocessed raster images.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid


def get_largest_polygon(geom) -> Optional[Polygon]:
    """
    Extract the largest polygon from a geometry.
    Handles both Polygon and MultiPolygon cases.

    Args:
        geom: Shapely geometry (Polygon or MultiPolygon)

    Returns:
        Largest Polygon or None if invalid
    """
    if geom is None:
        return None
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        if len(geom.geoms) == 0:
            return None
        return max(geom.geoms, key=lambda p: p.area)
    return None

from ..constants import (
    MIN_CONTOUR_AREA_RATIO,
    DP_EPSILON_BASE,
    DEFAULT_RENDER_DPI,
    FLOOD_FILL_TOLERANCE,
    MIN_VERTICES,
    Confidence,
)
from ..vector.polygonizer import RoomPolygon

logger = logging.getLogger(__name__)


@dataclass
class RasterDetectionResult:
    """Result of raster room detection."""
    rooms: List[RoomPolygon]
    method: str
    quality: str  # "GOOD", "SUSPECT", "POOR"
    details: dict


def transform_image_to_pdf_coords(
    image_points: List[Tuple[float, float]],
    image_width: int,
    image_height: int,
    page_width: float,
    page_height: float
) -> List[Tuple[float, float]]:
    """
    Transform image coordinates to PDF coordinates.

    Image: origin top-left, Y increases downward
    PDF: origin bottom-left, Y increases upward

    Args:
        image_points: Points in image coordinates
        image_width: Image width in pixels
        image_height: Image height in pixels
        page_width: Page width in PDF points
        page_height: Page height in PDF points

    Returns:
        Points in PDF coordinates
    """
    scale_x = page_width / image_width
    scale_y = page_height / image_height

    pdf_points = []
    for x, y in image_points:
        pdf_x = x * scale_x
        pdf_y = page_height - (y * scale_y)  # Flip Y
        pdf_points.append((pdf_x, pdf_y))

    return pdf_points


def simplify_contour(
    contour: np.ndarray,
    dpi: int = DEFAULT_RENDER_DPI
) -> np.ndarray:
    """
    Simplify contour using Douglas-Peucker algorithm.

    Args:
        contour: OpenCV contour
        dpi: Rendering DPI for epsilon calculation

    Returns:
        Simplified contour
    """
    epsilon = DP_EPSILON_BASE * (dpi / 300)
    return cv2.approxPolyDP(contour, epsilon, closed=True)


def contour_to_polygon(
    contour: np.ndarray,
    image_width: int,
    image_height: int,
    page_width: float,
    page_height: float,
    dpi: int = DEFAULT_RENDER_DPI
) -> Optional[Polygon]:
    """
    Convert OpenCV contour to Shapely polygon in PDF coordinates.

    Args:
        contour: OpenCV contour
        image_width: Image width
        image_height: Image height
        page_width: Page width in PDF points
        page_height: Page height in PDF points
        dpi: Rendering DPI

    Returns:
        Shapely Polygon or None if invalid
    """
    # Simplify contour
    simplified = simplify_contour(contour, dpi)

    # Need at least MIN_VERTICES points
    if len(simplified) < MIN_VERTICES:
        return None

    # Extract points
    image_points = [(float(pt[0][0]), float(pt[0][1])) for pt in simplified]

    # Transform to PDF coordinates
    pdf_points = transform_image_to_pdf_coords(
        image_points, image_width, image_height,
        page_width, page_height
    )

    # Create polygon
    try:
        polygon = Polygon(pdf_points)

        if not polygon.is_valid:
            polygon = make_valid(polygon)

        if polygon.is_valid and not polygon.is_empty:
            return polygon

    except Exception as e:
        logger.debug(f"Error creating polygon: {e}")

    return None


def detect_rooms_flood_fill(
    binary: np.ndarray,
    label_centroids: List[Tuple[float, float]],
    page_width: float,
    page_height: float,
    dpi: int = DEFAULT_RENDER_DPI,
    tolerance: int = FLOOD_FILL_TOLERANCE
) -> List[RoomPolygon]:
    """
    Detect rooms using flood fill from room label centroids.

    Args:
        binary: Binary image (rooms = white/255)
        label_centroids: List of label centroids in image coordinates
        page_width: Page width in PDF points
        page_height: Page height in PDF points
        dpi: Rendering DPI
        tolerance: Flood fill color tolerance

    Returns:
        List of RoomPolygon objects
    """
    rooms = []
    h, w = binary.shape[:2]
    room_counter = 1

    # Create mask for flood fill (needs +2 pixels on each side)
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    for cx, cy in label_centroids:
        # Convert to integer pixel coordinates
        px, py = int(cx), int(cy)

        if 0 <= px < w and 0 <= py < h:
            # Check if point is on white (room, not wall)
            if binary[py, px] == 255:
                # Create fresh mask
                fill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

                # Flood fill
                cv2.floodFill(
                    binary.copy(),
                    fill_mask,
                    (px, py),
                    128,
                    loDiff=tolerance,
                    upDiff=tolerance
                )

                # Extract filled region (value = 1 in mask)
                filled = fill_mask[1:-1, 1:-1]

                # Find contour of filled region
                contours, _ = cv2.findContours(
                    filled,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                if contours:
                    # Take largest contour
                    largest = max(contours, key=cv2.contourArea)

                    polygon = contour_to_polygon(
                        largest, w, h, page_width, page_height, dpi
                    )

                    polygon = get_largest_polygon(polygon)
                    if polygon and polygon.area > 0:
                        vertices = list(polygon.exterior.coords)[:-1]  # Remove closing point

                        rooms.append(RoomPolygon(
                            polygon_id=f"raster_{room_counter}",
                            vertices=vertices,
                            area_sq_points=polygon.area,
                            shapely_polygon=polygon,
                            source="raster",
                            confidence=Confidence.MEDIUM
                        ))
                        room_counter += 1

    logger.info(f"Flood fill detected {len(rooms)} rooms")
    return rooms


def detect_rooms_connected_components(
    binary: np.ndarray,
    page_width: float,
    page_height: float,
    dpi: int = DEFAULT_RENDER_DPI
) -> List[RoomPolygon]:
    """
    Detect rooms using connected components analysis.

    Args:
        binary: Binary image (walls = black/0, rooms = white/255)
        page_width: Page width in PDF points
        page_height: Page height in PDF points
        dpi: Rendering DPI

    Returns:
        List of RoomPolygon objects
    """
    rooms = []
    h, w = binary.shape[:2]
    image_area = h * w
    min_area = image_area * MIN_CONTOUR_AREA_RATIO
    room_counter = 1

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    for i in range(1, num_labels):  # Skip background (0)
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        if area > image_area * 0.8:  # Too large (probably background)
            continue

        # Create mask for this component
        component_mask = (labels == i).astype(np.uint8) * 255

        # Find contour
        contours, _ = cv2.findContours(
            component_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            largest = max(contours, key=cv2.contourArea)

            polygon = contour_to_polygon(
                largest, w, h, page_width, page_height, dpi
            )

            polygon = get_largest_polygon(polygon)
            if polygon and polygon.area > 0:
                vertices = list(polygon.exterior.coords)[:-1]

                rooms.append(RoomPolygon(
                    polygon_id=f"raster_{room_counter}",
                    vertices=vertices,
                    area_sq_points=polygon.area,
                    shapely_polygon=polygon,
                    source="raster",
                    confidence=Confidence.LOW
                ))
                room_counter += 1

    logger.info(f"Connected components detected {len(rooms)} rooms")
    return rooms


def detect_rooms_contours(
    binary: np.ndarray,
    page_width: float,
    page_height: float,
    dpi: int = DEFAULT_RENDER_DPI
) -> List[RoomPolygon]:
    """
    Detect rooms using contour detection.

    Args:
        binary: Binary image
        page_width: Page width in PDF points
        page_height: Page height in PDF points
        dpi: Rendering DPI

    Returns:
        List of RoomPolygon objects
    """
    rooms = []
    h, w = binary.shape[:2]
    image_area = h * w
    min_area = image_area * MIN_CONTOUR_AREA_RATIO
    max_area = image_area * 0.8
    room_counter = 1

    # Find all contours
    contours, hierarchy = cv2.findContours(
        binary,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area or area > max_area:
            continue

        polygon = contour_to_polygon(
            contour, w, h, page_width, page_height, dpi
        )

        polygon = get_largest_polygon(polygon)
        if polygon and polygon.area > 0:
            vertices = list(polygon.exterior.coords)[:-1]

            rooms.append(RoomPolygon(
                polygon_id=f"raster_{room_counter}",
                vertices=vertices,
                area_sq_points=polygon.area,
                shapely_polygon=polygon,
                source="raster",
                confidence=Confidence.LOW
            ))
            room_counter += 1

    logger.info(f"Contour detection found {len(rooms)} rooms")
    return rooms


def assess_quality(rooms: List[RoomPolygon], page_area: float) -> str:
    """
    Assess quality of raster detection results.

    Args:
        rooms: List of detected rooms
        page_area: Page area in PDF points squared

    Returns:
        Quality string: "GOOD", "SUSPECT", or "POOR"
    """
    if len(rooms) < 3:
        return "POOR"

    if len(rooms) > 50:
        return "SUSPECT"

    # Check for one giant room
    max_area = max(r.area_sq_points for r in rooms) if rooms else 0
    if max_area > page_area * 0.8:
        return "POOR"

    # Check if all rooms similar size
    if len(rooms) >= 3:
        areas = sorted([r.area_sq_points for r in rooms])
        median_area = areas[len(areas) // 2]
        similar_count = sum(1 for a in areas if abs(a - median_area) / median_area < 0.2)
        if similar_count == len(rooms):
            return "SUSPECT"

    return "GOOD"


def detect_rooms_from_raster(
    binary: np.ndarray,
    page_width: float,
    page_height: float,
    dpi: int = DEFAULT_RENDER_DPI,
    label_centroids: Optional[List[Tuple[float, float]]] = None
) -> RasterDetectionResult:
    """
    Detect rooms from raster image using appropriate method.

    Args:
        binary: Preprocessed binary image
        page_width: Page width in PDF points
        page_height: Page height in PDF points
        dpi: Rendering DPI
        label_centroids: Optional room label centroids for flood fill

    Returns:
        RasterDetectionResult with rooms and quality assessment
    """
    page_area = page_width * page_height

    # Try flood fill first if labels available
    if label_centroids and len(label_centroids) > 0:
        rooms = detect_rooms_flood_fill(
            binary, label_centroids, page_width, page_height, dpi
        )

        if len(rooms) >= 3:
            quality = assess_quality(rooms, page_area)
            return RasterDetectionResult(
                rooms=rooms,
                method="flood_fill",
                quality=quality,
                details={"label_count": len(label_centroids), "room_count": len(rooms)}
            )
        else:
            logger.info("Flood fill found few rooms, falling back to connected components")

    # Fall back to connected components
    rooms = detect_rooms_connected_components(
        binary, page_width, page_height, dpi
    )

    if len(rooms) >= 3:
        quality = assess_quality(rooms, page_area)
        return RasterDetectionResult(
            rooms=rooms,
            method="connected_components",
            quality=quality,
            details={"room_count": len(rooms)}
        )

    # Last resort: contour detection
    rooms = detect_rooms_contours(
        binary, page_width, page_height, dpi
    )

    quality = assess_quality(rooms, page_area)
    return RasterDetectionResult(
        rooms=rooms,
        method="contours",
        quality=quality,
        details={"room_count": len(rooms)}
    )
