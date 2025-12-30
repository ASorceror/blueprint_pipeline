"""
Label Matcher Module

Filters text to identify room labels and matches them to room polygons.
"""

import logging
import math
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional

from shapely.geometry import Point, Polygon

from ..constants import (
    ROOM_LABEL_MIN_FONTSIZE,
    ROOM_LABEL_MAX_FONTSIZE,
    MIN_LABEL_LENGTH,
    MAX_LABEL_LENGTH,
    OCR_CONFIDENCE_THRESHOLD,
    LABEL_SEARCH_EXPANSION_RATIO,
    Confidence,
)
from .ocr_engine import TextBlock
from ..vector.polygonizer import RoomPolygon

logger = logging.getLogger(__name__)


@dataclass
class RoomLabel:
    """A validated room label."""
    text: str
    bbox: Tuple[float, float, float, float]
    centroid: Tuple[float, float]
    confidence: float
    source: str
    is_matched: bool = False


@dataclass
class LabelMatch:
    """A match between a room label and a room polygon."""
    polygon_id: str
    room_name: str
    name_confidence: str
    name_source: str
    label: Optional[RoomLabel] = None


# Room type patterns (case-insensitive)
ROOM_TYPE_PATTERNS = [
    # Office spaces
    r"\bOFFICE\b", r"\bOFF\b", r"\bCONF\b", r"\bCONFERENCE\b", r"\bMEETING\b",
    # Kitchen/Break
    r"\bKITCHEN\b", r"\bBREAK\s*ROOM\b", r"\bBREAK\b", r"\bKITCHENETTE\b",
    # Restrooms
    r"\bRESTROOM\b", r"\bTOILET\b", r"\bMEN\b", r"\bWOMEN\b", r"\bUNISEX\b", r"\bRR\b",
    # Storage
    r"\bSTORAGE\b", r"\bSTOR\b", r"\bCLOSET\b", r"\bJANITOR\b",
    # Entry/Lobby
    r"\bLOBBY\b", r"\bRECEPTION\b", r"\bVESTIBULE\b", r"\bENTRY\b",
    # Corridors
    r"\bCORRIDOR\b", r"\bCORR\b", r"\bHALL\b", r"\bHALLWAY\b",
    # Mechanical/Electrical
    r"\bMECHANICAL\b", r"\bMECH\b", r"\bELECTRICAL\b", r"\bELEC\b",
    r"\bTELECOM\b", r"\bIDF\b", r"\bMDF\b",
    # Industrial
    r"\bWAREHOUSE\b", r"\bLOADING\b", r"\bDOCK\b",
    # Vertical circulation
    r"\bSTAIR\b", r"\bSTAIRWELL\b", r"\bELEVATOR\b", r"\bELEV\b",
    # Residential
    r"\bBEDROOM\b", r"\bLIVING\b", r"\bDINING\b", r"\bBATHROOM\b", r"\bBATH\b",
]

# Numbered room patterns
NUMBERED_ROOM_PATTERNS = [
    r"^\d{2,4}$",           # 101, 205, 1001
    r"^[A-Z]\d{2,4}$",      # A101, B205
    r"^\d{2,4}[A-Z]$",      # 101A, 205B
    r"^\d+-\d{2,4}$",       # 1-101, 2-205
    r"^[A-Z]-\d{2,4}$",     # A-101, B-205
]

# Exclude patterns (dimension, scale, notes, drawing numbers)
EXCLUDE_PATTERNS = [
    r"['\"]",                      # Dimension text (feet/inches)
    r"\d+\s*(ft|feet|inch|in|mm|cm|m)\b",  # Unit measurements
    r"=",                          # Scale text
    r":",                          # Scale ratio text
    r"^NOTE\b",                    # Notes
    r"^SEE\b",                     # References
    r"^REF\b",                     # References
    r"^TYP\b",                     # Typical
    r"^[ASMEP]\d+\.\d+",          # Drawing numbers like A1.01
    r"^\d+/\d+",                   # Fractions
]


def get_text_centroid(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Get the center point of a bounding box."""
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2, (y0 + y1) / 2)


def is_room_label_pattern(text: str) -> bool:
    """Check if text matches room label patterns."""
    text_upper = text.upper().strip()

    # Check explicit room type patterns
    for pattern in ROOM_TYPE_PATTERNS:
        if re.search(pattern, text_upper, re.IGNORECASE):
            return True

    # Check numbered room patterns
    for pattern in NUMBERED_ROOM_PATTERNS:
        if re.match(pattern, text_upper):
            return True

    return False


def is_excluded_pattern(text: str) -> bool:
    """Check if text matches exclusion patterns."""
    text_upper = text.upper().strip()

    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, text_upper, re.IGNORECASE):
            return True

    # Exclude if ALL CAPS and > 20 characters (likely title)
    if text_upper == text and len(text) > 20 and text.isalpha():
        return True

    return False


def filter_room_labels(text_blocks: List[TextBlock]) -> List[RoomLabel]:
    """
    Filter text blocks to identify likely room labels.

    Args:
        text_blocks: List of TextBlock objects

    Returns:
        List of RoomLabel objects
    """
    room_labels = []

    for block in text_blocks:
        text = block.text.strip()

        # Length filter
        if len(text) < MIN_LABEL_LENGTH or len(text) > MAX_LABEL_LENGTH:
            continue

        # Font size filter (if available)
        if block.font_size is not None:
            if block.font_size < ROOM_LABEL_MIN_FONTSIZE:
                continue
            if block.font_size > ROOM_LABEL_MAX_FONTSIZE:
                continue

        # Confidence filter (for OCR)
        if block.source == "ocr" and block.confidence < OCR_CONFIDENCE_THRESHOLD:
            continue

        # Exclusion patterns
        if is_excluded_pattern(text):
            continue

        # Room label patterns (must match at least one)
        if not is_room_label_pattern(text):
            continue

        centroid = get_text_centroid(block.bbox)

        room_labels.append(RoomLabel(
            text=text,
            bbox=block.bbox,
            centroid=centroid,
            confidence=block.confidence,
            source=block.source
        ))

    logger.info(f"Filtered {len(room_labels)} room labels from {len(text_blocks)} text blocks")
    return room_labels


def polygon_diagonal(polygon: RoomPolygon) -> float:
    """Calculate the diagonal of a polygon's bounding box."""
    bounds = polygon.shapely_polygon.bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    return math.sqrt(width ** 2 + height ** 2)


def expand_polygon_bounds(
    polygon: RoomPolygon,
    expansion_ratio: float
) -> Tuple[float, float, float, float]:
    """Expand polygon bounding box by a ratio."""
    bounds = polygon.shapely_polygon.bounds
    x0, y0, x1, y1 = bounds

    width = x1 - x0
    height = y1 - y0

    expand_x = width * expansion_ratio
    expand_y = height * expansion_ratio

    return (
        x0 - expand_x,
        y0 - expand_y,
        x1 + expand_x,
        y1 + expand_y
    )


def point_in_bounds(
    point: Tuple[float, float],
    bounds: Tuple[float, float, float, float]
) -> bool:
    """Check if point is within bounds."""
    x, y = point
    x0, y0, x1, y1 = bounds
    return x0 <= x <= x1 and y0 <= y <= y1


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate distance between two points."""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def match_labels_to_polygons(
    room_labels: List[RoomLabel],
    room_polygons: List[RoomPolygon]
) -> List[LabelMatch]:
    """
    Match room labels to room polygons.

    Uses three methods in order:
    1. Centroid containment (HIGH confidence)
    2. Expanded bounds containment (MEDIUM confidence)
    3. Nearest label (LOW confidence)

    Args:
        room_labels: List of room labels
        room_polygons: List of room polygons

    Returns:
        List of LabelMatch objects (one per polygon)
    """
    matches = []
    unmatched_labels = set(range(len(room_labels)))
    matched_polygons = set()

    # Track auto-increment for unnamed rooms
    auto_room_counter = 1

    for poly in room_polygons:
        match_found = False
        match_label = None
        match_confidence = Confidence.NONE
        match_source = "auto"

        poly_centroid = poly.shapely_polygon.centroid

        # Method 1: Centroid containment (HIGH confidence)
        for i in list(unmatched_labels):
            label = room_labels[i]
            label_point = Point(label.centroid)

            if poly.shapely_polygon.contains(label_point):
                match_found = True
                match_label = label
                match_confidence = Confidence.HIGH
                match_source = label.source
                label.is_matched = True
                unmatched_labels.discard(i)
                break

        # Method 2: Expanded bounds containment (MEDIUM confidence)
        if not match_found:
            expanded_bounds = expand_polygon_bounds(poly, LABEL_SEARCH_EXPANSION_RATIO)

            for i in list(unmatched_labels):
                label = room_labels[i]

                if point_in_bounds(label.centroid, expanded_bounds):
                    match_found = True
                    match_label = label
                    match_confidence = Confidence.MEDIUM
                    match_source = label.source
                    label.is_matched = True
                    unmatched_labels.discard(i)
                    break

        # Method 3: Nearest label (LOW confidence)
        if not match_found and unmatched_labels:
            poly_diag = polygon_diagonal(poly)
            max_distance = poly_diag * 0.5

            best_label_idx = None
            best_distance = float('inf')

            for i in unmatched_labels:
                label = room_labels[i]
                dist = distance(label.centroid, (poly_centroid.x, poly_centroid.y))

                if dist < best_distance and dist < max_distance:
                    best_distance = dist
                    best_label_idx = i

            if best_label_idx is not None:
                match_found = True
                match_label = room_labels[best_label_idx]
                match_confidence = Confidence.LOW
                match_source = match_label.source
                match_label.is_matched = True
                unmatched_labels.discard(best_label_idx)

        # Create match result
        if match_label:
            room_name = match_label.text
        else:
            room_name = f"ROOM_{auto_room_counter}"
            auto_room_counter += 1

        matches.append(LabelMatch(
            polygon_id=poly.polygon_id,
            room_name=room_name,
            name_confidence=match_confidence,
            name_source=match_source,
            label=match_label
        ))

        matched_polygons.add(poly.polygon_id)

    # Log unmatched labels for debugging
    if unmatched_labels:
        unmatched_texts = [room_labels[i].text for i in unmatched_labels]
        logger.debug(f"Unmatched labels: {unmatched_texts}")

    # Log match statistics
    high_count = sum(1 for m in matches if m.name_confidence == Confidence.HIGH)
    medium_count = sum(1 for m in matches if m.name_confidence == Confidence.MEDIUM)
    low_count = sum(1 for m in matches if m.name_confidence == Confidence.LOW)
    none_count = sum(1 for m in matches if m.name_confidence == Confidence.NONE)

    logger.info(
        f"Label matching: {high_count} HIGH, {medium_count} MEDIUM, "
        f"{low_count} LOW, {none_count} auto-named"
    )

    return matches


def apply_labels_to_polygons(
    room_polygons: List[RoomPolygon],
    matches: List[LabelMatch]
) -> List[dict]:
    """
    Apply label matches to room polygons to create final room data.

    Args:
        room_polygons: List of room polygons
        matches: List of label matches

    Returns:
        List of room dictionaries with all data
    """
    # Create lookup from polygon_id to match
    match_lookup = {m.polygon_id: m for m in matches}

    rooms = []
    for poly in room_polygons:
        match = match_lookup.get(poly.polygon_id)

        room = {
            "polygon_id": poly.polygon_id,
            "vertices": poly.vertices,
            "area_sq_points": poly.area_sq_points,
            "source": poly.source,
            "polygon_confidence": poly.confidence,
            "room_name": match.room_name if match else f"ROOM_UNKNOWN",
            "name_confidence": match.name_confidence if match else Confidence.NONE,
            "name_source": match.name_source if match else "auto",
        }
        rooms.append(room)

    return rooms
