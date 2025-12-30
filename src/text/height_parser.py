"""
Height Parser Module

Extracts ceiling heights from RCP sheets and matches to floor plan rooms.
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import pymupdf

from ..constants import (
    MIN_CEILING_HEIGHT_FT,
    MAX_CEILING_HEIGHT_FT,
    DEFAULT_CEILING_HEIGHT_FT,
    HEIGHT_LABEL_SEARCH_RADIUS_RATIO,
    Confidence,
)
from .ocr_engine import TextBlock, extract_all_text

logger = logging.getLogger(__name__)


@dataclass
class HeightAnnotation:
    """A ceiling height annotation found on a page."""
    location: Tuple[float, float]  # Centroid in PDF coordinates
    height_feet: float
    raw_text: str
    confidence: float
    source: str  # "embedded" or "ocr"
    room_name: Optional[str] = None  # Extracted room name if found nearby


# Height annotation patterns (case-insensitive)
# Imperial patterns
HEIGHT_PATTERNS_IMPERIAL = [
    # CLG patterns with feet-inches: CLG 9'-6", CLG: 9'0", CLG @ 9 FT
    r"CLG[:\s@]*(\d+)['\'][-\s]?(\d+)?[\"″]?",
    # CEIL patterns with feet-inches: CEIL 10'-6", CEILING: 10'
    r"CEIL(?:ING)?[:\s@]*(\d+)['\'][-\s]?(\d+)?[\"″]?",
    # A.F.F. patterns with feet-inches: A.F.F. 9'-6", AFF 9'6"
    r"A\.?F\.?F\.?[:\s@]*(\d+)['\'][-\s]?(\d+)?[\"″]?",
    # Height at end: 9'-0" CLG, 10 FT CEILING
    r"(\d+)['\'][-\s]?(\d+)?[\"″]?\s*(?:CLG|CEIL(?:ING)?)",
    # Relative heights: +9'-0", +10'
    r"\+(\d+)['\'][-\s]?(\d+)?[\"″]?",
    # Simple feet: 10 FT, 9 FEET
    r"(\d+)\s*(?:FT|FEET)\b",
]

# Metric patterns
HEIGHT_PATTERNS_METRIC = [
    # Millimeters: CLG 2700mm, CLG 2700 mm
    r"CLG[:\s@]*(\d+)\s*(?:mm|MM)",
    r"CEIL(?:ING)?[:\s@]*(\d+)\s*(?:mm|MM)",
    # Meters: CLG 2.7m, CEILING 3.0m
    r"CLG[:\s@]*(\d+(?:\.\d+)?)\s*(?:m|M)(?!m)",
    r"CEIL(?:ING)?[:\s@]*(\d+(?:\.\d+)?)\s*(?:m|M)(?!m)",
]

# Room number/name extraction pattern
ROOM_NUMBER_PATTERN = re.compile(r"(?:^|\s)(\d{2,4}[A-Z]?|[A-Z]-?\d{2,4})(?:\s|$)")


def get_text_centroid(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Get the center point of a bounding box."""
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2, (y0 + y1) / 2)


def parse_imperial_height(feet: str, inches: Optional[str] = None) -> float:
    """Parse imperial height to feet."""
    feet_val = float(feet)
    inches_val = float(inches) if inches else 0
    return feet_val + (inches_val / 12.0)


def parse_metric_height_to_feet(value: float, unit: str) -> float:
    """Convert metric height to feet."""
    if unit.lower() == "mm":
        meters = value / 1000
    else:  # meters
        meters = value
    return meters * 3.28084  # meters to feet


def extract_height_from_text(text: str) -> Optional[Tuple[float, str]]:
    """
    Extract ceiling height from text.

    Args:
        text: Text to parse

    Returns:
        Tuple of (height_in_feet, original_match) or None
    """
    text_upper = text.upper().strip()

    # Try imperial patterns first
    for pattern in HEIGHT_PATTERNS_IMPERIAL:
        match = re.search(pattern, text_upper, re.IGNORECASE)
        if match:
            groups = match.groups()
            if len(groups) >= 1:
                feet = groups[0]
                inches = groups[1] if len(groups) > 1 else None
                height = parse_imperial_height(feet, inches)

                # Validate height
                if MIN_CEILING_HEIGHT_FT <= height <= MAX_CEILING_HEIGHT_FT:
                    return (height, match.group())

    # Try metric patterns
    for pattern in HEIGHT_PATTERNS_METRIC:
        match = re.search(pattern, text_upper, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            # Determine unit from pattern
            unit = "mm" if "mm" in pattern.lower() else "m"
            height = parse_metric_height_to_feet(value, unit)

            # Validate height
            if MIN_CEILING_HEIGHT_FT <= height <= MAX_CEILING_HEIGHT_FT:
                return (height, match.group())

    return None


def extract_room_number(text: str) -> Optional[str]:
    """Extract room number from text if present."""
    match = ROOM_NUMBER_PATTERN.search(text.upper())
    if match:
        return match.group(1)
    return None


def extract_heights_from_page(
    page: pymupdf.Page,
    page_image=None,
    force_ocr: bool = False
) -> List[HeightAnnotation]:
    """
    Extract all ceiling height annotations from a page.

    Args:
        page: pymupdf.Page object (typically an RCP page)
        page_image: Optional rendered image for OCR
        force_ocr: Force OCR even if embedded text exists

    Returns:
        List of HeightAnnotation objects
    """
    # Get all text from page
    text_blocks = extract_all_text(page, page_image, force_ocr)

    heights = []
    for block in text_blocks:
        result = extract_height_from_text(block.text)
        if result:
            height_feet, raw_match = result
            location = get_text_centroid(block.bbox)

            # Try to extract room number from the same text
            room_name = extract_room_number(block.text)

            heights.append(HeightAnnotation(
                location=location,
                height_feet=height_feet,
                raw_text=raw_match,
                confidence=block.confidence,
                source=block.source,
                room_name=room_name
            ))

    logger.info(f"Found {len(heights)} height annotations on page")
    return heights


def normalize_room_name(name: str) -> str:
    """Normalize room name for matching."""
    return re.sub(r'[\s\-_]+', '', name.upper())


def extract_room_number_from_name(name: str) -> Optional[str]:
    """Extract just the numeric portion of a room name."""
    match = re.search(r'(\d{2,4})', name)
    if match:
        return match.group(1)
    return None


def match_heights_to_rooms(
    floor_plan_rooms: List[Dict],
    rcp_heights: List[HeightAnnotation],
    page_width: float
) -> List[Dict]:
    """
    Match RCP height annotations to floor plan rooms.

    Uses multiple matching methods in order of confidence:
    1. Match by room name (HIGH)
    2. Match by room number (MEDIUM)
    3. Match by location (LOW)
    4. Default height (NONE)

    Args:
        floor_plan_rooms: List of room dictionaries with room_name
        rcp_heights: List of HeightAnnotation from RCP
        page_width: Page width for location-based matching

    Returns:
        Updated room list with ceiling_height_ft, height_confidence, height_source
    """
    # Build lookup dictionaries
    height_by_name = {}
    height_by_number = {}
    height_by_location = []

    for h in rcp_heights:
        if h.room_name:
            normalized = normalize_room_name(h.room_name)
            height_by_name[normalized] = h
            # Also extract just the number
            num = extract_room_number_from_name(h.room_name)
            if num:
                height_by_number[num] = h

        height_by_location.append(h)

    # Match rooms to heights
    for room in floor_plan_rooms:
        room_name = room.get("room_name", "")
        matched = False

        # Method 1: Match by room name
        normalized_name = normalize_room_name(room_name)
        if normalized_name in height_by_name:
            h = height_by_name[normalized_name]
            room["ceiling_height_ft"] = h.height_feet
            room["height_confidence"] = Confidence.HIGH
            room["height_source"] = "rcp_name_match"
            matched = True
            logger.debug(f"Height matched by name: {room_name} -> {h.height_feet}'")

        # Method 2: Match by room number
        if not matched:
            room_num = extract_room_number_from_name(room_name)
            if room_num and room_num in height_by_number:
                h = height_by_number[room_num]
                room["ceiling_height_ft"] = h.height_feet
                room["height_confidence"] = Confidence.MEDIUM
                room["height_source"] = "rcp_number_match"
                matched = True
                logger.debug(f"Height matched by number: {room_name} -> {h.height_feet}'")

        # Method 3: Match by location (nearest height)
        if not matched and height_by_location:
            # Get room centroid from vertices if available
            vertices = room.get("vertices", [])
            if vertices:
                xs = [v[0] for v in vertices]
                ys = [v[1] for v in vertices]
                room_centroid = (sum(xs) / len(xs), sum(ys) / len(ys))

                # Find nearest height annotation
                max_distance = page_width * HEIGHT_LABEL_SEARCH_RADIUS_RATIO
                best_height = None
                best_distance = float('inf')

                for h in height_by_location:
                    dist = ((h.location[0] - room_centroid[0]) ** 2 +
                            (h.location[1] - room_centroid[1]) ** 2) ** 0.5
                    if dist < best_distance and dist < max_distance:
                        best_distance = dist
                        best_height = h

                if best_height:
                    room["ceiling_height_ft"] = best_height.height_feet
                    room["height_confidence"] = Confidence.LOW
                    room["height_source"] = "rcp_location"
                    matched = True
                    logger.debug(f"Height matched by location: {room_name} -> {best_height.height_feet}'")

        # Method 4: Default
        if not matched:
            room["ceiling_height_ft"] = DEFAULT_CEILING_HEIGHT_FT
            room["height_confidence"] = Confidence.NONE
            room["height_source"] = "default"
            logger.debug(f"Height defaulted: {room_name} -> {DEFAULT_CEILING_HEIGHT_FT}'")

    # Log statistics
    high_count = sum(1 for r in floor_plan_rooms if r.get("height_confidence") == Confidence.HIGH)
    medium_count = sum(1 for r in floor_plan_rooms if r.get("height_confidence") == Confidence.MEDIUM)
    low_count = sum(1 for r in floor_plan_rooms if r.get("height_confidence") == Confidence.LOW)
    none_count = sum(1 for r in floor_plan_rooms if r.get("height_confidence") == Confidence.NONE)

    logger.info(
        f"Height matching: {high_count} HIGH, {medium_count} MEDIUM, "
        f"{low_count} LOW, {none_count} default"
    )

    return floor_plan_rooms
