"""
Room Data Structure Module

Defines the Room class with all measurement fields.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

from shapely.geometry import Polygon

from ..constants import Confidence

logger = logging.getLogger(__name__)


@dataclass
class Room:
    """
    Complete room object with all measurements.

    Represents a detected room from a blueprint with geometry and metadata.
    """
    # Identification
    room_id: str
    room_name: str

    # Location
    sheet_number: int  # 0-indexed page number
    sheet_name: str = ""  # From PDF if available
    floor_level: str = ""  # Extracted from sheet name

    # Geometry in PDF coordinates
    polygon_pdf_points: List[Tuple[float, float]] = field(default_factory=list)
    polygon_real_units: List[Tuple[float, float]] = field(default_factory=list)
    shapely_polygon: Optional[Polygon] = None

    # Measurements (imperial by default)
    floor_area_sqft: float = 0.0
    perimeter_ft: float = 0.0
    ceiling_height_ft: float = 0.0
    wall_area_sqft: float = 0.0
    ceiling_area_sqft: float = 0.0

    # Detection info
    source: str = "vector"  # "vector" or "raster"
    confidence: str = Confidence.MEDIUM
    name_confidence: str = Confidence.NONE
    height_confidence: str = Confidence.NONE

    # Scale info
    scale_factor: float = 0.0
    scale_source: str = "default"

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Validation warnings
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert room to dictionary for JSON serialization."""
        return {
            "room_id": self.room_id,
            "room_name": self.room_name,
            "sheet_number": self.sheet_number,
            "sheet_name": self.sheet_name,
            "floor_level": self.floor_level,
            "polygon_pdf_points": self.polygon_pdf_points,
            "polygon_real_units": self.polygon_real_units,
            "floor_area_sqft": round(self.floor_area_sqft, 2),
            "perimeter_ft": round(self.perimeter_ft, 2),
            "ceiling_height_ft": round(self.ceiling_height_ft, 2),
            "wall_area_sqft": round(self.wall_area_sqft, 2),
            "ceiling_area_sqft": round(self.ceiling_area_sqft, 2),
            "source": self.source,
            "confidence": self.confidence,
            "name_confidence": self.name_confidence,
            "height_confidence": self.height_confidence,
            "scale_factor": round(self.scale_factor, 2),
            "scale_source": self.scale_source,
            "metadata": self.metadata,
            "warnings": self.warnings,
        }

    def to_csv_row(self) -> List[Any]:
        """Convert room to CSV row values."""
        return [
            self.room_id,
            self.room_name,
            self.sheet_number,
            self.sheet_name,
            self.floor_level,
            round(self.floor_area_sqft, 2),
            round(self.perimeter_ft, 2),
            round(self.ceiling_height_ft, 2),
            round(self.wall_area_sqft, 2),
            round(self.ceiling_area_sqft, 2),
            self.source,
            self.confidence,
        ]

    @staticmethod
    def csv_header() -> List[str]:
        """Return CSV header row."""
        return [
            "room_id",
            "room_name",
            "sheet_number",
            "sheet_name",
            "floor_level",
            "floor_area_sqft",
            "perimeter_ft",
            "ceiling_height_ft",
            "wall_area_sqft",
            "ceiling_area_sqft",
            "source",
            "confidence",
        ]


def extract_floor_level(sheet_name: str) -> str:
    """
    Extract floor level from sheet name.

    Examples:
        "A1.1 - LEVEL 1 FLOOR PLAN" -> "LEVEL 1"
        "FIRST FLOOR PLAN" -> "FIRST FLOOR"
        "2ND FLOOR" -> "2ND FLOOR"

    Args:
        sheet_name: Sheet name from PDF

    Returns:
        Extracted floor level or empty string
    """
    import re

    sheet_upper = sheet_name.upper()

    # Try common patterns
    patterns = [
        r"(LEVEL\s*\d+)",
        r"(FLOOR\s*\d+)",
        r"(\d+(?:ST|ND|RD|TH)\s*FLOOR)",
        r"(FIRST\s*FLOOR)",
        r"(SECOND\s*FLOOR)",
        r"(THIRD\s*FLOOR)",
        r"(GROUND\s*FLOOR)",
        r"(BASEMENT)",
        r"(MEZZANINE)",
    ]

    for pattern in patterns:
        match = re.search(pattern, sheet_upper)
        if match:
            return match.group(1)

    return ""
