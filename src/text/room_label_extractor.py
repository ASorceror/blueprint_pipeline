"""
Room Label Extractor Module

Enhanced label extraction for label-driven room detection.
Designed to be the PRIMARY source of room information, not secondary.

Key differences from label_matcher.py:
- Extracts labels as primary data source
- More aggressive pattern matching
- Better confidence scoring
- Clustering for multi-label rooms
- Designed for label-driven pipeline
"""

import logging
import math
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
from enum import Enum

from shapely.geometry import Point, box
from shapely.strtree import STRtree

from ..constants import (
    POINTS_PER_INCH,
)
from .ocr_engine import TextBlock

logger = logging.getLogger(__name__)


class LabelType(Enum):
    """Types of room labels."""
    ROOM_NUMBER = "room_number"      # 100, 101A, A-101
    ROOM_NAME = "room_name"          # OFFICE, LOBBY
    AREA_TAG = "area_tag"            # P100, C-01
    COMBINED = "combined"            # 101 OFFICE
    UNKNOWN = "unknown"


@dataclass
class ExtractedLabel:
    """
    An extracted room label with full metadata.

    This is the PRIMARY data source for label-driven detection.
    """
    text: str
    original_text: str              # Before normalization
    label_type: LabelType
    bbox: Tuple[float, float, float, float]
    centroid: Tuple[float, float]
    confidence: float               # 0-1, combined OCR + pattern confidence
    ocr_confidence: float           # Raw OCR confidence
    pattern_confidence: float       # Pattern match confidence
    source: str                     # "ocr", "embedded", "hybrid"
    font_size: Optional[float] = None
    is_in_drawing_area: bool = True
    cluster_id: Optional[int] = None
    matched_patterns: List[str] = field(default_factory=list)


@dataclass
class LabelCluster:
    """
    A cluster of labels that may belong to the same room.

    Some rooms have multiple labels (number + name, or split text).
    """
    labels: List[ExtractedLabel]
    primary_label: ExtractedLabel
    centroid: Tuple[float, float]
    combined_text: str


# === PATTERN DEFINITIONS ===

# Room number patterns (HIGH confidence)
ROOM_NUMBER_PATTERNS = [
    (r'^(\d{3})$', 0.95),                    # 100, 201, 305
    (r'^(\d{3}[A-Z])$', 0.95),               # 100A, 201B
    (r'^([A-Z])(\d{3})$', 0.90),             # A100, B201
    (r'^(\d{2,3})-(\d{2,3})$', 0.85),        # 1-101, 2-205
    (r'^([A-Z])-(\d{2,3})$', 0.90),          # A-101, B-205
    (r'^(\d{4})$', 0.85),                    # 1001, 2005 (4-digit)
    (r'^(\d{2})$', 0.70),                    # 10, 15 (2-digit, lower confidence)
]

# Area tag patterns (MEDIUM-HIGH confidence)
AREA_TAG_PATTERNS = [
    (r'^P(\d+)$', 0.90),                     # P100, P12 (parking)
    (r'^C-?(\d+)$', 0.90),                   # C-01, C01 (corridor)
    (r'^[ABCD]-(\d{2})$', 0.85),             # A-01, B-02 (lettered areas)
    (r'^(ST|STAIR)-?(\d*)$', 0.90),          # ST1, STAIR-2
    (r'^(EL|ELEV)-?(\d*)$', 0.90),           # EL1, ELEV-2
]

# Room name patterns (MEDIUM confidence)
ROOM_NAME_PATTERNS = [
    (r'^OFFICE$', 0.90),
    (r'^LOBBY$', 0.90),
    (r'^RESTROOM$', 0.90),
    (r'^CORRIDOR$', 0.85),
    (r'^CORR\.?$', 0.80),
    (r'^STORAGE$', 0.90),
    (r'^STOR\.?$', 0.80),
    (r'^MECHANICAL$', 0.90),
    (r'^MECH\.?$', 0.80),
    (r'^ELECTRICAL$', 0.90),
    (r'^ELEC\.?$', 0.80),
    (r'^CONFERENCE$', 0.90),
    (r'^CONF\.?$', 0.80),
    (r'^BREAK\s*ROOM$', 0.90),
    (r'^KITCHEN$', 0.90),
    (r'^(MEN|WOMEN|UNISEX)$', 0.85),
    (r'^ELEVATOR$', 0.90),
    (r'^STAIR(WELL)?$', 0.90),
    (r'^VESTIBULE$', 0.90),
    (r'^RECEPTION$', 0.90),
    (r'^ENTRY$', 0.85),
    (r'^HALL(WAY)?$', 0.85),
    (r'^JANITOR$', 0.85),
    (r'^CLOSET$', 0.85),
    (r'^TELECOM$', 0.85),
    (r'^(IDF|MDF)$', 0.85),
    (r'^WAREHOUSE$', 0.90),
    (r'^LOADING$', 0.85),
    (r'^DOCK$', 0.85),
]

# Patterns to EXCLUDE (definitely not room labels)
EXCLUSION_PATTERNS = [
    r'[\'"\-]\s*\d+',                # Dimension text: 10'-6", -5"
    r'\d+\s*[\'"]',                  # More dimension: 10', 6"
    r'\d+\s*(ft|feet|in|inch|mm|cm|m)\b',  # Units
    r'^[\d\.]+\s*[\'"]',             # Leading numbers with units
    r'[=:]',                         # Scale text
    r'^NOTE[S]?\b',                  # Notes
    r'^SEE\s',                       # References
    r'^REF\.?\b',                    # References
    r'^TYP\.?\b',                    # Typical
    r'^[ASMEP]\d+\.\d+',             # Drawing numbers: A1.01
    r'^\d+/\d+',                     # Fractions
    r'^@',                           # Symbols
    r'^\(.*\)$',                     # Parenthetical
    r'^\d+\s*SF$',                   # Area labels: 100 SF
    r'^[+-]?\d+\.\d{2,}$',           # Decimal numbers (elevations)
    r'^SCALE\b',                     # Scale text
    r'^NORTH$',                      # Orientation
    r'^N\.T\.S\.$',                  # Not to scale
    r'^EXIST',                       # Existing
    r'^NEW\b',                       # New
    r'^DEMO',                        # Demolition
    r'^\d+\s*X\s*\d+',               # Dimensions like 10 X 12
    r'^[A-Z]{2,}\d*$',               # Grid lines like AA, BB
    # Sheet/drawing references (NOT room numbers):
    r'^[ASMEP]\d{3}$',               # A101, S201, M301, E401, P501
    r'^[ASMEP]\d{2}$',               # A10, S20, M30
    r'^[ASMEP]-\d+$',                # A-1, S-2
]

# Constants
MIN_LABEL_LENGTH = 1
MAX_LABEL_LENGTH = 30
MIN_FONT_SIZE = 4.0
MAX_FONT_SIZE = 24.0
OCR_CONFIDENCE_THRESHOLD = 50
CLUSTER_DISTANCE_THRESHOLD = 50  # points


class RoomLabelExtractor:
    """
    Extract and validate room labels from floor plans.

    This is the primary data source for label-driven room detection.
    """

    def __init__(
        self,
        drawing_bounds: Optional[Tuple[float, float, float, float]] = None,
        min_confidence: float = 0.5
    ):
        """
        Initialize extractor.

        Args:
            drawing_bounds: (x0, y0, x1, y1) of drawing area (excludes title block)
            min_confidence: Minimum confidence to include label
        """
        self.drawing_bounds = drawing_bounds
        self.min_confidence = min_confidence

        # Compile patterns for performance
        self._compiled_exclusions = [
            re.compile(p, re.IGNORECASE) for p in EXCLUSION_PATTERNS
        ]
        self._compiled_room_numbers = [
            (re.compile(p, re.IGNORECASE), conf)
            for p, conf in ROOM_NUMBER_PATTERNS
        ]
        self._compiled_area_tags = [
            (re.compile(p, re.IGNORECASE), conf)
            for p, conf in AREA_TAG_PATTERNS
        ]
        self._compiled_room_names = [
            (re.compile(p, re.IGNORECASE), conf)
            for p, conf in ROOM_NAME_PATTERNS
        ]

    def extract(
        self,
        text_blocks: List[TextBlock],
        page_width: float,
        page_height: float
    ) -> List[ExtractedLabel]:
        """
        Extract room labels from text blocks.

        Args:
            text_blocks: List of TextBlock from OCR/embedded text
            page_width: Page width in points
            page_height: Page height in points

        Returns:
            List of ExtractedLabel objects
        """
        labels = []

        for block in text_blocks:
            label = self._process_text_block(block, page_width, page_height)
            if label is not None:
                labels.append(label)

        # Remove duplicates (same text near same location)
        labels = self._remove_duplicates(labels)

        # Filter out grid line labels (at page edges)
        labels = self._filter_grid_lines(labels, page_width, page_height)

        # Filter by confidence
        labels = [l for l in labels if l.confidence >= self.min_confidence]

        logger.info(
            f"Extracted {len(labels)} room labels from {len(text_blocks)} text blocks"
        )

        return labels

    def _filter_grid_lines(
        self,
        labels: List[ExtractedLabel],
        page_width: float,
        page_height: float
    ) -> List[ExtractedLabel]:
        """
        Filter out labels that are likely grid line markers.

        Grid lines typically:
        - Are at page edges (top/bottom/left/right margins)
        - Are single or two character labels (A, B, 1, 2, 10, 11)
        - Form regular patterns along one axis
        """
        edge_threshold = 300  # Points from edge to consider "at edge"
        filtered = []

        for label in labels:
            x, y = label.centroid

            # Check if at edge of page (PyMuPDF: y=0 is at TOP)
            at_top_edge = y < edge_threshold
            at_bottom_edge = y > page_height - edge_threshold
            at_left_edge = x < edge_threshold
            at_right_edge = x > page_width - edge_threshold

            is_at_edge = at_top_edge or at_bottom_edge or at_left_edge or at_right_edge

            # Check if likely grid line (short labels at edges)
            is_short_label = len(label.text) <= 2

            # Grid line patterns: single letters, single/double digits
            is_grid_pattern = (
                label.text.isdigit() or
                (label.text.isalpha() and len(label.text) == 1) or
                re.match(r'^\d{1,2}$', label.text)
            )

            if is_at_edge and is_short_label and is_grid_pattern:
                logger.debug(
                    f"Filtering grid line: {label.text} at ({x:.0f}, {y:.0f})"
                )
                continue

            filtered.append(label)

        removed = len(labels) - len(filtered)
        if removed > 0:
            logger.info(f"Filtered {removed} grid line labels")

        return filtered

    def _process_text_block(
        self,
        block: TextBlock,
        page_width: float,
        page_height: float
    ) -> Optional[ExtractedLabel]:
        """Process a single text block."""
        text = block.text.strip()
        original_text = text

        # Basic length filter
        if len(text) < MIN_LABEL_LENGTH or len(text) > MAX_LABEL_LENGTH:
            return None

        # Font size filter
        if block.font_size is not None:
            if block.font_size < MIN_FONT_SIZE or block.font_size > MAX_FONT_SIZE:
                return None

        # OCR confidence filter
        ocr_confidence = block.confidence / 100.0 if block.confidence else 0.8
        if block.source == "ocr" and block.confidence < OCR_CONFIDENCE_THRESHOLD:
            return None

        # Check exclusion patterns
        if self._is_excluded(text):
            return None

        # Normalize text
        text = self._normalize_text(text)

        # Classify and get pattern confidence
        label_type, pattern_confidence, matched_patterns = self._classify_label(text)

        if label_type == LabelType.UNKNOWN:
            return None

        # Check if in drawing area
        centroid = self._get_centroid(block.bbox)
        is_in_drawing = self._is_in_drawing_area(centroid, page_width, page_height)

        # Calculate combined confidence
        combined_confidence = self._calculate_confidence(
            ocr_confidence, pattern_confidence, block.source, is_in_drawing
        )

        return ExtractedLabel(
            text=text,
            original_text=original_text,
            label_type=label_type,
            bbox=block.bbox,
            centroid=centroid,
            confidence=combined_confidence,
            ocr_confidence=ocr_confidence,
            pattern_confidence=pattern_confidence,
            source=block.source,
            font_size=block.font_size,
            is_in_drawing_area=is_in_drawing,
            matched_patterns=matched_patterns
        )

    def _is_excluded(self, text: str) -> bool:
        """Check if text matches exclusion patterns."""
        for pattern in self._compiled_exclusions:
            if pattern.search(text):
                return True
        return False

    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Uppercase for consistency
        text = text.upper()
        return text

    def _classify_label(
        self,
        text: str
    ) -> Tuple[LabelType, float, List[str]]:
        """
        Classify label type and get confidence.

        Returns:
            Tuple of (LabelType, confidence, matched_patterns)
        """
        matched_patterns = []

        # Check room numbers first (highest priority)
        for pattern, confidence in self._compiled_room_numbers:
            if pattern.match(text):
                matched_patterns.append(pattern.pattern)
                return LabelType.ROOM_NUMBER, confidence, matched_patterns

        # Check area tags
        for pattern, confidence in self._compiled_area_tags:
            if pattern.match(text):
                matched_patterns.append(pattern.pattern)
                return LabelType.AREA_TAG, confidence, matched_patterns

        # Check room names
        for pattern, confidence in self._compiled_room_names:
            if pattern.match(text):
                matched_patterns.append(pattern.pattern)
                return LabelType.ROOM_NAME, confidence, matched_patterns

        return LabelType.UNKNOWN, 0.0, matched_patterns

    def _get_centroid(
        self,
        bbox: Tuple[float, float, float, float]
    ) -> Tuple[float, float]:
        """Get centroid of bounding box."""
        x0, y0, x1, y1 = bbox
        return ((x0 + x1) / 2, (y0 + y1) / 2)

    def _is_in_drawing_area(
        self,
        centroid: Tuple[float, float],
        page_width: float,
        page_height: float
    ) -> bool:
        """Check if point is in drawing area (not title block)."""
        if self.drawing_bounds:
            x0, y0, x1, y1 = self.drawing_bounds
            return x0 <= centroid[0] <= x1 and y0 <= centroid[1] <= y1

        # Default: exclude right 20% (title block)
        return centroid[0] < page_width * 0.80

    def _calculate_confidence(
        self,
        ocr_confidence: float,
        pattern_confidence: float,
        source: str,
        is_in_drawing: bool
    ) -> float:
        """Calculate combined confidence score."""
        # Base: average of OCR and pattern confidence
        if source == "embedded":
            # Embedded text is more reliable
            base = (0.95 + pattern_confidence) / 2
        else:
            base = (ocr_confidence + pattern_confidence) / 2

        # Penalty if outside drawing area
        if not is_in_drawing:
            base *= 0.5

        return min(1.0, max(0.0, base))

    def _remove_duplicates(
        self,
        labels: List[ExtractedLabel]
    ) -> List[ExtractedLabel]:
        """Remove duplicate labels (same text near same location)."""
        if not labels:
            return labels

        unique = []
        seen: Set[str] = set()

        for label in labels:
            # Create key from text and approximate location
            loc_key = f"{int(label.centroid[0]/50)}_{int(label.centroid[1]/50)}"
            key = f"{label.text}_{loc_key}"

            if key not in seen:
                seen.add(key)
                unique.append(label)

        return unique

    def cluster_labels(
        self,
        labels: List[ExtractedLabel],
        distance_threshold: float = CLUSTER_DISTANCE_THRESHOLD
    ) -> List[LabelCluster]:
        """
        Cluster nearby labels that may belong to the same room.

        Some rooms have multiple labels (number + name, or wrapped text).

        Args:
            labels: List of extracted labels
            distance_threshold: Max distance (points) to cluster

        Returns:
            List of LabelCluster objects
        """
        if not labels:
            return []

        # Build spatial index
        points = [Point(l.centroid) for l in labels]
        tree = STRtree(points)

        # Track which labels are clustered
        clustered: Set[int] = set()
        clusters: List[LabelCluster] = []

        for i, label in enumerate(labels):
            if i in clustered:
                continue

            # Find nearby labels
            buffer = Point(label.centroid).buffer(distance_threshold)
            nearby_indices = tree.query(buffer)

            cluster_labels = [labels[j] for j in nearby_indices if j not in clustered]

            if len(cluster_labels) == 1:
                # Single label = single room
                clusters.append(LabelCluster(
                    labels=[label],
                    primary_label=label,
                    centroid=label.centroid,
                    combined_text=label.text
                ))
            else:
                # Multiple labels - find primary (room number preferred)
                primary = self._find_primary_label(cluster_labels)
                combined = self._combine_texts(cluster_labels)
                centroid = self._cluster_centroid(cluster_labels)

                clusters.append(LabelCluster(
                    labels=cluster_labels,
                    primary_label=primary,
                    centroid=centroid,
                    combined_text=combined
                ))

            # Mark as clustered
            for j in nearby_indices:
                clustered.add(j)

        # Assign cluster IDs
        for i, cluster in enumerate(clusters):
            for label in cluster.labels:
                label.cluster_id = i

        return clusters

    def _find_primary_label(
        self,
        labels: List[ExtractedLabel]
    ) -> ExtractedLabel:
        """Find the primary label in a cluster (prefer room numbers)."""
        # Priority: ROOM_NUMBER > AREA_TAG > ROOM_NAME > UNKNOWN
        priority = {
            LabelType.ROOM_NUMBER: 0,
            LabelType.AREA_TAG: 1,
            LabelType.ROOM_NAME: 2,
            LabelType.COMBINED: 3,
            LabelType.UNKNOWN: 4,
        }

        return min(labels, key=lambda l: (priority[l.label_type], -l.confidence))

    def _combine_texts(
        self,
        labels: List[ExtractedLabel]
    ) -> str:
        """Combine texts from clustered labels."""
        # Sort by position (left to right, top to bottom)
        sorted_labels = sorted(
            labels,
            key=lambda l: (l.centroid[1], l.centroid[0])
        )
        return " ".join(l.text for l in sorted_labels)

    def _cluster_centroid(
        self,
        labels: List[ExtractedLabel]
    ) -> Tuple[float, float]:
        """Calculate centroid of label cluster."""
        if not labels:
            return (0, 0)
        avg_x = sum(l.centroid[0] for l in labels) / len(labels)
        avg_y = sum(l.centroid[1] for l in labels) / len(labels)
        return (avg_x, avg_y)


def extract_room_labels(
    text_blocks: List[TextBlock],
    page_width: float,
    page_height: float,
    drawing_bounds: Optional[Tuple[float, float, float, float]] = None
) -> List[ExtractedLabel]:
    """
    Convenience function to extract room labels.

    Args:
        text_blocks: List of TextBlock from OCR/embedded text
        page_width: Page width in points
        page_height: Page height in points
        drawing_bounds: Optional drawing area bounds

    Returns:
        List of ExtractedLabel objects
    """
    extractor = RoomLabelExtractor(drawing_bounds=drawing_bounds)
    return extractor.extract(text_blocks, page_width, page_height)
