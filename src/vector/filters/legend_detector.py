"""
Legend Detector Module

Auto-detects and parses Construction Plan Legends from blueprint pages.

The legend defines the visual conventions for construction phases:
- NEW CONSTRUCTION = outlined walls (no fill)
- EXISTING CONSTRUCTION = gray-filled walls
- NOT IN CONTRACT (N.I.C.) = hatched areas
- DEMOLITION = dashed lines

Detection Strategy:
1. Search for "LEGEND" keyword in page text
2. Find nearby text blocks containing phase labels
3. Parse phase entries using regex patterns
4. Map visual patterns to construction phases
5. Fall back to industry defaults if no legend found
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

import pymupdf

from ..construction_phase import (
    ConstructionPhase,
    FillPattern,
    LegendEntry,
    LegendDetectionResult,
    get_default_legend,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Legend search keywords (in priority order - more specific first)
LEGEND_KEYWORDS = [
    "CONSTRUCTION PLAN LEGEND",
    "LEGEND - CONSTRUCTION",
    "WALL LEGEND",
    "SYMBOL LEGEND",
    "LEGEND",
    # Lower priority - only use if no better match
    # "KEY",  # Too generic, matches KEYNOTES
    # "WALL TYPES",  # Too generic
]

# Phase label patterns (regex)
PHASE_PATTERNS = {
    ConstructionPhase.NEW: [
        r"NEW\s*(CONSTRUCTION|WALLS?|WORK)?",
        r"PROPOSED\s*(CONSTRUCTION|WALLS?)?",
        r"NEW\s*INTERIOR\s*WALLS?",
    ],
    ConstructionPhase.EXISTING: [
        r"EXIST(ING)?\s*(CONSTRUCTION|WALLS?|TO\s*REMAIN)?",
        r"EXISTING\s*BUILDING",
        r"WALLS?\s*TO\s*REMAIN",
        r"REMAIN(ING)?\s*WALLS?",
    ],
    ConstructionPhase.NOT_IN_CONTRACT: [
        r"N\.?I\.?C\.?",
        r"NOT\s*IN\s*CONTRACT",
        r"AREA\s*NOT\s*IN\s*CONTRACT",
        r"OUTSIDE\s*SCOPE",
        r"BY\s*OTHERS",
    ],
    ConstructionPhase.DEMO: [
        r"DEMO(LITION)?",
        r"REMOVE",
        r"DEMOLISH",
        r"TO\s*BE\s*REMOVED",
        r"WALLS?\s*TO\s*REMOVE",
    ],
}

# Title block region (typically right side of page)
TITLE_BLOCK_X_RATIO = 0.70  # Legend usually in right 30% of page
LEGEND_SEARCH_MARGIN = 300  # Points to expand search around legend keyword


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TextBlock:
    """A text block extracted from PDF."""
    text: str
    bbox: Tuple[float, float, float, float]
    font_size: float = 0.0

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def x1(self) -> float:
        return self.bbox[2]

    @property
    def y1(self) -> float:
        return self.bbox[3]

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)


@dataclass
class LegendSearchResult:
    """Result from searching for legend on a page."""
    found: bool = False
    keyword_block: Optional[TextBlock] = None
    legend_region: Optional[Tuple[float, float, float, float]] = None
    nearby_text: List[TextBlock] = field(default_factory=list)
    detection_method: str = "none"


# =============================================================================
# LEGEND DETECTOR CLASS
# =============================================================================

class LegendDetector:
    """
    Detects and parses Construction Plan Legends from PDF pages.

    Usage:
        detector = LegendDetector()
        result = detector.detect(page)

        if result.has_legend:
            for entry in result.entries:
                print(f"{entry.label}: {entry.phase.value}")
    """

    def __init__(
        self,
        search_title_block: bool = True,
        title_block_x_ratio: float = TITLE_BLOCK_X_RATIO,
        search_margin: float = LEGEND_SEARCH_MARGIN,
        use_industry_defaults: bool = True,
    ):
        """
        Initialize the legend detector.

        Args:
            search_title_block: Whether to prioritize title block region
            title_block_x_ratio: X ratio for title block boundary
            search_margin: Margin around legend keyword for text search
            use_industry_defaults: Return defaults if no legend found
        """
        self.search_title_block = search_title_block
        self.title_block_x_ratio = title_block_x_ratio
        self.search_margin = search_margin
        self.use_industry_defaults = use_industry_defaults

        # Compile regex patterns
        self._compiled_patterns = {}
        for phase, patterns in PHASE_PATTERNS.items():
            self._compiled_patterns[phase] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def detect(self, page: pymupdf.Page) -> LegendDetectionResult:
        """
        Detect and parse the construction legend from a page.

        Args:
            page: PyMuPDF Page object

        Returns:
            LegendDetectionResult with parsed entries
        """
        page_width = page.rect.width
        page_height = page.rect.height

        # Extract text blocks from page
        text_blocks = self._extract_text_blocks(page)

        if not text_blocks:
            logger.debug("No text blocks found on page")
            if self.use_industry_defaults:
                return get_default_legend()
            return LegendDetectionResult()

        # Search for legend keyword
        search_result = self._find_legend_region(text_blocks, page_width, page_height)

        if not search_result.found:
            logger.debug("No legend keyword found on page")
            if self.use_industry_defaults:
                return get_default_legend()
            return LegendDetectionResult()

        # Parse legend entries from nearby text
        entries = self._parse_legend_entries(search_result.nearby_text)

        if not entries:
            logger.debug("No phase entries found near legend")
            if self.use_industry_defaults:
                return get_default_legend()
            return LegendDetectionResult()

        # Build result
        result = LegendDetectionResult(
            entries=entries,
            legend_bbox=search_result.legend_region,
            detection_method=search_result.detection_method,
            confidence=self._calculate_confidence(entries),
            page_num=page.number,
            has_legend=True,
        )

        logger.info(f"Legend detected: {len(entries)} entries, "
                   f"method={result.detection_method}, "
                   f"confidence={result.confidence:.2f}")

        return result

    def _extract_text_blocks(self, page: pymupdf.Page) -> List[TextBlock]:
        """Extract text blocks from page."""
        blocks = []

        try:
            # Get text as dictionary with position info
            text_dict = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)

            for block in text_dict.get("blocks", []):
                if block.get("type") != 0:  # Skip non-text blocks
                    continue

                bbox = block.get("bbox", (0, 0, 0, 0))

                # Combine all text from block lines/spans
                text_parts = []
                max_font_size = 0

                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            text_parts.append(text)
                            font_size = span.get("size", 0)
                            max_font_size = max(max_font_size, font_size)

                if text_parts:
                    combined_text = " ".join(text_parts)
                    blocks.append(TextBlock(
                        text=combined_text,
                        bbox=bbox,
                        font_size=max_font_size
                    ))

        except Exception as e:
            logger.warning(f"Error extracting text blocks: {e}")

        return blocks

    def _find_legend_region(
        self,
        text_blocks: List[TextBlock],
        page_width: float,
        page_height: float
    ) -> LegendSearchResult:
        """Find the legend region by searching for keywords."""
        result = LegendSearchResult()

        # Optionally prioritize title block region (right side of page)
        title_block_x = page_width * self.title_block_x_ratio if self.search_title_block else 0

        # Search for legend keywords (in priority order)
        # First pass: find all matches and pick the best one
        candidates = []

        for block in text_blocks:
            block_text = block.text.upper()

            for priority, keyword in enumerate(LEGEND_KEYWORDS):
                if keyword in block_text:
                    in_title_block = block.x0 >= title_block_x
                    candidates.append({
                        "block": block,
                        "keyword": keyword,
                        "priority": priority,
                        "in_title_block": in_title_block,
                    })
                    break  # Only match first keyword per block

        if not candidates:
            return result

        # Sort: prefer title block region, then by keyword priority
        candidates.sort(key=lambda c: (not c["in_title_block"], c["priority"]))

        # Use best candidate
        best = candidates[0]
        result.keyword_block = best["block"]
        result.found = True
        result.detection_method = "keyword_search"

        # Define legend region around keyword
        kb = result.keyword_block
        margin = self.search_margin

        # Expand more below and to the right (legend entries are typically below/right of title)
        result.legend_region = (
            max(0, kb.x0 - margin / 2),
            max(0, kb.y0 - margin / 4),
            min(page_width, kb.x1 + margin),
            min(page_height, kb.y1 + margin * 3)  # Much more space below for entries
        )

        # Find text blocks near the legend
        for block in text_blocks:
            if block == result.keyword_block:
                continue

            # Check if block is within legend region
            bx, by = block.center
            rx0, ry0, rx1, ry1 = result.legend_region

            if rx0 <= bx <= rx1 and ry0 <= by <= ry1:
                result.nearby_text.append(block)

        return result

    def _parse_legend_entries(
        self,
        text_blocks: List[TextBlock]
    ) -> List[LegendEntry]:
        """Parse legend entries from text blocks."""
        entries = []
        found_phases = set()

        for block in text_blocks:
            text = block.text.upper().strip()

            # Try to match each phase pattern
            for phase, patterns in self._compiled_patterns.items():
                if phase in found_phases:
                    continue  # Already found this phase

                for pattern in patterns:
                    if pattern.search(text):
                        # Determine fill pattern based on phase (industry defaults)
                        fill_pattern, fill_color = self._get_default_fill_for_phase(phase)

                        entry = LegendEntry(
                            label=block.text.strip(),
                            phase=phase,
                            fill_pattern=fill_pattern,
                            fill_color=fill_color,
                            confidence=0.85,
                            bbox=block.bbox,
                        )
                        entries.append(entry)
                        found_phases.add(phase)
                        break

        return entries

    def _get_default_fill_for_phase(
        self,
        phase: ConstructionPhase
    ) -> Tuple[FillPattern, Optional[Tuple[float, float, float]]]:
        """Get default fill pattern for a phase."""
        if phase == ConstructionPhase.NEW:
            return FillPattern.NONE, None
        elif phase == ConstructionPhase.EXISTING:
            return FillPattern.SOLID_GRAY, (0.5, 0.5, 0.5)
        elif phase == ConstructionPhase.NOT_IN_CONTRACT:
            return FillPattern.HATCHED, None
        elif phase == ConstructionPhase.DEMO:
            return FillPattern.NONE, None
        else:
            return FillPattern.UNKNOWN, None

    def _calculate_confidence(self, entries: List[LegendEntry]) -> float:
        """Calculate overall confidence based on entries found."""
        if not entries:
            return 0.0

        # Higher confidence with more entries
        base_confidence = 0.70
        entry_bonus = min(0.25, len(entries) * 0.05)

        # Bonus for finding key phases
        phase_set = {e.phase for e in entries}
        if ConstructionPhase.NEW in phase_set:
            entry_bonus += 0.05
        if ConstructionPhase.EXISTING in phase_set:
            entry_bonus += 0.05

        return min(0.95, base_confidence + entry_bonus)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def detect_legend(page: pymupdf.Page) -> LegendDetectionResult:
    """
    Convenience function to detect legend on a page.

    Args:
        page: PyMuPDF Page object

    Returns:
        LegendDetectionResult
    """
    detector = LegendDetector()
    return detector.detect(page)


def get_legend_or_defaults(page: pymupdf.Page) -> LegendDetectionResult:
    """
    Get legend from page, or return industry defaults.

    Args:
        page: PyMuPDF Page object

    Returns:
        LegendDetectionResult (always has entries)
    """
    detector = LegendDetector(use_industry_defaults=True)
    return detector.detect(page)
