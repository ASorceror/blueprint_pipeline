"""
Construction Phase Module

Defines enums, dataclasses, and utilities for construction phase detection
in architectural blueprints.

Construction phases identify the status of walls/elements:
- NEW: To be constructed (in scope for new work)
- EXISTING: Already built, to remain (context/reference)
- NOT_IN_CONTRACT: Outside project scope (N.I.C.)
- DEMO: To be demolished/removed

The phase classification enables:
- Scope filtering (measure only NEW construction)
- Context awareness (understand existing conditions)
- Scope exclusion (ignore N.I.C. areas)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Dict

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class ConstructionPhase(Enum):
    """
    Construction phase classification for wall segments.

    Phases indicate the construction status of elements on architectural
    drawings, typically defined in a construction plan legend.

    Values:
        NEW: New construction to be built (primary scope)
        EXISTING: Existing construction to remain (reference/context)
        NOT_IN_CONTRACT: Not in contract scope (N.I.C.) - exclude from measurements
        DEMO: Demolition/removal scheduled
        UNKNOWN: Phase could not be determined
    """
    NEW = "NEW"
    EXISTING = "EXISTING"
    NOT_IN_CONTRACT = "NOT_IN_CONTRACT"
    DEMO = "DEMO"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_string(cls, value: str) -> "ConstructionPhase":
        """
        Parse a construction phase from string, with fuzzy matching.

        Args:
            value: String to parse (case-insensitive)

        Returns:
            ConstructionPhase enum value

        Examples:
            >>> ConstructionPhase.from_string("new")
            ConstructionPhase.NEW
            >>> ConstructionPhase.from_string("EXISTING TO REMAIN")
            ConstructionPhase.EXISTING
            >>> ConstructionPhase.from_string("N.I.C.")
            ConstructionPhase.NOT_IN_CONTRACT
        """
        if value is None:
            return cls.UNKNOWN

        normalized = value.upper().strip()

        # Direct match
        try:
            return cls(normalized)
        except ValueError:
            pass

        # Fuzzy matching for common variations
        if "NEW" in normalized and ("CONSTRUCTION" in normalized or "WORK" in normalized or "WALL" in normalized):
            return cls.NEW
        if normalized == "NEW":
            return cls.NEW
        if "EXIST" in normalized:
            return cls.EXISTING
        if "N.I.C" in normalized or "NIC" in normalized or "NOT IN CONTRACT" in normalized:
            return cls.NOT_IN_CONTRACT
        if "DEMO" in normalized or "REMOVE" in normalized or "DEMOLISH" in normalized:
            return cls.DEMO

        return cls.UNKNOWN

    def is_in_scope(self) -> bool:
        """
        Check if this phase is within the project scope for measurements.

        NEW and EXISTING are typically in scope, N.I.C. is explicitly out of scope.

        Returns:
            True if this phase should be included in measurements
        """
        return self in (ConstructionPhase.NEW, ConstructionPhase.EXISTING)

    def is_new_work(self) -> bool:
        """Check if this phase represents new construction work."""
        return self == ConstructionPhase.NEW


class FillPattern(Enum):
    """
    Visual fill patterns used in construction legends.

    These patterns correspond to how walls/elements are visually
    represented in architectural drawings.
    """
    NONE = "none"           # Outline only (typically NEW)
    SOLID_GRAY = "gray"     # Solid gray fill (typically EXISTING)
    SOLID_BLACK = "black"   # Solid black fill (structural)
    HATCHED = "hatched"     # Diagonal lines (typically N.I.C. or material)
    CROSS_HATCHED = "cross_hatched"  # Cross-diagonal (demolition or special)
    WHITE = "white"         # White fill (background)
    UNKNOWN = "unknown"


class ClassificationMethod(Enum):
    """
    Method used to classify a segment's construction phase.

    Tracking the method enables confidence calibration and debugging.
    """
    FILL_PATTERN = "fill_pattern"       # Based on PyMuPDF fill color
    LEGEND_MATCH = "legend_match"       # Matched to detected legend entry
    SPATIAL_REGION = "spatial_region"   # Inside a classified region
    VLM_INFERENCE = "vlm_inference"     # Vision Language Model
    INDUSTRY_DEFAULT = "industry_default"  # Standard conventions
    MANUAL = "manual"                   # User-specified
    UNKNOWN = "unknown"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class LegendEntry:
    """
    A single entry from a construction plan legend.

    Represents one symbol/pattern definition from the drawing legend,
    mapping a visual representation to a construction phase.

    Attributes:
        label: Text label from legend (e.g., "NEW CONSTRUCTION")
        phase: Classified construction phase
        fill_pattern: Visual fill pattern type
        fill_color: RGB fill color if solid (0-1 range), None if no fill
        line_style: Line style (solid, dashed, etc.)
        confidence: Confidence in the classification (0-1)
        bbox: Bounding box of legend entry (x0, y0, x1, y1) in PDF points
    """
    label: str
    phase: ConstructionPhase
    fill_pattern: FillPattern
    fill_color: Optional[Tuple[float, float, float]] = None
    line_style: str = "solid"
    confidence: float = 1.0
    bbox: Optional[Tuple[float, float, float, float]] = None

    def matches_fill(self, fill: Optional[Tuple[float, ...]], tolerance: float = 0.1) -> bool:
        """
        Check if a fill color matches this legend entry.

        Args:
            fill: RGB fill color to check (0-1 range), None for no fill
            tolerance: Color matching tolerance

        Returns:
            True if the fill matches this entry's pattern
        """
        # No fill case
        if fill is None:
            return self.fill_pattern == FillPattern.NONE

        # Has fill - check if this entry has a fill color to match
        if self.fill_color is None:
            return False

        # Compare RGB values
        if len(fill) < 3:
            return False

        r_match = abs(fill[0] - self.fill_color[0]) < tolerance
        g_match = abs(fill[1] - self.fill_color[1]) < tolerance
        b_match = abs(fill[2] - self.fill_color[2]) < tolerance

        return r_match and g_match and b_match


@dataclass
class LegendDetectionResult:
    """
    Result from legend detection on a page.

    Contains all detected legend entries and metadata about the detection.

    Attributes:
        entries: List of detected legend entries
        legend_bbox: Bounding box of the legend area
        detection_method: How the legend was detected
        confidence: Overall confidence in detection
        page_num: Page number where legend was found
        has_legend: Whether a legend was actually found
    """
    entries: List[LegendEntry] = field(default_factory=list)
    legend_bbox: Optional[Tuple[float, float, float, float]] = None
    detection_method: str = "none"
    confidence: float = 0.0
    page_num: int = 0
    has_legend: bool = False

    def get_entry_for_phase(self, phase: ConstructionPhase) -> Optional[LegendEntry]:
        """Get the legend entry for a specific phase, if it exists."""
        for entry in self.entries:
            if entry.phase == phase:
                return entry
        return None

    def get_phases(self) -> List[ConstructionPhase]:
        """Get list of phases present in the legend."""
        return [entry.phase for entry in self.entries]


@dataclass
class PhaseClassificationStats:
    """
    Statistics from construction phase classification.

    Tracks the distribution of phases and classification methods
    for a set of wall segments.

    Attributes:
        total_segments: Total number of segments classified
        new_count: Number classified as NEW
        existing_count: Number classified as EXISTING
        nic_count: Number classified as NOT_IN_CONTRACT
        demo_count: Number classified as DEMO
        unknown_count: Number with UNKNOWN phase
        avg_confidence: Average confidence across all classifications
        min_confidence: Minimum confidence observed
        methods_used: Count by classification method
    """
    total_segments: int = 0
    new_count: int = 0
    existing_count: int = 0
    nic_count: int = 0
    demo_count: int = 0
    unknown_count: int = 0
    avg_confidence: float = 0.0
    min_confidence: float = 1.0
    methods_used: Dict[str, int] = field(default_factory=dict)

    def add_classification(
        self,
        phase: ConstructionPhase,
        confidence: float,
        method: ClassificationMethod
    ) -> None:
        """
        Record a single classification result.

        Args:
            phase: The classified phase
            confidence: Classification confidence (0-1)
            method: Method used for classification
        """
        self.total_segments += 1

        # Update phase counts
        if phase == ConstructionPhase.NEW:
            self.new_count += 1
        elif phase == ConstructionPhase.EXISTING:
            self.existing_count += 1
        elif phase == ConstructionPhase.NOT_IN_CONTRACT:
            self.nic_count += 1
        elif phase == ConstructionPhase.DEMO:
            self.demo_count += 1
        else:
            self.unknown_count += 1

        # Update confidence stats
        self.min_confidence = min(self.min_confidence, confidence)
        # Rolling average
        prev_total = self.total_segments - 1
        if prev_total > 0:
            self.avg_confidence = (self.avg_confidence * prev_total + confidence) / self.total_segments
        else:
            self.avg_confidence = confidence

        # Update method counts
        method_key = method.value
        self.methods_used[method_key] = self.methods_used.get(method_key, 0) + 1

    def get_phase_distribution(self) -> Dict[str, float]:
        """
        Get phase distribution as percentages.

        Returns:
            Dict mapping phase name to percentage (0-100)
        """
        if self.total_segments == 0:
            return {}

        return {
            "NEW": 100 * self.new_count / self.total_segments,
            "EXISTING": 100 * self.existing_count / self.total_segments,
            "NOT_IN_CONTRACT": 100 * self.nic_count / self.total_segments,
            "DEMO": 100 * self.demo_count / self.total_segments,
            "UNKNOWN": 100 * self.unknown_count / self.total_segments,
        }

    def summary(self) -> str:
        """Generate a human-readable summary string."""
        if self.total_segments == 0:
            return "No segments classified"

        dist = self.get_phase_distribution()
        lines = [
            f"Total segments: {self.total_segments}",
            f"  NEW:             {self.new_count:5d} ({dist['NEW']:5.1f}%)",
            f"  EXISTING:        {self.existing_count:5d} ({dist['EXISTING']:5.1f}%)",
            f"  NOT_IN_CONTRACT: {self.nic_count:5d} ({dist['NOT_IN_CONTRACT']:5.1f}%)",
            f"  DEMO:            {self.demo_count:5d} ({dist['DEMO']:5.1f}%)",
            f"  UNKNOWN:         {self.unknown_count:5d} ({dist['UNKNOWN']:5.1f}%)",
            f"Average confidence: {self.avg_confidence:.1%}",
            f"Min confidence: {self.min_confidence:.1%}",
        ]

        if self.methods_used:
            lines.append("Classification methods:")
            for method, count in sorted(self.methods_used.items(), key=lambda x: -x[1]):
                lines.append(f"  {method}: {count}")

        return "\n".join(lines)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_gray_fill(fill: Optional[Tuple[float, ...]],
                 gray_min: float = 0.25,
                 gray_max: float = 0.75,
                 tolerance: float = 0.05) -> bool:
    """
    Check if a fill color is medium gray (indicating EXISTING construction).

    Gray fills in architectural drawings typically represent existing
    construction per industry convention.

    Args:
        fill: RGB fill color tuple (0-1 range), or None
        gray_min: Minimum gray value (darker bound)
        gray_max: Maximum gray value (lighter bound)
        tolerance: Max difference between R, G, B for grayscale

    Returns:
        True if fill is a gray within the specified range

    Examples:
        >>> is_gray_fill((0.5, 0.5, 0.5))
        True
        >>> is_gray_fill((0.498, 0.498, 0.498))  # Common PDF gray
        True
        >>> is_gray_fill(None)
        False
        >>> is_gray_fill((1.0, 0.0, 0.0))  # Red
        False
    """
    if fill is None:
        return False

    if len(fill) < 3:
        # Grayscale value directly
        gray = fill[0]
        return gray_min <= gray <= gray_max

    r, g, b = fill[0], fill[1], fill[2]

    # Check if it's actually gray (R ≈ G ≈ B)
    max_diff = max(abs(r - g), abs(g - b), abs(r - b))
    if max_diff > tolerance:
        return False

    # Check gray value is in range
    gray_value = (r + g + b) / 3
    return gray_min <= gray_value <= gray_max


def classify_fill_color(fill: Optional[Tuple[float, ...]],
                        fill_type: Optional[str] = None) -> Tuple[FillPattern, ConstructionPhase, float]:
    """
    Classify a fill color into pattern and construction phase.

    Uses industry-standard conventions for fill pattern interpretation.

    Args:
        fill: RGB fill color tuple (0-1 range), or None
        fill_type: PyMuPDF draw type ('s'=stroke, 'f'=fill, 'fs'=both)

    Returns:
        Tuple of (FillPattern, ConstructionPhase, confidence)

    Industry Defaults:
        - No fill (stroke only) → NEW construction
        - Gray fill (0.25-0.75) → EXISTING construction
        - Black fill (<0.1) → Structural/special
        - White fill (>0.95) → Background
    """
    # No fill case (stroke only = NEW construction)
    if fill is None or fill_type == 's':
        return FillPattern.NONE, ConstructionPhase.NEW, 0.80

    # Analyze fill color
    if len(fill) < 3:
        gray = fill[0]
    else:
        r, g, b = fill[0], fill[1], fill[2]
        gray = (r + g + b) / 3

        # Check if it's colored (not grayscale)
        max_diff = max(abs(r - g), abs(g - b), abs(r - b))
        if max_diff > 0.1:
            # Colored fill - could be MEP or special, classify as unknown
            return FillPattern.UNKNOWN, ConstructionPhase.UNKNOWN, 0.50

    # Classify by gray level
    if gray > 0.95:
        return FillPattern.WHITE, ConstructionPhase.UNKNOWN, 0.60
    elif gray < 0.10:
        return FillPattern.SOLID_BLACK, ConstructionPhase.EXISTING, 0.70
    elif 0.25 <= gray <= 0.75:
        return FillPattern.SOLID_GRAY, ConstructionPhase.EXISTING, 0.85
    else:
        return FillPattern.UNKNOWN, ConstructionPhase.UNKNOWN, 0.50


# =============================================================================
# INDUSTRY DEFAULT MAPPINGS
# =============================================================================

# Default legend entries when no legend is detected
INDUSTRY_DEFAULT_LEGEND = [
    LegendEntry(
        label="NEW CONSTRUCTION",
        phase=ConstructionPhase.NEW,
        fill_pattern=FillPattern.NONE,
        fill_color=None,
        line_style="solid",
        confidence=0.80,
    ),
    LegendEntry(
        label="EXISTING CONSTRUCTION",
        phase=ConstructionPhase.EXISTING,
        fill_pattern=FillPattern.SOLID_GRAY,
        fill_color=(0.5, 0.5, 0.5),
        line_style="solid",
        confidence=0.85,
    ),
    LegendEntry(
        label="NOT IN CONTRACT",
        phase=ConstructionPhase.NOT_IN_CONTRACT,
        fill_pattern=FillPattern.HATCHED,
        fill_color=None,
        line_style="solid",
        confidence=0.75,
    ),
    LegendEntry(
        label="DEMOLITION",
        phase=ConstructionPhase.DEMO,
        fill_pattern=FillPattern.NONE,
        fill_color=None,
        line_style="dashed",
        confidence=0.70,
    ),
]


def get_default_legend() -> LegendDetectionResult:
    """
    Get industry-standard default legend when no legend is detected.

    Returns:
        LegendDetectionResult with standard entries
    """
    return LegendDetectionResult(
        entries=INDUSTRY_DEFAULT_LEGEND.copy(),
        legend_bbox=None,
        detection_method="industry_default",
        confidence=0.75,
        page_num=0,
        has_legend=False,
    )


# =============================================================================
# PHASE COLORS FOR VISUALIZATION
# =============================================================================

PHASE_COLORS: Dict[ConstructionPhase, Tuple[float, float, float]] = {
    ConstructionPhase.NEW: (0.0, 0.0, 1.0),           # Blue
    ConstructionPhase.EXISTING: (0.5, 0.5, 0.5),      # Gray
    ConstructionPhase.NOT_IN_CONTRACT: (1.0, 0.5, 0.0),  # Orange
    ConstructionPhase.DEMO: (1.0, 0.0, 0.0),          # Red
    ConstructionPhase.UNKNOWN: (0.8, 0.8, 0.8),       # Light gray
}


def get_phase_color(phase: ConstructionPhase) -> Tuple[float, float, float]:
    """Get the visualization color for a construction phase."""
    return PHASE_COLORS.get(phase, PHASE_COLORS[ConstructionPhase.UNKNOWN])
