"""
Blueprint Pipeline - Master Constants Reference

CRITICAL: These are the EXACT values from the specification.
Do not modify these values unless the specification changes.
"""

# =============================================================================
# VECTOR DETECTION CONSTANTS
# =============================================================================

# Minimum drawing objects to classify as vector PDF
MIN_DRAWINGS_FOR_VECTOR = 50

# Maximum image count to still consider vector
MAX_IMAGES_FOR_VECTOR = 3

# Maximum % of page covered by images for vector classification
MAX_IMAGE_COVERAGE_PERCENT = 30

# =============================================================================
# LINE FILTERING CONSTANTS
# =============================================================================

# Ignore segments shorter than this (in PDF points)
MIN_SEGMENT_LENGTH_POINTS = 5

# Ignore segments longer than 90% of page diagonal (border lines)
MAX_SEGMENT_LENGTH_RATIO = 0.9

# Lines thinner than this are annotations, not walls
ANNOTATION_LINE_WIDTH_MAX = 0.25

# Lines thicker than this are likely walls
WALL_LINE_WIDTH_MIN = 0.5

# Max angle difference to consider lines parallel (degrees)
PARALLEL_ANGLE_TOLERANCE_DEG = 3

# Max distance between parallel lines to merge as double-wall
DOUBLE_LINE_DISTANCE_MAX = 10

# =============================================================================
# GAP BRIDGING CONSTANTS
# =============================================================================

# Default door gap tolerance (36 PDF points = 0.5 inch)
DEFAULT_GAP_TOLERANCE_INCHES = 0.5

# Minimum allowed gap tolerance
MIN_GAP_TOLERANCE_INCHES = 0.25

# Maximum gap (larger = not a door, it's an opening)
MAX_GAP_TOLERANCE_INCHES = 4.0

# Points buffer when checking if bridge crosses existing segment
BRIDGE_CROSS_SEGMENT_BUFFER = 2

# =============================================================================
# POLYGON FILTERING CONSTANTS
# =============================================================================

# Minimum polygon area in PDF points (~2 sq inches)
MIN_AREA_SQ_POINTS = 10000

# Minimum room area after scale conversion (commercial)
MIN_AREA_REAL_SQFT = 25

# Maximum room area (catches "entire floor" polygons)
MAX_AREA_REAL_SQFT = 100000

# Minimum polygon vertices (triangle = not a room)
MIN_VERTICES = 4

# Reject polygons covering >85% of page
MAX_AREA_PAGE_RATIO = 0.85

# =============================================================================
# TEXT AND LABEL CONSTANTS
# =============================================================================

# Minimum font size for room labels
ROOM_LABEL_MIN_FONTSIZE = 5

# Maximum font size (larger = title, not label)
ROOM_LABEL_MAX_FONTSIZE = 36

# Expand polygon bounds by 15% when searching for labels
LABEL_SEARCH_EXPANSION_RATIO = 0.15

# Minimum OCR confidence to use result
OCR_CONFIDENCE_THRESHOLD = 0.6

# Minimum characters for valid room label
MIN_LABEL_LENGTH = 2

# Maximum characters (longer = not a room label)
MAX_LABEL_LENGTH = 50

# =============================================================================
# RASTER PROCESSING CONSTANTS
# =============================================================================

# Default DPI for raster conversion
DEFAULT_RENDER_DPI = 300

# DPI when low-memory mode active
LOW_MEMORY_DPI = 150

# kernel_size = max(3, dpi / 100)
KERNEL_SIZE_DIVISOR = 100

# Maximum auto-skew correction (degrees)
MAX_SKEW_CORRECTION_DEG = 5

# Douglas-Peucker epsilon at 300 DPI
DP_EPSILON_BASE = 2.0

# Color tolerance for flood fill (0-255)
FLOOD_FILL_TOLERANCE = 20

# Min contour area as ratio of image area
MIN_CONTOUR_AREA_RATIO = 0.001

# =============================================================================
# SCALE DETECTION CONSTANTS
# =============================================================================

# Warn if scale sources differ by >15%
SCALE_CONFLICT_THRESHOLD_PERCENT = 15

# Points radius to search for dimension tick marks
DIMENSION_TICK_SEARCH_RADIUS = 5

# Points radius to search for associated line
DIMENSION_LINE_SEARCH_RADIUS = 50

# Default scale if detection fails
DEFAULT_COMMERCIAL_SCALE = "1/8 inch = 1 foot"

# PDF points per foot at default scale
DEFAULT_COMMERCIAL_SCALE_FACTOR = 96

# =============================================================================
# CEILING HEIGHT CONSTANTS (COMMERCIAL)
# =============================================================================

# Default if not found (commercial standard)
DEFAULT_CEILING_HEIGHT_FT = 10.0

# Minimum valid ceiling height
MIN_CEILING_HEIGHT_FT = 7

# Maximum valid (warehouses, gyms)
MAX_CEILING_HEIGHT_FT = 60

# Search within 10% of page size for height labels
HEIGHT_LABEL_SEARCH_RADIUS_RATIO = 0.1

# =============================================================================
# MEMORY MANAGEMENT CONSTANTS
# =============================================================================

# Enable low-memory mode above this (MB)
LARGE_FILE_MB_THRESHOLD = 500

# Enable low-memory mode above this page count
LARGE_PAGE_COUNT_THRESHOLD = 100

# Split pages larger than this (~138 inches)
MAX_PAGE_DIMENSION_POINTS = 10000

# Save partial results every N pages
PARTIAL_SAVE_INTERVAL = 10

# =============================================================================
# DERIVED CONSTANTS
# =============================================================================

# PDF points per inch
POINTS_PER_INCH = 72

# Default gap tolerance in PDF points
DEFAULT_GAP_TOLERANCE_POINTS = DEFAULT_GAP_TOLERANCE_INCHES * POINTS_PER_INCH

# Min gap tolerance in PDF points
MIN_GAP_TOLERANCE_POINTS = MIN_GAP_TOLERANCE_INCHES * POINTS_PER_INCH

# Max gap tolerance in PDF points
MAX_GAP_TOLERANCE_POINTS = MAX_GAP_TOLERANCE_INCHES * POINTS_PER_INCH

# =============================================================================
# COMMON SCALE FACTORS
# PDF points per real foot for common scales
# =============================================================================

SCALE_FACTORS = {
    "1/16 inch = 1 foot": 4.5,
    "1/8 inch = 1 foot": 9,
    "3/16 inch = 1 foot": 13.5,
    "1/4 inch = 1 foot": 18,
    "3/8 inch = 1 foot": 27,
    "1/2 inch = 1 foot": 36,
    "3/4 inch = 1 foot": 54,
    "1 inch = 1 foot": 72,
}

# Metric scale factors (PDF points per real meter)
METRIC_SCALE_FACTORS = {
    "1:200": 0.36,
    "1:100": 0.72,
    "1:50": 1.44,
}

# =============================================================================
# PAGE TYPE KEYWORDS
# =============================================================================

PAGE_TYPE_KEYWORDS = {
    "FLOOR_PLAN": [
        "floor plan", "plan view", "level", "first floor", "ground floor",
        "1st floor", "2nd floor", "3rd floor", "4th floor", "5th floor",
        "basement", "mezzanine"
    ],
    "RCP": [
        "reflected ceiling", "rcp", "ceiling plan", "reflected"
    ],
    "ELEVATION": [
        "elevation", "interior elevation", "elev", "north elevation",
        "south elevation", "east elevation", "west elevation"
    ],
    "SECTION": [
        "section", "building section", "wall section"
    ],
    "SCHEDULE": [
        "schedule", "room schedule", "door schedule", "finish schedule",
        "window schedule"
    ],
}

# =============================================================================
# ROOM LABEL PATTERNS
# =============================================================================

ROOM_TYPE_KEYWORDS = [
    # Office/Admin
    "OFFICE", "OFF", "CONF", "CONFERENCE", "MEETING", "BOARDROOM",
    "RECEPTION", "RECEP", "LOBBY", "WAITING", "WORK ROOM", "COPY", "MAIL",
    # Support
    "STORAGE", "STOR", "CLOSET", "JANITOR", "JAN", "ELECTRICAL", "ELEC",
    "MECHANICAL", "MECH", "TELECOM", "IDF", "MDF", "SERVER",
    # Restroom
    "RESTROOM", "RR", "TOILET", "MEN", "WOMEN", "UNISEX", "LOCKER", "SHOWER",
    # Circulation
    "CORRIDOR", "CORR", "HALL", "HALLWAY", "VESTIBULE", "ENTRY",
    "STAIR", "STAIRWELL", "ELEVATOR", "ELEV",
    # Food Service
    "KITCHEN", "BREAK", "BREAK ROOM", "KITCHENETTE", "CAFE", "CAFETERIA",
    "DINING", "VENDING",
    # Industrial
    "WAREHOUSE", "LOADING", "DOCK", "SHIPPING", "RECEIVING",
    "MANUFACTURING", "ASSEMBLY",
    # Specialty
    "LAB", "LABORATORY", "EXAM", "CLEAN ROOM", "DATA CENTER",
    # Residential (less common in commercial)
    "BEDROOM", "LIVING", "DINING", "BATHROOM", "BATH",
]

# Patterns to exclude from room labels
EXCLUDE_PATTERNS = [
    # Dimension text markers
    "'", '"', "feet", "inch", "mm", "cm", "ft", "in",
    # Scale text
    "=", ":",
    # Note markers
    "NOTE", "SEE", "REF", "TYP",
]

# Drawing number prefixes to exclude
DRAWING_NUMBER_PREFIXES = ["A", "S", "M", "E", "P"]

# =============================================================================
# CONFIDENCE LEVELS
# =============================================================================

class Confidence:
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"

# =============================================================================
# PROCESSING PATHS
# =============================================================================

class ProcessingPath:
    VECTOR = "VECTOR"
    RASTER = "RASTER"
    HYBRID = "HYBRID"
    SKIP = "SKIP"

# =============================================================================
# PAGE TYPES
# =============================================================================

class PageType:
    FLOOR_PLAN = "FLOOR_PLAN"
    RCP = "RCP"
    ELEVATION = "ELEVATION"
    SECTION = "SECTION"
    SCHEDULE = "SCHEDULE"
    OTHER = "OTHER"
