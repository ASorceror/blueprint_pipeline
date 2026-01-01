# Architectural Grid Lines Research Document

## Overview

Grid lines (also called column lines or structural grid) are a fundamental architectural drawing convention that creates a coordinate reference system for buildings.

## Visual Characteristics

### Grid Markers (Bubbles)
- **Left/Right edges**: Letters in circles (A, B, C, D...) - typically horizontal grid lines
- **Top/Bottom edges**: Numbers in circles (1, 2, 3, 4...) - typically vertical grid lines
- **Symbol**: Circle or hexagon containing single letter or 1-2 digit number
- **Size**: Typically 0.5" to 0.75" diameter

### Grid Lines
- **Line Pattern**: Long-dash-short-dash (center line pattern)
  - Long segment: 0.5" - 1.0"
  - Gap: 0.125" - 0.25"
  - Short segment: 0.125" - 0.25"
- **Line Weight**: 0.5pt to 1.5pt (thin to medium)
- **Span**: Full page width (horizontal) or height (vertical)
- **Spacing**: Regular intervals (typically 20'-0", 25'-0", or 30'-0" in real units)

## Why Grid Lines Are Critical for Room Detection

### Problems They Cause (If Not Filtered)
1. **False wall detection**: Grid lines pass width thresholds and are treated as walls
2. **Incorrect room boundaries**: Ray-casting hits grid lines instead of actual walls
3. **Spurious rooms**: Grid intersections create phantom room boundaries
4. **Area calculation errors**: Room areas include space beyond actual walls

### Opportunities They Provide (If Properly Detected)
1. **Building extent definition**: Outer grid lines mark structural footprint
2. **Reference coordinate system**: Room locations can be described as "between A-B and 2-3"
3. **Scale validation**: Known grid spacings (20', 25', 30') can verify scale factor
4. **Cross-sheet correlation**: Same grid references appear on all sheets

## Current Implementation Status

### What's Working (Text/Label Level)
- `room_label_extractor.py` filters grid markers (A, B, 1, 2) at page edges
- Edge threshold: 300 points from page boundary
- Pattern matching: Single letters, single/double digits

### What's Missing (Geometry/Segment Level)
1. **No dash pattern extraction**: PyMuPDF provides `dashes` but we don't capture it
2. **No line style in Segment class**: Only stores start, end, width, color
3. **No dedicated grid line filter**: Grid segments treated as potential walls
4. **No full-span detection**: Lines spanning entire page not specifically identified

## Detection Characteristics

| Characteristic | Grid Lines | Walls | Dimensions | Hatching |
|----------------|------------|-------|------------|----------|
| **Length** | 70-100% of page | 2-50% of page | 5-20% of page | 5-30% |
| **Width** | 0.5-1.5pt | 2-6pt | 0.3-1pt | 0.3-1pt |
| **Pattern** | Dashed/Center | Solid | Solid | Parallel |
| **Spacing** | Regular (20-30') | Irregular | N/A | Very regular |
| **Full-Page Span** | Yes | No | No | No |
| **Edge Labels** | Yes (bubbles) | No | Yes (text) | No |

## PyMuPDF Capabilities

### Available in `page.get_drawings()`
```python
drawing = {
    'width': float,           # Line width in points
    'color': tuple,           # (R, G, B) color
    'dashes': list,           # Dash pattern [dash, gap, ...] or []
    'items': list,            # Path items (lines, curves, etc.)
    'lineCap': int,           # Line cap style
    'lineJoin': int,          # Line join style
}
```

### Dash Pattern Interpretation
- `dashes = []` or `None` → Solid line
- `dashes = [3, 2]` → 3pt dash, 2pt gap (simple dashed)
- `dashes = [6, 2, 2, 2]` → Long-short-dash pattern (grid/center line)

## Recommended Detection Algorithm

### Phase 1: Extract Grid Line Candidates
```
For each segment:
  1. Check if dashes array exists (dashed line)
  2. Check if width < 1.5pt (thin line)
  3. Check if length > 70% of page dimension
  4. Check if angle is 0° or 90° (horizontal/vertical)

  If all true → Grid line candidate
```

### Phase 2: Validate Regular Spacing
```
For horizontal candidates:
  1. Group by Y-coordinate (within tolerance)
  2. Sort by position
  3. Calculate spacing between adjacent lines
  4. Check spacing regularity (CV < 10%)

For vertical candidates:
  1. Group by X-coordinate (within tolerance)
  2. Sort by position
  3. Calculate spacing between adjacent lines
  4. Check spacing regularity (CV < 10%)

Regular spacing → Confirmed grid lines
```

### Phase 3: Detect Grid Bubbles
```
For each text block near page edge:
  1. Check if text is single letter (A-Z) or 1-2 digits
  2. Check if position is at page margin
  3. Look for circular path nearby (bubble symbol)
  4. Associate bubble with nearest grid line

Bubbles found → Extract grid labels
```

### Phase 4: Build Grid Coordinate System
```
1. Assign letters to vertical grid lines (left to right)
2. Assign numbers to horizontal grid lines (top to bottom)
3. Create coordinate mapping: (letter, number) → (x, y)
4. For each room, calculate grid cell reference
```

## Implementation Files

| File | Current Role | Needed Changes |
|------|--------------|----------------|
| `extractor.py` | Segment extraction | Add dashes to Segment class |
| `segment_filters.py` | Filter hatching/dimensions | Add GridLineFilter |
| `filter_pipeline.py` | Orchestrate filters | Include grid filter |
| `room_label_extractor.py` | Filter grid labels | Already working |
| `constants.py` | Thresholds | Add grid line constants |
| `hybrid_detector.py` | Room detection | Use grid-filtered segments |

## Key Constants to Add

```python
# Grid line detection
GRID_LINE_WIDTH_MAX = 1.5          # Points - grid lines are thin
GRID_LINE_LENGTH_MIN_RATIO = 0.7   # Must span 70% of page
GRID_SPACING_TOLERANCE = 0.1       # 10% variation allowed
GRID_EDGE_MARGIN = 100             # Points from page edge for bubbles

# Dash pattern detection
DASHED_LINE_MIN_SEGMENTS = 2       # At least 2 dash segments
CENTER_LINE_PATTERN = [6, 2, 2, 2] # Long-short pattern
```

## References

- AIA CAD Layer Standard: A-GRID layer
- CSI Layer Convention: GRIDS layer
- PyMuPDF documentation: page.get_drawings()
- Current implementation: src/text/room_label_extractor.py
