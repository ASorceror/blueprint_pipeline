#!/usr/bin/env python3
"""
Debug script for wall detection with hybrid pipeline.

Pipeline stages:
1. Morphological grid removal (OpenCV)
2. Vector extraction with improved filtering
3. VLM validation (Claude Vision) for correction
4. Final wall output with debug visualizations
"""

import sys
sys.path.insert(0, 'C:/Measure/blueprint_pipeline')

import os
import re
import math
import json
import base64
import pymupdf
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from PIL import Image, ImageDraw, ImageFont
from src.vector.extractor import extract_wall_segments, extract_wall_segments_simple
from src.pdf.reader import render_page_to_image
from src.vector.filters.title_block_detector import TitleBlockDetector

# Try to import anthropic for VLM validation
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: anthropic package not installed. VLM validation disabled.")


# ============================================================
# STAGE 1: MORPHOLOGICAL GRID REMOVAL (OpenCV)
# ============================================================

def remove_grid_lines_morphological(image_np: np.ndarray,
                                     min_line_length_ratio: float = 0.3,
                                     output_debug: bool = False,
                                     output_dir: Path = None,
                                     base_name: str = "") -> Tuple[np.ndarray, Dict]:
    """
    Remove grid lines from floor plan image using morphological operations.

    Grid lines are typically:
    - Long horizontal/vertical lines spanning significant portion of page
    - Thin lines (often dashed visually but rendered as segments)
    - Located at regular intervals

    Args:
        image_np: Input image as numpy array (BGR format)
        min_line_length_ratio: Minimum line length as ratio of image dimension
        output_debug: Whether to save debug images
        output_dir: Directory for debug output
        base_name: Base name for output files

    Returns:
        Tuple of (cleaned_image, stats_dict)
    """
    stats = {
        'horizontal_lines_removed': 0,
        'vertical_lines_removed': 0,
        'total_lines_removed': 0,
    }

    # Convert to grayscale
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_np.copy()

    height, width = gray.shape

    # Invert image (lines become white on black)
    inverted = cv2.bitwise_not(gray)

    # Apply binary threshold
    _, binary = cv2.threshold(inverted, 200, 255, cv2.THRESH_BINARY)

    # Calculate minimum line lengths
    min_horizontal_length = int(width * min_line_length_ratio)
    min_vertical_length = int(height * min_line_length_ratio)

    # Create horizontal kernel to detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_horizontal_length, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

    # Create vertical kernel to detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_vertical_length))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

    # Dilate detected lines slightly to ensure complete removal
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    horizontal_lines = cv2.dilate(horizontal_lines, dilate_kernel, iterations=1)
    vertical_lines = cv2.dilate(vertical_lines, dilate_kernel, iterations=1)

    # Combine horizontal and vertical lines
    all_grid_lines = cv2.bitwise_or(horizontal_lines, vertical_lines)

    # Count removed lines using connected components
    num_h_labels, _ = cv2.connectedComponents(horizontal_lines)
    num_v_labels, _ = cv2.connectedComponents(vertical_lines)
    stats['horizontal_lines_removed'] = num_h_labels - 1  # Subtract background
    stats['vertical_lines_removed'] = num_v_labels - 1
    stats['total_lines_removed'] = stats['horizontal_lines_removed'] + stats['vertical_lines_removed']

    # Remove grid lines from original image
    # Where grid lines are white (255), set original to white (remove the line)
    cleaned = gray.copy()
    cleaned[all_grid_lines == 255] = 255  # Set grid line pixels to white

    # Convert back to BGR if input was BGR
    if len(image_np.shape) == 3:
        cleaned_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    else:
        cleaned_bgr = cleaned

    # Save debug images if requested
    if output_debug and output_dir:
        # Grid lines mask
        grid_mask_path = output_dir / f"{base_name}_grid_lines_detected.png"
        cv2.imwrite(str(grid_mask_path), all_grid_lines)
        stats['grid_mask_path'] = str(grid_mask_path)

        # Cleaned image
        cleaned_path = output_dir / f"{base_name}_grid_removed.png"
        cv2.imwrite(str(cleaned_path), cleaned_bgr)
        stats['cleaned_path'] = str(cleaned_path)

        # Create comparison image (side by side)
        if len(image_np.shape) == 3:
            original_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        else:
            original_gray = image_np
        comparison = np.hstack([original_gray, cleaned])
        comparison_path = output_dir / f"{base_name}_grid_removal_comparison.png"
        cv2.imwrite(str(comparison_path), comparison)
        stats['comparison_path'] = str(comparison_path)

    return cleaned_bgr, stats


def detect_grid_line_positions_from_image(image_np: np.ndarray,
                                           min_line_length_ratio: float = 0.4) -> Tuple[List[int], List[int]]:
    """
    Detect Y positions of horizontal grid lines and X positions of vertical grid lines.

    Returns:
        Tuple of (horizontal_y_positions, vertical_x_positions)
    """
    # Convert to grayscale
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_np.copy()

    height, width = gray.shape

    # Invert and threshold
    inverted = cv2.bitwise_not(gray)
    _, binary = cv2.threshold(inverted, 200, 255, cv2.THRESH_BINARY)

    # Detect horizontal lines
    min_h_length = int(width * min_line_length_ratio)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_h_length, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)

    # Find Y positions of horizontal lines (row indices with most white pixels)
    h_projection = np.sum(h_lines, axis=1)
    h_threshold = width * 0.3 * 255  # At least 30% of width
    horizontal_y_positions = np.where(h_projection > h_threshold)[0].tolist()

    # Cluster nearby Y positions
    horizontal_y_clustered = cluster_positions(horizontal_y_positions, tolerance=10)

    # Detect vertical lines
    min_v_length = int(height * min_line_length_ratio)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_v_length))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=1)

    # Find X positions of vertical lines
    v_projection = np.sum(v_lines, axis=0)
    v_threshold = height * 0.3 * 255
    vertical_x_positions = np.where(v_projection > v_threshold)[0].tolist()

    # Cluster nearby X positions
    vertical_x_clustered = cluster_positions(vertical_x_positions, tolerance=10)

    return horizontal_y_clustered, vertical_x_clustered


def cluster_positions(positions: List[int], tolerance: int = 10) -> List[int]:
    """Cluster nearby positions and return cluster centers."""
    if not positions:
        return []

    positions = sorted(positions)
    clusters = []
    current_cluster = [positions[0]]

    for pos in positions[1:]:
        if pos - current_cluster[-1] <= tolerance:
            current_cluster.append(pos)
        else:
            # Save cluster center
            clusters.append(int(np.mean(current_cluster)))
            current_cluster = [pos]

    # Don't forget last cluster
    clusters.append(int(np.mean(current_cluster)))

    return clusters


# ============================================================
# IMPROVED DASH PATTERN PARSING
# ============================================================

def parse_dash_pattern_improved(dashes) -> Optional[List[float]]:
    """
    Parse dash patterns from PyMuPDF format.

    PyMuPDF can return dashes in multiple formats:
    - None or [] for solid lines
    - [dash, gap] as list
    - "[3 4] 0" as string (includes phase)

    Args:
        dashes: Dash pattern from PyMuPDF

    Returns:
        List of [dash, gap] values or None for solid lines
    """
    if dashes is None:
        return None

    # Handle empty list/tuple
    if isinstance(dashes, (list, tuple)) and len(dashes) == 0:
        return None

    # Handle string format "[3 4] 0" or "[3 4 2 1] 0"
    if isinstance(dashes, str):
        dashes = dashes.strip()
        if not dashes or dashes == "[]":
            return None

        # Parse "[values] phase" format
        match = re.match(r'\[([\d\s.]+)\]\s*([\d.]*)', dashes)
        if match:
            values_str = match.group(1).strip()
            if not values_str:
                return None
            values = [float(x) for x in values_str.split() if x]
            if values and any(v > 0 for v in values):
                return values
        return None

    # Handle list/tuple format
    if isinstance(dashes, (list, tuple)):
        values = [float(d) for d in dashes if d is not None and d > 0]
        return values if values else None

    return None


def is_segment_dashed(segment) -> bool:
    """
    Check if a segment is dashed, with improved parsing.
    """
    # First try the built-in property
    if hasattr(segment, 'is_dashed') and segment.is_dashed:
        return True

    # Then try improved parsing
    if hasattr(segment, 'dashes'):
        parsed = parse_dash_pattern_improved(segment.dashes)
        return parsed is not None and len(parsed) > 0

    return False


# ============================================================
# STAGE 3: VLM VALIDATION (Claude Vision)
# ============================================================

def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def validate_walls_with_vlm(
    original_image_path: str,
    walls_overlay_path: str,
    wall_count: int,
    filter_stats: Dict,
    api_key: str = None
) -> Dict:
    """
    Use Claude Vision to validate wall detection results.

    Args:
        original_image_path: Path to original floor plan image
        walls_overlay_path: Path to image with detected walls highlighted
        wall_count: Number of walls detected
        filter_stats: Statistics from filtering process
        api_key: Anthropic API key (uses env var if not provided)

    Returns:
        Dict with validation results and corrections
    """
    if not ANTHROPIC_AVAILABLE:
        return {
            "status": "skipped",
            "reason": "anthropic package not installed",
            "corrections": []
        }

    # Get API key
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {
            "status": "skipped",
            "reason": "No API key provided (set ANTHROPIC_API_KEY env var)",
            "corrections": []
        }

    try:
        client = anthropic.Anthropic(api_key=api_key)

        # Encode images
        original_b64 = encode_image_to_base64(original_image_path)
        overlay_b64 = encode_image_to_base64(walls_overlay_path)

        # Determine image type
        ext = Path(original_image_path).suffix.lower()
        media_type = "image/png" if ext == ".png" else "image/jpeg"

        # Create validation prompt
        prompt = f"""You are an expert architectural drawing analyst. I need you to validate automated wall detection results on a floor plan.

**IMAGE 1 (First image)**: Original floor plan
**IMAGE 2 (Second image)**: Same floor plan with detected walls highlighted in BLUE (interior) and RED (exterior/heavy)

**Current Detection Statistics:**
- Total walls detected: {wall_count}
- Segments filtered as colored MEP: {filter_stats.get('colored_mep', 0)}
- Segments filtered as too short: {filter_stats.get('too_short', 0)}
- Segments filtered as grid lines: {filter_stats.get('dashed_grid', 0)}

**YOUR TASK:**
Analyze both images and identify:

1. **FALSE POSITIVES** - Elements highlighted as walls that are NOT walls:
   - Grid lines (structural reference lines, usually labeled A-Z or 1-10 at edges)
   - Dimension lines (measurement annotations)
   - MEP symbols (mechanical/electrical/plumbing symbols)
   - Furniture or equipment outlines
   - Hatching patterns

2. **FALSE NEGATIVES** - Actual walls that were MISSED:
   - Interior partition walls
   - Exterior walls
   - Partial walls or wall segments

3. **QUALITY ASSESSMENT**:
   - Overall accuracy estimate (percentage)
   - Are room boundaries clearly defined by detected walls?
   - Major issues that need fixing?

**RESPOND IN THIS JSON FORMAT:**
```json
{{
    "overall_accuracy_percent": <number 0-100>,
    "quality_rating": "<excellent|good|fair|poor>",
    "false_positives": [
        {{
            "type": "<grid_line|dimension_line|mep_symbol|furniture|hatching|other>",
            "location": "<description of where on the drawing>",
            "severity": "<high|medium|low>"
        }}
    ],
    "false_negatives": [
        {{
            "type": "<interior_wall|exterior_wall|partial_wall>",
            "location": "<description of where on the drawing>",
            "severity": "<high|medium|low>"
        }}
    ],
    "recommendations": [
        "<specific recommendation for improving detection>"
    ],
    "rooms_properly_bounded": <true|false>,
    "summary": "<2-3 sentence summary of wall detection quality>"
}}
```

Be specific about locations using room names, grid references, or positional descriptions (top-left, center, etc.)."""

        # Call Claude Vision
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": original_b64
                            }
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": overlay_b64
                            }
                        }
                    ]
                }
            ]
        )

        # Parse response
        response_text = response.content[0].text

        # Try to extract JSON from response
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            validation_result = json.loads(json_str)
        else:
            # Try parsing entire response as JSON
            try:
                validation_result = json.loads(response_text)
            except:
                validation_result = {
                    "raw_response": response_text,
                    "parse_error": "Could not parse JSON from response"
                }

        validation_result["status"] = "success"
        return validation_result

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "corrections": []
        }


def filter_hatching_patterns(segments, angle_tolerance=2.0, min_group_size=5, spacing_tolerance=0.5):
    """
    Filter out hatching patterns (groups of parallel lines at regular spacing).

    Returns (filtered_segments, num_removed)
    """
    if len(segments) < min_group_size:
        return segments, 0

    # Calculate angle and length for each segment
    seg_data = []
    for seg in segments:
        dx = seg.end[0] - seg.start[0]
        dy = seg.end[1] - seg.start[1]
        angle = math.degrees(math.atan2(dy, dx)) % 180  # Normalize to 0-180
        length = (dx**2 + dy**2)**0.5
        seg_data.append((seg, angle, length))

    # Group by angle
    angle_groups = defaultdict(list)
    for seg, angle, length in seg_data:
        # Round angle to tolerance
        rounded_angle = round(angle / angle_tolerance) * angle_tolerance
        angle_groups[rounded_angle].append((seg, length))

    # Find hatching groups (many parallel lines of similar length)
    hatching_segments = set()
    for angle, group in angle_groups.items():
        if len(group) >= min_group_size:
            lengths = [length for seg, length in group]
            avg_length = sum(lengths) / len(lengths)

            # Skip if average length is very long (these are walls, not hatching)
            if avg_length > 200:
                continue

            # Hatching lines are typically short and uniform length
            length_variance = sum((l - avg_length)**2 for l in lengths) / len(lengths)
            cv = (length_variance ** 0.5) / avg_length if avg_length > 0 else 0

            # If coefficient of variation is low and there are many lines, it's hatching
            # Hatching typically has 10+ parallel lines of uniform length
            if cv < spacing_tolerance and len(group) >= 10:
                for seg, length in group:
                    hatching_segments.add(id(seg))

    # Filter out hatching
    filtered = [seg for seg in segments if id(seg) not in hatching_segments]
    return filtered, len(hatching_segments)


def filter_grid_lines(segments, page_width, page_height, span_threshold=0.5, min_grid_length=400):
    """
    Filter out structural grid lines (lines that span most of the page).
    Also detects grid lines that are broken into multiple segments.

    Returns (filtered_segments, num_removed)
    """
    from collections import defaultdict

    # First pass: identify line chains (segments at same Y or X that together span page)
    horizontal_by_y = defaultdict(list)
    vertical_by_x = defaultdict(list)

    for i, seg in enumerate(segments):
        x1, y1 = seg.start
        x2, y2 = seg.end
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        length = (dx**2 + dy**2)**0.5

        if length < 10:
            continue

        # Check if horizontal
        if dy < length * 0.05:
            y_rounded = round((y1 + y2) / 2 / 5) * 5  # Round to nearest 5 pts
            horizontal_by_y[y_rounded].append((i, seg, length, min(x1, x2), max(x1, x2)))

        # Check if vertical
        if dx < length * 0.05:
            x_rounded = round((x1 + x2) / 2 / 5) * 5
            vertical_by_x[x_rounded].append((i, seg, length, min(y1, y2), max(y1, y2)))

    # Find grid line chains (segments that together span significant portion)
    grid_segment_ids = set()

    # Check horizontal chains
    for y, segs in horizontal_by_y.items():
        if len(segs) < 3:
            continue
        total_length = sum(l for i, s, l, xmin, xmax in segs)
        x_min = min(xmin for i, s, l, xmin, xmax in segs)
        x_max = max(xmax for i, s, l, xmin, xmax in segs)
        span = (x_max - x_min) / page_width

        # If chain spans >60% of page, it's likely a grid line
        if span > 0.6 and total_length > 1500:
            for i, s, l, xmin, xmax in segs:
                grid_segment_ids.add(i)

    # Check vertical chains
    for x, segs in vertical_by_x.items():
        if len(segs) < 3:
            continue
        total_length = sum(l for i, s, l, ymin, ymax in segs)
        y_min = min(ymin for i, s, l, ymin, ymax in segs)
        y_max = max(ymax for i, s, l, ymin, ymax in segs)
        span = (y_max - y_min) / page_height

        # If chain spans >60% of page, it's likely a grid line
        if span > 0.6 and total_length > 1000:
            for i, s, l, ymin, ymax in segs:
                grid_segment_ids.add(i)

    # Second pass: filter individual long segments
    filtered = []
    removed = 0

    for i, seg in enumerate(segments):
        # Skip if part of grid chain
        if i in grid_segment_ids:
            removed += 1
            continue

        x1, y1 = seg.start
        x2, y2 = seg.end
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        length = (dx**2 + dy**2)**0.5

        # Check if very long single segment
        is_horizontal = dy < length * 0.05 if length > 0 else False
        is_vertical = dx < length * 0.05 if length > 0 else False

        if (is_horizontal or is_vertical) and length > min_grid_length:
            removed += 1
            continue

        filtered.append(seg)

    return filtered, removed


def filter_door_swings(segments, arc_tolerance=15.0, min_arc_segments=3, max_arc_length=100):
    """
    Filter door swing arcs - groups of short segments that form curved patterns.

    Door swings are typically:
    - Multiple short segments forming an arc
    - Located near door openings
    - Segments share endpoints or are very close
    - Total arc spans roughly 90 degrees

    Also detects arcs by finding segments at similar radius from potential pivot points.

    Returns (filtered_segments, num_removed)
    """
    if len(segments) < min_arc_segments:
        return segments, 0

    from collections import defaultdict
    import math

    arc_segment_ids = set()

    # METHOD 1: Connected segments sharing endpoints
    endpoint_groups = defaultdict(list)

    for i, seg in enumerate(segments):
        x1, y1 = seg.start
        x2, y2 = seg.end
        length = seg.length

        if length > max_arc_length:
            continue

        key1 = (round(x1 / arc_tolerance), round(y1 / arc_tolerance))
        key2 = (round(x2 / arc_tolerance), round(y2 / arc_tolerance))

        endpoint_groups[key1].append(i)
        endpoint_groups[key2].append(i)

    visited = set()
    for key, seg_ids in endpoint_groups.items():
        if len(seg_ids) >= 2:
            chain = set(seg_ids)
            to_check = list(seg_ids)
            while to_check:
                sid = to_check.pop()
                if sid in visited:
                    continue
                visited.add(sid)

                seg = segments[sid]
                x1, y1 = seg.start
                x2, y2 = seg.end

                for x, y in [(x1, y1), (x2, y2)]:
                    k = (round(x / arc_tolerance), round(y / arc_tolerance))
                    for connected_id in endpoint_groups.get(k, []):
                        if connected_id not in chain and segments[connected_id].length <= max_arc_length:
                            chain.add(connected_id)
                            to_check.append(connected_id)

            if len(chain) >= min_arc_segments:
                arc_segment_ids.update(chain)

    # METHOD 2: Detect arcs by finding segments at similar radius from common centers
    # This catches arcs that don't share clean endpoints
    short_segments = [(i, seg) for i, seg in enumerate(segments)
                      if seg.length <= max_arc_length and seg.length >= 5]

    # Group short segments by their midpoint position (potential arc location)
    grid_size = 50  # Group arcs within 50pt regions
    arc_regions = defaultdict(list)

    for i, seg in short_segments:
        mid_x = (seg.start[0] + seg.end[0]) / 2
        mid_y = (seg.start[1] + seg.end[1]) / 2
        region = (int(mid_x / grid_size), int(mid_y / grid_size))
        arc_regions[region].append((i, seg, mid_x, mid_y))

    # For each region with multiple short segments, check if they form an arc
    for region, seg_list in arc_regions.items():
        if len(seg_list) < 4:
            continue

        # Check all pairs of endpoints as potential arc centers
        for i, seg, mid_x, mid_y in seg_list:
            x1, y1 = seg.start
            x2, y2 = seg.end

            # Try each endpoint as potential pivot
            for pivot_x, pivot_y in [(x1, y1), (x2, y2)]:
                # Find other segments at similar radius from this pivot
                segments_at_radius = []

                for j, seg2, _, _ in seg_list:
                    if i == j:
                        continue

                    # Calculate distance from pivot to segment midpoint
                    m2_x = (seg2.start[0] + seg2.end[0]) / 2
                    m2_y = (seg2.start[1] + seg2.end[1]) / 2
                    dist = math.sqrt((m2_x - pivot_x)**2 + (m2_y - pivot_y)**2)

                    # Door swings are typically 24-36 inches (70-100 pts at 1/8" scale)
                    if 20 < dist < 120:
                        segments_at_radius.append((j, dist))

                # If we found multiple segments at similar radius, it's likely an arc
                if len(segments_at_radius) >= 3:
                    # Check radius consistency
                    distances = [d for _, d in segments_at_radius]
                    avg_dist = sum(distances) / len(distances)
                    variance = sum((d - avg_dist)**2 for d in distances) / len(distances)

                    # Low variance means consistent radius = arc
                    if variance < 100:  # Allow ~10pt variation
                        arc_segment_ids.add(i)
                        for j, _ in segments_at_radius:
                            arc_segment_ids.add(j)

    filtered = [seg for i, seg in enumerate(segments) if i not in arc_segment_ids]
    return filtered, len(arc_segment_ids)


def filter_curved_segments(segments, max_curve_length=60, curvature_threshold=0.15):
    """
    Filter segments that are part of curved shapes (door swings, circles, arcs).

    Curved segments have endpoints where the angle between adjacent segments
    changes progressively (unlike walls which are straight).

    Returns (filtered_segments, num_removed)
    """
    from collections import defaultdict
    import math

    if len(segments) < 3:
        return segments, 0

    # Build adjacency by endpoints
    endpoint_map = defaultdict(list)
    tolerance = 5  # pts

    for i, seg in enumerate(segments):
        if seg.length > max_curve_length:
            continue

        x1, y1 = seg.start
        x2, y2 = seg.end

        key1 = (round(x1 / tolerance), round(y1 / tolerance))
        key2 = (round(x2 / tolerance), round(y2 / tolerance))

        endpoint_map[key1].append((i, 'start', seg))
        endpoint_map[key2].append((i, 'end', seg))

    curved_ids = set()

    # Find segments that connect with angle changes (curves)
    for key, connections in endpoint_map.items():
        if len(connections) < 2:
            continue

        # Calculate angles of all segments at this junction
        angles = []
        for idx, end_type, seg in connections:
            x1, y1 = seg.start
            x2, y2 = seg.end

            if end_type == 'start':
                angle = math.atan2(y2 - y1, x2 - x1)
            else:
                angle = math.atan2(y1 - y2, x1 - x2)

            angles.append((idx, angle, seg.length))

        # Sort by angle
        angles.sort(key=lambda x: x[1])

        # Check for progressive angle changes (characteristic of curves)
        for i in range(len(angles)):
            for j in range(i + 1, min(i + 4, len(angles))):
                idx1, a1, len1 = angles[i]
                idx2, a2, len2 = angles[j]

                angle_diff = abs(a2 - a1)
                if angle_diff > math.pi:
                    angle_diff = 2 * math.pi - angle_diff

                # Curves have small progressive angle changes (not 90° corners)
                # Typical curve segment angle: 5-30 degrees
                if 0.05 < angle_diff < 0.6 and len1 < max_curve_length and len2 < max_curve_length:
                    curved_ids.add(idx1)
                    curved_ids.add(idx2)

    filtered = [seg for i, seg in enumerate(segments) if i not in curved_ids]
    return filtered, len(curved_ids)


def filter_room_label_boxes(segments, box_size_min=15, box_size_max=80, tolerance=5):
    """
    Filter small rectangular boxes that surround room labels/numbers.

    Room label boxes are typically:
    - 4 segments forming a rectangle
    - Small size (15-80 pts per side)
    - H/V oriented segments

    Returns (filtered_segments, num_removed)
    """
    if len(segments) < 4:
        return segments, 0

    from collections import defaultdict

    # Separate horizontal and vertical segments
    horizontal = []  # (index, y, x_min, x_max, length)
    vertical = []    # (index, x, y_min, y_max, length)

    for i, seg in enumerate(segments):
        x1, y1 = seg.start
        x2, y2 = seg.end
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        length = seg.length

        # Only consider segments in the right size range
        if length < box_size_min or length > box_size_max:
            continue

        # Check if horizontal
        if dy < length * 0.1:
            y_avg = (y1 + y2) / 2
            horizontal.append((i, y_avg, min(x1, x2), max(x1, x2), length))
        # Check if vertical
        elif dx < length * 0.1:
            x_avg = (x1 + x2) / 2
            vertical.append((i, x_avg, min(y1, y2), max(y1, y2), length))

    # Find rectangular patterns
    box_segment_ids = set()

    # Group horizontal segments by Y position
    h_by_y = defaultdict(list)
    for idx, y, x_min, x_max, length in horizontal:
        y_rounded = round(y / tolerance)
        h_by_y[y_rounded].append((idx, y, x_min, x_max, length))

    # Group vertical segments by X position
    v_by_x = defaultdict(list)
    for idx, x, y_min, y_max, length in vertical:
        x_rounded = round(x / tolerance)
        v_by_x[x_rounded].append((idx, x, y_min, y_max, length))

    # Look for pairs of horizontal lines at similar X range
    y_keys = sorted(h_by_y.keys())
    for i, y1_key in enumerate(y_keys):
        for y2_key in y_keys[i+1:]:
            y_diff = abs(y2_key - y1_key) * tolerance
            if y_diff < box_size_min or y_diff > box_size_max:
                continue

            # Check if there are horizontal segments at both Y positions
            for h1_idx, h1_y, h1_xmin, h1_xmax, h1_len in h_by_y[y1_key]:
                for h2_idx, h2_y, h2_xmin, h2_xmax, h2_len in h_by_y[y2_key]:
                    # Check if X ranges overlap significantly
                    x_overlap_min = max(h1_xmin, h2_xmin)
                    x_overlap_max = min(h1_xmax, h2_xmax)

                    if x_overlap_max - x_overlap_min < box_size_min * 0.5:
                        continue

                    # Look for vertical segments connecting them
                    left_x = round(min(h1_xmin, h2_xmin) / tolerance)
                    right_x = round(max(h1_xmax, h2_xmax) / tolerance)

                    left_verticals = []
                    right_verticals = []

                    for x_key in [left_x - 1, left_x, left_x + 1]:
                        for v_idx, v_x, v_ymin, v_ymax, v_len in v_by_x.get(x_key, []):
                            if v_ymin <= max(h1_y, h2_y) + tolerance and v_ymax >= min(h1_y, h2_y) - tolerance:
                                left_verticals.append(v_idx)

                    for x_key in [right_x - 1, right_x, right_x + 1]:
                        for v_idx, v_x, v_ymin, v_ymax, v_len in v_by_x.get(x_key, []):
                            if v_ymin <= max(h1_y, h2_y) + tolerance and v_ymax >= min(h1_y, h2_y) - tolerance:
                                right_verticals.append(v_idx)

                    # If we found a complete rectangle
                    if left_verticals and right_verticals:
                        box_segment_ids.add(h1_idx)
                        box_segment_ids.add(h2_idx)
                        box_segment_ids.update(left_verticals)
                        box_segment_ids.update(right_verticals)

    # Filter out box segments
    filtered = [seg for i, seg in enumerate(segments) if i not in box_segment_ids]
    return filtered, len(box_segment_ids)


def filter_diagonal_hatching(segments, angle_tolerance=5.0, min_count=20):
    """
    Filter diagonal lines that are likely hatching (45-degree patterns).

    Returns (filtered_segments, num_removed)
    """
    filtered = []
    diagonal_segments = []

    for seg in segments:
        dx = seg.end[0] - seg.start[0]
        dy = seg.end[1] - seg.start[1]
        length = (dx**2 + dy**2)**0.5

        if length < 10:
            filtered.append(seg)
            continue

        angle = abs(math.degrees(math.atan2(dy, dx)))
        # Normalize to 0-90 range
        if angle > 90:
            angle = 180 - angle

        # Check if diagonal (around 45 degrees or 135 degrees)
        is_diagonal = (40 < angle < 50) or (130 < angle < 140)

        if is_diagonal and length < 150:  # Short diagonal = likely hatching
            diagonal_segments.append(seg)
        else:
            filtered.append(seg)

    # Only filter if we found many diagonal lines (indicates hatching pattern)
    if len(diagonal_segments) >= min_count:
        return filtered, len(diagonal_segments)
    else:
        # Not enough to be hatching, keep them all
        return filtered + diagonal_segments, 0


def visualize_walls(pdf_path: str, output_path: str,
                    min_width: float = 0.8,
                    exclude_title_block: bool = True,
                    title_block_pct: float = 0.15,
                    exclude_margins: bool = True,
                    margin_pct: float = 0.05,
                    min_length: float = 10.0,
                    filter_hatching: bool = True):
    """
    Extract walls and create annotated PDF showing just the wall segments.

    Args:
        pdf_path: Input PDF path
        output_path: Output PDF path
        min_width: Minimum line width to include (filters thin annotation lines)
        exclude_title_block: Remove segments in title block area
        title_block_pct: Title block as percentage of page width (right side)
        exclude_margins: Remove segments near page edges
        margin_pct: Margin as percentage of page dimensions
        min_length: Minimum segment length to include
        filter_hatching: Attempt to filter hatching patterns
    """
    print(f"\n{'='*60}")
    print(f"WALL DETECTION: {Path(pdf_path).name}")
    print(f"{'='*60}")

    doc = pymupdf.open(pdf_path)
    page = doc[0]
    page_width = page.rect.width
    page_height = page.rect.height
    print(f"Page size: {page_width:.1f} x {page_height:.1f} points")

    # Extract wall segments
    print(f"\n--- Extracting Wall Segments ---")
    segments_raw = extract_wall_segments_simple(page)
    print(f"Found {len(segments_raw)} raw segments")

    if not segments_raw:
        print("ERROR: No wall segments found!")
        doc.close()
        return

    # === FILTERING ===
    print(f"\n--- Applying Filters ---")
    print(f"  Min width: {min_width} pts")
    print(f"  Min length: {min_length} pts")
    print(f"  Exclude title block: {exclude_title_block} ({title_block_pct*100:.0f}% from right)")
    print(f"  Exclude margins: {exclude_margins} ({margin_pct*100:.0f}%)")

    # Calculate boundaries
    title_block_x = page_width * (1 - title_block_pct) if exclude_title_block else page_width
    margin_left = page_width * margin_pct if exclude_margins else 0
    margin_right = page_width * (1 - margin_pct) if exclude_margins else page_width
    margin_top = page_height * margin_pct if exclude_margins else 0
    margin_bottom = page_height * (1 - margin_pct) if exclude_margins else page_height

    # Also exclude bottom area where dimension strings typically are
    dimension_area_y = page_height * 0.92  # Bottom 8% often has dimensions

    filtered_segments = []
    filter_stats = {
        'too_thin': 0,
        'too_short': 0,
        'in_title_block': 0,
        'in_margin': 0,
        'in_dimension_area': 0,
        'hatching': 0,
    }

    for seg in segments_raw:
        x1, y1 = seg.start
        x2, y2 = seg.end
        width = seg.width if hasattr(seg, 'width') and seg.width is not None else 1.0
        length = ((x2-x1)**2 + (y2-y1)**2)**0.5

        # Filter by width
        if width < min_width:
            filter_stats['too_thin'] += 1
            continue

        # Filter by length
        if length < min_length:
            filter_stats['too_short'] += 1
            continue

        # Filter title block (right side)
        if exclude_title_block:
            if x1 > title_block_x and x2 > title_block_x:
                filter_stats['in_title_block'] += 1
                continue

        # Filter margins (page edges - often have border lines)
        if exclude_margins:
            # Check if segment is entirely in margin area
            in_left_margin = x1 < margin_left and x2 < margin_left
            in_top_margin = y1 < margin_top and y2 < margin_top
            if in_left_margin or in_top_margin:
                filter_stats['in_margin'] += 1
                continue

        # Filter dimension string area (bottom of page)
        if y1 > dimension_area_y and y2 > dimension_area_y:
            filter_stats['in_dimension_area'] += 1
            continue

        filtered_segments.append(seg)

    # Grid line filter - remove lines spanning most of page
    filtered_segments, grid_removed = filter_grid_lines(filtered_segments, page_width, page_height)
    filter_stats['grid_lines'] = grid_removed

    # Hatching filter - detect parallel lines at regular spacing
    if filter_hatching and len(filtered_segments) > 100:
        filtered_segments, hatching_removed = filter_hatching_patterns(filtered_segments)
        filter_stats['hatching'] = hatching_removed

        # Also filter diagonal hatching
        filtered_segments, diagonal_removed = filter_diagonal_hatching(filtered_segments)
        filter_stats['diagonal_hatching'] = diagonal_removed

    print(f"\nFilter results:")
    print(f"  Too thin (< {min_width} pts): {filter_stats['too_thin']}")
    print(f"  Too short (< {min_length} pts): {filter_stats['too_short']}")
    print(f"  In title block: {filter_stats['in_title_block']}")
    print(f"  In margins: {filter_stats['in_margin']}")
    print(f"  In dimension area: {filter_stats['in_dimension_area']}")
    print(f"  Grid lines (span >70%): {filter_stats['grid_lines']}")
    print(f"  Hatching patterns: {filter_stats['hatching']}")
    print(f"  Diagonal hatching: {filter_stats.get('diagonal_hatching', 0)}")
    print(f"  KEPT: {len(filtered_segments)} of {len(segments_raw)} ({100*len(filtered_segments)/len(segments_raw):.1f}%)")

    segments = filtered_segments

    # Analyze filtered segment properties
    widths = [s.width for s in segments if hasattr(s, 'width') and s.width is not None]
    lengths = [((s.end[0]-s.start[0])**2 + (s.end[1]-s.start[1])**2)**0.5 for s in segments]

    print(f"\nFiltered segment statistics:")
    if widths:
        print(f"  Width range: {min(widths):.2f} - {max(widths):.2f} pts")
    else:
        print(f"  Width: No width data available")
    if lengths:
        print(f"  Length range: {min(lengths):.1f} - {max(lengths):.1f} pts")
        print(f"  Average length: {sum(lengths)/len(lengths):.1f} pts")

    # Group by width to understand wall types
    width_groups = {}
    for seg in segments:
        w = seg.width if hasattr(seg, 'width') and seg.width is not None else 1.0
        w = round(w, 1)
        if w not in width_groups:
            width_groups[w] = 0
        width_groups[w] += 1

    print(f"\nFiltered segments by line width:")
    for w in sorted(width_groups.keys())[:10]:
        print(f"  Width {w:.1f}: {width_groups[w]} segments")
    if len(width_groups) > 10:
        print(f"  ... and {len(width_groups) - 10} more width values")

    # Create output PDF with wall visualization
    print(f"\n--- Creating Wall Visualization ---")

    # Open fresh copy for annotation
    out_doc = pymupdf.open(pdf_path)
    out_page = out_doc[0]

    # Draw filtered wall segments
    shape = out_page.new_shape()

    # Color code by line weight
    for seg in segments:
        x1, y1 = seg.start
        x2, y2 = seg.end
        width = seg.width if hasattr(seg, 'width') and seg.width is not None else 1.0

        # Color based on line weight - now all should be walls
        if width >= 1.5:
            color = (1, 0, 0)  # Red - heavy walls
        elif width >= 1.0:
            color = (0, 0, 1)  # Blue - standard walls
        else:
            color = (0, 0.7, 0)  # Green - light walls/partitions

        shape.draw_line((x1, y1), (x2, y2))
        shape.finish(color=color, width=max(0.5, width * 0.5))

    shape.commit()

    # Also draw the exclusion zones for reference
    ref_shape = out_page.new_shape()
    # Title block boundary
    ref_shape.draw_line((title_block_x, 0), (title_block_x, page_height))
    ref_shape.finish(color=(1, 0.5, 0), width=2, dashes="[5 5]")  # Orange dashed
    # Dimension area boundary
    ref_shape.draw_line((0, dimension_area_y), (title_block_x, dimension_area_y))
    ref_shape.finish(color=(1, 0.5, 0), width=2, dashes="[5 5]")  # Orange dashed
    ref_shape.commit()

    # Save output
    out_doc.save(output_path)
    out_doc.close()
    doc.close()

    print(f"Wall visualization saved to: {output_path}")
    print(f"\nColor legend:")
    print(f"  RED    = Heavy walls (width >= 1.5 pts)")
    print(f"  BLUE   = Standard walls (width 1.0-1.5 pts)")
    print(f"  GREEN  = Light walls/partitions (width 0.8-1.0 pts)")
    print(f"  ORANGE DASHED = Exclusion zone boundaries")
    print(f"\n{'='*60}")


def detect_grid_positions(segments, page_width, page_height, tolerance=8.0):
    """
    Detect Y positions where horizontal grid lines exist (multiple segments at same Y).
    Detect X positions where vertical grid lines exist (multiple segments at same X).
    Returns sets of (y_positions, x_positions) that are grid lines.
    """
    from collections import defaultdict

    horizontal_y = defaultdict(list)
    vertical_x = defaultdict(list)

    for seg in segments:
        x1, y1 = seg.start
        x2, y2 = seg.end
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        length = (dx**2 + dy**2)**0.5

        if length < 20:
            continue

        # Horizontal segments
        if dy < length * 0.05:
            y_avg = (y1 + y2) / 2
            y_rounded = round(y_avg / tolerance) * tolerance
            horizontal_y[y_rounded].append((seg, length, min(x1, x2), max(x1, x2)))

        # Vertical segments
        if dx < length * 0.05:
            x_avg = (x1 + x2) / 2
            x_rounded = round(x_avg / tolerance) * tolerance
            vertical_x[x_rounded].append((seg, length, min(y1, y2), max(y1, y2)))

    # Find Y positions that are grid lines (segments span significant width)
    grid_y_positions = set()
    for y, segs in horizontal_y.items():
        total_length = sum(l for s, l, xmin, xmax in segs)
        x_min = min(xmin for s, l, xmin, xmax in segs)
        x_max = max(xmax for s, l, xmin, xmax in segs)
        span = (x_max - x_min) / page_width

        # Grid lines span >40% of page width OR have 2+ segments totaling >500 pts
        if span > 0.4 or (len(segs) >= 2 and total_length > 500):
            grid_y_positions.add(y)

    # Find X positions that are grid lines
    grid_x_positions = set()
    for x, segs in vertical_x.items():
        total_length = sum(l for s, l, ymin, ymax in segs)
        y_min = min(ymin for s, l, ymin, ymax in segs)
        y_max = max(ymax for s, l, ymin, ymax in segs)
        span = (y_max - y_min) / page_height

        # Grid lines span >40% of page height OR have 2+ segments totaling >400 pts
        if span > 0.4 or (len(segs) >= 2 and total_length > 400):
            grid_x_positions.add(x)

    return grid_y_positions, grid_x_positions


def is_wall_color(color):
    """
    Check if segment color is black/dark (walls) vs colored (MEP symbols).

    Walls are drawn in black or very dark gray.
    MEP symbols (mechanical, electrical, plumbing) use colored lines.

    Args:
        color: RGB tuple from segment, or None

    Returns:
        True if color is black/dark (likely wall), False if colored (likely MEP)
    """
    if color is None:
        return True  # No color info = assume black

    if len(color) == 0:
        return True  # Empty = assume black

    if len(color) == 1:
        # Grayscale - single value
        return color[0] < 0.2  # Dark if < 20% brightness

    if len(color) >= 3:
        # RGB - all channels must be dark (<0.2 = 20% brightness)
        return all(c < 0.2 for c in color[:3])

    return True  # Default to wall if unknown format


def visualize_walls_only(pdf_path: str, output_path: str):
    """
    Extract and visualize ONLY walls (no doors, hatching, symbols, MEP).

    Walls are identified by:
    - Solid lines (not dashed - filters grid lines)
    - Black/dark color (not colored - filters MEP symbols)
    - Line weight >= 0.5 pts (filters annotations)
    - Horizontal or vertical orientation
    - Within drawing area (not in title block)

    Generates multiple debug output files for verification.
    """
    print(f"\n{'='*60}")
    print(f"WALL-ONLY DETECTION: {Path(pdf_path).name}")
    print(f"{'='*60}")

    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(pdf_path).stem

    doc = pymupdf.open(pdf_path)
    page = doc[0]
    page_width = page.rect.width
    page_height = page.rect.height
    print(f"Page size: {page_width:.1f} x {page_height:.1f} points")

    # ========================================
    # STEP 1: Detect Drawing Area (Title Block)
    # ========================================
    print(f"\n--- Step 1: Detecting Drawing Area ---")

    # Render page for title block detection
    page_image_np = render_page_to_image(page, dpi=150)  # Returns BGR numpy array
    print(f"Rendered page image: {page_image_np.shape}")

    # Convert numpy BGR to PIL RGB for TitleBlockDetector
    page_image_pil = Image.fromarray(page_image_np[:, :, ::-1])  # BGR -> RGB

    # Detect title block boundary
    detector = TitleBlockDetector()
    tb_result = detector.detect([page_image_pil])
    title_block_x1 = tb_result['x1']  # Fraction 0.0-1.0

    # If title block detection failed or low confidence, use more conservative default
    if tb_result.get('confidence', 0) < 0.5:
        title_block_x1 = 0.78  # Use 78% instead of 85% to catch more title block content
        print(f"  Low confidence detection - using conservative 78% boundary")

    drawing_x_max = title_block_x1 * page_width - (page_width * 0.02)  # With 2% margin

    print(f"Title block detected at x={title_block_x1:.1%} of page width")
    print(f"Drawing area: 0 to {drawing_x_max:.0f} pts (of {page_width:.0f} total)")
    print(f"Detection method: {tb_result.get('method', 'unknown')}")
    print(f"Confidence: {tb_result.get('confidence', 0):.0%}")

    # OUTPUT 1: Drawing area visualization
    debug_img = page_image_pil.copy()
    draw = ImageDraw.Draw(debug_img)
    tb_x_pixels = int(title_block_x1 * debug_img.width)
    # Draw vertical line at detected title block boundary
    draw.line([(tb_x_pixels, 0), (tb_x_pixels, debug_img.height)], fill='red', width=5)
    # Add label
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    draw.text((tb_x_pixels + 20, 100), f"Title Block\nx1={title_block_x1:.1%}", fill='red', font=font)
    draw.text((50, 100), "DRAWING AREA", fill='green', font=font)

    drawing_area_path = output_dir / f"{base_name}_drawing_area.png"
    debug_img.save(drawing_area_path)
    print(f"OUTPUT 1: Drawing area visualization -> {drawing_area_path}")

    # OUTPUT 5: Drawing area crop (do it now while we have the image)
    crop_x = int(title_block_x1 * page_image_pil.width)
    drawing_crop = page_image_pil.crop((0, 0, crop_x, page_image_pil.height))
    drawing_crop_path = output_dir / f"{base_name}_drawing_crop.png"
    drawing_crop.save(drawing_crop_path)
    print(f"OUTPUT 5: Drawing area crop -> {drawing_crop_path}")

    # ========================================
    # STEP 1b: Morphological Grid Line Detection
    # ========================================
    print(f"\n--- Step 1b: Morphological Grid Detection ---")

    # Run morphological grid line detection on the drawing area
    grid_cleaned, grid_stats = remove_grid_lines_morphological(
        page_image_np,
        min_line_length_ratio=0.3,
        output_debug=True,
        output_dir=output_dir,
        base_name=base_name
    )
    print(f"Grid lines detected:")
    print(f"  Horizontal lines: {grid_stats['horizontal_lines_removed']}")
    print(f"  Vertical lines: {grid_stats['vertical_lines_removed']}")
    print(f"  Total: {grid_stats['total_lines_removed']}")
    if 'grid_mask_path' in grid_stats:
        print(f"OUTPUT 1b: Grid lines mask -> {grid_stats['grid_mask_path']}")
    if 'cleaned_path' in grid_stats:
        print(f"OUTPUT 1c: Grid-removed image -> {grid_stats['cleaned_path']}")

    # Also detect grid line positions for segment filtering
    h_grid_positions, v_grid_positions = detect_grid_line_positions_from_image(
        page_image_np, min_line_length_ratio=0.4
    )
    print(f"Detected grid positions:")
    print(f"  Horizontal Y positions: {len(h_grid_positions)} lines")
    print(f"  Vertical X positions: {len(v_grid_positions)} lines")

    # ========================================
    # STEP 2: Extract All Segments
    # ========================================
    print(f"\n--- Step 2: Extracting Segments ---")
    segments_raw = extract_wall_segments_simple(page)
    print(f"Found {len(segments_raw)} raw segments")

    # Analyze segment properties (using improved dash detection)
    dashed_count = sum(1 for s in segments_raw if is_segment_dashed(s))
    colored_count = sum(1 for s in segments_raw if not is_wall_color(s.color))
    print(f"  Dashed segments (improved detection): {dashed_count}")
    print(f"  Colored segments: {colored_count}")

    # OUTPUT 2: Segment analysis visualization
    print(f"\n--- Creating Segment Analysis ---")
    analysis_doc = pymupdf.open(pdf_path)
    analysis_page = analysis_doc[0]
    shape = analysis_page.new_shape()

    for seg in segments_raw:
        x1, y1 = seg.start
        x2, y2 = seg.end
        seg_width = seg.width if seg.width is not None else 0

        # Color code by segment type (in priority order)
        if is_segment_dashed(seg):  # Use improved dash detection
            color = (1, 0, 0)  # Red - dashed (grid lines)
        elif not is_wall_color(seg.color):
            color = (0, 0, 1)  # Blue - colored (MEP symbols)
        elif seg_width > 0 and seg_width < 0.3:
            color = (0, 0.8, 0)  # Green - thin (annotations)
        else:
            color = (0.3, 0.3, 0.3)  # Dark gray - solid (potential walls)

        shape.draw_line((x1, y1), (x2, y2))
        shape.finish(color=color, width=0.5)

    shape.commit()

    # Draw title block boundary
    ref_shape = analysis_page.new_shape()
    ref_shape.draw_line((drawing_x_max, 0), (drawing_x_max, page_height))
    ref_shape.finish(color=(1, 0.5, 0), width=3, dashes="[10 5]")
    ref_shape.commit()

    segments_analysis_path = output_dir / f"{base_name}_segments_analysis.pdf"
    analysis_doc.save(segments_analysis_path)
    analysis_doc.close()
    print(f"OUTPUT 2: Segment analysis -> {segments_analysis_path}")
    print(f"  Legend: RED=dashed(grid), BLUE=colored(MEP), GREEN=thin(annotation), GRAY=solid(walls)")

    # ========================================
    # STEP 3: Filter Segments to Walls Only
    # ========================================
    print(f"\n--- Step 3: Filtering for Walls Only ---")

    walls = []
    stats = {
        'outside_drawing': 0,
        'dashed_grid': 0,
        'morpho_grid': 0,
        'colored_mep': 0,
        'annotation': 0,
        'too_short': 0,
        'too_thin': 0,
        'diagonal': 0,
        'too_long': 0,
        'kept': 0,
    }

    # Convert morphological grid positions from image pixels to PDF points
    # Image was rendered at 150 DPI, PDF is 72 DPI
    dpi_scale = 150 / 72  # Image pixels per PDF point
    h_grid_pts = set(int(y / dpi_scale) for y in h_grid_positions)
    v_grid_pts = set(int(x / dpi_scale) for x in v_grid_positions)
    grid_tolerance = 8  # pts tolerance for matching

    def is_on_grid_line(y_pts, x_pts, is_horizontal, is_vertical):
        """Check if segment is on a detected grid line position."""
        if is_horizontal:
            for gy in h_grid_pts:
                if abs(y_pts - gy) < grid_tolerance:
                    return True
        if is_vertical:
            for gx in v_grid_pts:
                if abs(x_pts - gx) < grid_tolerance:
                    return True
        return False

    for seg in segments_raw:
        x1, y1 = seg.start
        x2, y2 = seg.end
        width = seg.width if seg.width is not None else 0
        length = seg.length

        # 1. Outside drawing area (in title block/schedules)
        if x1 > drawing_x_max and x2 > drawing_x_max:
            stats['outside_drawing'] += 1
            continue

        # 2. Dashed lines = grid lines (using improved detection)
        if is_segment_dashed(seg):
            stats['dashed_grid'] += 1
            continue

        # 3. Colored lines = MEP symbols (KEY FILTER)
        if not is_wall_color(seg.color):
            stats['colored_mep'] += 1
            continue

        # 4. Very thin = annotations (0 < width < 0.2) - RELAXED from 0.3
        if width > 0 and width < 0.2:
            stats['annotation'] += 1
            continue

        # 5. Too short = symbols/details (< 15 pts) - RELAXED from 20
        if length < 15:
            stats['too_short'] += 1
            continue

        # 6. Medium thin = not walls (0.2 <= width < 0.4) - RELAXED from 0.5
        if width > 0 and width < 0.4:
            stats['too_thin'] += 1
            continue

        # 7. Check H/V orientation
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        is_horizontal = dy < length * 0.1 if length > 0 else False
        is_vertical = dx < length * 0.1 if length > 0 else False

        # 8. Diagonal = hatching/symbols
        if not is_horizontal and not is_vertical:
            stats['diagonal'] += 1
            continue

        # 9. Check if segment aligns with morphologically detected grid lines
        y_center = (y1 + y2) / 2
        x_center = (x1 + x2) / 2
        if length > 150 and is_on_grid_line(y_center, x_center, is_horizontal, is_vertical):
            stats['morpho_grid'] += 1
            continue

        # 10. Too long = remaining grid lines (likely not walls if > 500 pts) - RELAXED from 300
        if length > 500:
            stats['too_long'] += 1
            continue

        walls.append(seg)
        stats['kept'] += 1

    # Apply additional grid chain filter for any remaining grid segments
    walls_before_chain = len(walls)
    walls, grid_removed = filter_grid_lines(walls, page_width, page_height)
    stats['dashed_grid'] += grid_removed

    # Filter door swings (arc patterns) - DISABLED as it removes wall corners
    # walls, door_swing_removed = filter_door_swings(walls, arc_tolerance=8.0, min_arc_segments=6, max_arc_length=60)
    door_swing_removed = 0
    stats['door_swings'] = 0

    # Filter curved segments (door arcs, circles) - disabled as it removes wall corners
    curved_removed = 0
    stats['curved'] = 0

    # Filter room label boxes - RELAXED: only filter very small boxes (likely room number tags)
    walls, label_box_removed = filter_room_label_boxes(walls, box_size_min=10, box_size_max=50, tolerance=5)
    stats['label_boxes'] = label_box_removed

    print(f"\nFilter results:")
    print(f"  Outside drawing area: {stats['outside_drawing']}")
    print(f"  Dashed (grid lines): {stats['dashed_grid']}")
    print(f"  Morpho grid (image-detected): {stats['morpho_grid']}")
    print(f"  Colored (MEP symbols): {stats['colored_mep']}")
    print(f"  Annotations (< 0.3 pts): {stats['annotation']}")
    print(f"  Too short (< 20 pts): {stats['too_short']}")
    print(f"  Too thin (0.3-0.5 pts): {stats['too_thin']}")
    print(f"  Diagonal (hatching): {stats['diagonal']}")
    print(f"  Too long (> 300 pts): {stats['too_long']}")
    print(f"  Grid chain filter: {grid_removed}")
    print(f"  Door swings (arcs): {door_swing_removed}")
    print(f"  Curved segments: {curved_removed}")
    print(f"  Room label boxes: {label_box_removed}")
    print(f"  WALLS KEPT: {len(walls)} of {len(segments_raw)} ({100*len(walls)/len(segments_raw):.1f}%)")

    # OUTPUT 3: Filter statistics file
    stats_path = output_dir / f"{base_name}_filter_stats.txt"
    with open(stats_path, 'w') as f:
        f.write(f"Wall Detection Filter Statistics\n")
        f.write(f"================================\n")
        f.write(f"Input: {pdf_path}\n")
        f.write(f"Page size: {page_width:.1f} x {page_height:.1f} pts\n")
        f.write(f"Title block at: {title_block_x1:.1%} (x={drawing_x_max:.0f} pts)\n")
        f.write(f"Total raw segments: {len(segments_raw)}\n\n")
        f.write(f"Filter Results:\n")
        for key, count in stats.items():
            pct = 100 * count / len(segments_raw) if segments_raw else 0
            f.write(f"  {key}: {count} ({pct:.1f}%)\n")
        f.write(f"\nWalls kept: {len(walls)} ({100*len(walls)/len(segments_raw):.1f}%)\n")

        # Wall statistics
        if walls:
            widths = [s.width for s in walls if s.width is not None]
            lengths = [s.length for s in walls]
            f.write(f"\nWall Statistics:\n")
            f.write(f"  Width range: {min(widths):.2f} - {max(widths):.2f} pts\n")
            f.write(f"  Length range: {min(lengths):.1f} - {max(lengths):.1f} pts\n")
            f.write(f"  Average length: {sum(lengths)/len(lengths):.1f} pts\n")

    print(f"OUTPUT 3: Filter statistics -> {stats_path}")

    # Analyze wall segments
    if walls:
        widths = [s.width for s in walls if s.width is not None]
        lengths = [s.length for s in walls]
        print(f"\nWall statistics:")
        print(f"  Width range: {min(widths):.2f} - {max(widths):.2f} pts")
        print(f"  Length range: {min(lengths):.1f} - {max(lengths):.1f} pts")
        print(f"  Average length: {sum(lengths)/len(lengths):.1f} pts")

    # ========================================
    # STEP 4: Create Final Wall Visualization
    # ========================================
    print(f"\n--- Step 4: Creating Wall Visualization ---")

    # Create TWO outputs:
    # 1. Walls overlaid on original PDF (for context)
    # 2. Walls ONLY on white background (clean view)

    # --- Output 4a: Walls on original PDF ---
    out_doc = pymupdf.open(pdf_path)
    out_page = out_doc[0]

    shape = out_page.new_shape()

    for seg in walls:
        x1, y1 = seg.start
        x2, y2 = seg.end
        width = seg.width if seg.width is not None else 1.0

        # Color by wall weight
        if width >= 1.5:
            color = (1, 0, 0)  # Red - heavy walls (exterior/structural)
        else:
            color = (0, 0, 1)  # Blue - standard walls (interior)

        shape.draw_line((x1, y1), (x2, y2))
        shape.finish(color=color, width=1.5)

    shape.commit()

    # Draw exclusion zone boundary
    ref_shape = out_page.new_shape()
    ref_shape.draw_line((drawing_x_max, 0), (drawing_x_max, page_height))
    ref_shape.finish(color=(1, 0.5, 0), width=3, dashes="[10 5]")
    ref_shape.commit()

    # --- Output 4b: Walls ONLY on white background (clean view) ---
    clean_doc = pymupdf.open()  # New blank document
    clean_page = clean_doc.new_page(width=page_width, height=page_height)

    # Fill with white background
    clean_page.draw_rect(clean_page.rect, color=(1, 1, 1), fill=(1, 1, 1))

    # Draw walls only
    clean_shape = clean_page.new_shape()

    for seg in walls:
        x1, y1 = seg.start
        x2, y2 = seg.end
        width = seg.width if seg.width is not None else 1.0

        # Color by wall weight - using darker colors for visibility on white
        if width >= 1.5:
            color = (0.8, 0, 0)  # Dark red - heavy walls
        else:
            color = (0, 0, 0.8)  # Dark blue - standard walls

        # Draw with actual wall width for accuracy
        clean_shape.draw_line((x1, y1), (x2, y2))
        clean_shape.finish(color=color, width=max(1.0, width))

    clean_shape.commit()

    # Draw drawing area boundary
    clean_ref = clean_page.new_shape()
    clean_ref.draw_line((drawing_x_max, 0), (drawing_x_max, page_height))
    clean_ref.finish(color=(1, 0.5, 0), width=2, dashes="[10 5]")
    clean_ref.commit()

    # Save clean walls-only PDF
    clean_output_path = output_dir / f"{base_name}_walls_clean.pdf"
    clean_doc.save(str(clean_output_path))
    print(f"OUTPUT 4b: Clean walls only -> {clean_output_path}")

    # Also save as PNG for easy viewing
    clean_pix = clean_page.get_pixmap(dpi=150)
    clean_png_path = output_dir / f"{base_name}_walls_clean.png"
    clean_pix.save(str(clean_png_path))
    print(f"OUTPUT 4c: Clean walls PNG -> {clean_png_path}")

    clean_doc.close()

    out_doc.save(output_path)

    # Render the wall overlay as an image for VLM validation
    walls_overlay_path = output_dir / f"{base_name}_walls_overlay.png"
    out_page_pix = out_doc[0].get_pixmap(dpi=150)
    out_page_pix.save(str(walls_overlay_path))
    print(f"OUTPUT 6: Wall overlay image -> {walls_overlay_path}")

    out_doc.close()
    doc.close()

    print(f"OUTPUT 4: Wall visualization -> {output_path}")
    print(f"\nColor legend:")
    print(f"  RED  = Heavy walls (width >= 1.5 pts) - exterior/structural")
    print(f"  BLUE = Standard walls (width 0.5-1.5 pts) - interior partitions")
    print(f"  ORANGE DASHED = Drawing area boundary")

    # ========================================
    # STEP 5: VLM Validation (Optional)
    # ========================================
    print(f"\n--- Step 5: VLM Validation ---")
    vlm_result_path = output_dir / f"{base_name}_vlm_validation.json"

    if ANTHROPIC_AVAILABLE and os.environ.get("ANTHROPIC_API_KEY"):
        print("Running VLM validation with Claude Vision...")
        vlm_result = validate_walls_with_vlm(
            original_image_path=str(drawing_crop_path),
            walls_overlay_path=str(walls_overlay_path),
            wall_count=len(walls),
            filter_stats=stats
        )

        # Save VLM results
        with open(vlm_result_path, 'w') as f:
            json.dump(vlm_result, f, indent=2)
        print(f"OUTPUT 7: VLM validation results -> {vlm_result_path}")

        if vlm_result.get('status') == 'success':
            print(f"\nVLM Assessment:")
            print(f"  Overall accuracy: {vlm_result.get('overall_accuracy_percent', 'N/A')}%")
            print(f"  Quality rating: {vlm_result.get('quality_rating', 'N/A')}")
            print(f"  Rooms properly bounded: {vlm_result.get('rooms_properly_bounded', 'N/A')}")
            if vlm_result.get('false_positives'):
                print(f"  False positives: {len(vlm_result['false_positives'])}")
            if vlm_result.get('false_negatives'):
                print(f"  False negatives: {len(vlm_result['false_negatives'])}")
            if vlm_result.get('summary'):
                print(f"  Summary: {vlm_result['summary']}")
        else:
            print(f"  VLM validation: {vlm_result.get('status', 'unknown')} - {vlm_result.get('reason', vlm_result.get('error', ''))}")
    else:
        print("VLM validation skipped (no API key or anthropic package not installed)")
        print("To enable: set ANTHROPIC_API_KEY environment variable")
        vlm_result = {"status": "skipped", "reason": "No API key"}
        with open(vlm_result_path, 'w') as f:
            json.dump(vlm_result, f, indent=2)

    # ========================================
    # Summary of all output files
    # ========================================
    print(f"\n{'='*60}")
    print(f"ALL OUTPUT FILES:")
    print(f"  1. {drawing_area_path}")
    print(f"  2. {segments_analysis_path}")
    print(f"  3. {stats_path}")
    print(f"  4. {output_path}")
    print(f"  5. {drawing_crop_path}")
    print(f"  6. {walls_overlay_path}")
    print(f"  7. {vlm_result_path}")
    if 'grid_mask_path' in grid_stats:
        print(f"  8. {grid_stats['grid_mask_path']}")
    if 'cleaned_path' in grid_stats:
        print(f"  9. {grid_stats['cleaned_path']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Test floor plans
    floor_plans = [
        {
            "name": "Ellinwood",
            "pdf": "C:/tb/blueprint_processor/output/classified_sheets/floor_plans/2018-1203_Ellinwood_GMP_Permit_A1.1E_LEVEL_01_FLOOR_PLAN.pdf",
            "output": "C:/measure/test_output/ellinwood_fixed/ellinwood_walls_only.pdf"
        },
        {
            "name": "Woodstock",
            "pdf": "C:/tb/blueprint_processor/output/classified_sheets/floor_plans/2025-05-16_Woodstock_Recreatio_A111_CONSTRUCTION_PLAN_-_LEVEL_1.pdf",
            "output": "C:/measure/test_output/woodstock_fixed/woodstock_walls_only.pdf"
        }
    ]

    # Run wall-only detection on each floor plan
    for plan in floor_plans:
        print(f"\n\n{'#'*80}")
        print(f"# PROCESSING: {plan['name']}")
        print(f"{'#'*80}\n")

        # Create output directory
        output_dir = Path(plan['output']).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        visualize_walls_only(plan['pdf'], plan['output'])

    print(f"\n\n{'#'*80}")
    print(f"# DONE - All floor plans processed")
    print(f"{'#'*80}")
