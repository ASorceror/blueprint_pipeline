"""
Title Block Detector for Blueprint PDFs

Multi-stage detection pipeline combining:
1. CV Transition Detection (edge density analysis)
2. Hough Line Detection (vertical line detection)
3. Consensus Voting (robust multi-method fusion)

Adapted from blueprint_processor for use in blueprint_pipeline.
"""

import logging
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from pathlib import Path
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger(__name__)

# Try to import OpenCV for Hough line detection
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    logger.warning("OpenCV not available - Hough line detection disabled")

# Try to import scipy for smoothing
try:
    from scipy.ndimage import uniform_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("scipy not available - using numpy for smoothing")


def _smooth_array(arr: np.ndarray, size: int) -> np.ndarray:
    """Smooth array with uniform filter, using numpy fallback if scipy unavailable."""
    if HAS_SCIPY:
        return uniform_filter1d(arr.astype(float), size=size)
    else:
        # Simple moving average fallback
        kernel = np.ones(size) / size
        return np.convolve(arr.astype(float), kernel, mode='same')


# =============================================================================
# CV TRANSITION DETECTION
# =============================================================================

def get_common_edges(
    page_images: List[Image.Image],
    edge_threshold: int = 25
) -> np.ndarray:
    """
    Find edges that appear on ALL sample pages (common structure).

    Args:
        page_images: List of PIL Images
        edge_threshold: Threshold for edge detection

    Returns:
        2D numpy array of common edges (255 where edge on all pages)
    """
    edge_images = []
    target_size = None

    for img in page_images:
        if target_size is None:
            target_size = img.size
        elif img.size != target_size:
            img = img.resize(target_size, Image.Resampling.LANCZOS)

        gray = img.convert('L')
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges_array = np.array(edges)
        binary = (edges_array > edge_threshold).astype(np.uint8) * 255
        edge_images.append(binary)

    # Find edges present on ALL pages
    stack = np.stack(edge_images, axis=0)
    common_edges = np.all(stack > 128, axis=0).astype(np.uint8) * 255

    return common_edges


def get_majority_edges(
    page_images: List[Image.Image],
    edge_threshold: int = 25,
    agreement_ratio: float = 0.6
) -> np.ndarray:
    """
    Find edges that appear on MOST sample pages (majority vote).
    """
    edge_images = []
    target_size = None

    for img in page_images:
        if target_size is None:
            target_size = img.size
        elif img.size != target_size:
            img = img.resize(target_size, Image.Resampling.LANCZOS)

        gray = img.convert('L')
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges_array = np.array(edges)
        binary = (edges_array > edge_threshold).astype(np.uint8)
        edge_images.append(binary)

    # Count agreement
    stack = np.stack(edge_images, axis=0)
    agreement = np.sum(stack, axis=0)

    min_pages = int(np.ceil(len(page_images) * agreement_ratio))
    majority_edges = (agreement >= min_pages).astype(np.uint8) * 255

    return majority_edges


def find_transition_x1(
    common_edges: np.ndarray,
    search_start: float = 0.60,
    search_end: float = 0.97,
    window_size: int = 20,
    smoothing_size: int = 10
) -> Optional[float]:
    """
    Find the x position where edge density transitions from low to high.

    Returns:
        x1 as fraction (0.0-1.0) or None if not found
    """
    height, width = common_edges.shape

    # Calculate column edge counts
    col_edge_count = np.sum(common_edges > 128, axis=0)

    # Smooth
    col_smoothed = _smooth_array(col_edge_count, smoothing_size)

    # Determine threshold from the left portion (should be noise/drawing)
    left_portion = col_smoothed[:int(width * 0.5)]
    noise_level = np.percentile(left_portion, 95)
    threshold = max(noise_level * 2, 20)

    # Find transition point
    search_start_px = int(width * search_start)
    search_end_px = int(width * search_end)

    x1_px = None
    for i in range(search_start_px, search_end_px - window_size):
        window = col_smoothed[i:i + window_size]
        if np.all(window > threshold):
            x1_px = i
            break

    if x1_px is not None:
        return x1_px / width
    return None


def detect_title_block_cv(
    page_images: List[Image.Image],
    use_majority: bool = False,
    agreement_ratio: float = 0.6
) -> Optional[Dict]:
    """
    Detect title block left boundary using CV transition detection.
    """
    if not page_images:
        return None

    if use_majority:
        edges = get_majority_edges(page_images, agreement_ratio=agreement_ratio)
        method = 'cv_majority'
    else:
        edges = get_common_edges(page_images)
        method = 'cv_transition'

    x1 = find_transition_x1(edges)

    if x1 is not None:
        return {
            'x1': x1,
            'method': method,
            'width_pct': 1.0 - x1
        }
    return None


# =============================================================================
# HOUGH LINE DETECTION
# =============================================================================

def find_vertical_lines(
    page_image: Image.Image,
    search_region: Tuple[float, float] = (0.60, 1.0),
    min_line_length_ratio: float = 0.3,
    max_line_gap: int = 30,
    angle_tolerance: float = 5.0,
    threshold: int = 100
) -> List[Dict]:
    """
    Find strong vertical lines in the specified region of the page.
    """
    if not HAS_OPENCV:
        return []

    width, height = page_image.size

    # Convert to grayscale numpy array
    gray = np.array(page_image.convert('L'))

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Mask to only search in the specified region
    region_start = int(search_region[0] * width)
    region_end = int(search_region[1] * width)

    mask = np.zeros_like(edges)
    mask[:, region_start:region_end] = 255
    edges_masked = cv2.bitwise_and(edges, mask)

    # Detect lines using probabilistic Hough Transform
    min_line_length = int(height * min_line_length_ratio)

    lines = cv2.HoughLinesP(
        edges_masked,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    vertical_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate angle from vertical
            if x2 - x1 == 0:
                angle = 0  # Perfectly vertical
            else:
                angle = abs(np.degrees(np.arctan((y2 - y1) / (x2 - x1))))
                angle = 90 - angle  # Convert to angle from vertical

            # Only keep lines close to vertical
            if angle <= angle_tolerance:
                x_avg = (x1 + x2) / 2
                line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

                vertical_lines.append({
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'x_avg': x_avg,
                    'x_avg_pct': x_avg / width,
                    'length': line_length,
                    'length_pct': line_length / height,
                    'angle_from_vertical': angle
                })

    # Sort by x position (leftmost first)
    vertical_lines.sort(key=lambda l: l['x_avg'])

    return vertical_lines


def cluster_lines_by_x(
    lines: List[Dict],
    cluster_threshold: float = 0.02
) -> List[Dict]:
    """
    Cluster nearby vertical lines into groups.
    """
    if not lines:
        return []

    clusters = []
    current_cluster = [lines[0]]

    for line in lines[1:]:
        cluster_x = np.mean([l['x_avg_pct'] for l in current_cluster])

        if abs(line['x_avg_pct'] - cluster_x) <= cluster_threshold:
            current_cluster.append(line)
        else:
            clusters.append(_summarize_cluster(current_cluster))
            current_cluster = [line]

    if current_cluster:
        clusters.append(_summarize_cluster(current_cluster))

    return clusters


def _summarize_cluster(lines: List[Dict]) -> Dict:
    """Summarize a cluster of lines into a single line descriptor."""
    x_positions = [l['x_avg_pct'] for l in lines]
    lengths = [l['length_pct'] for l in lines]

    return {
        'x_avg_pct': np.mean(x_positions),
        'x_min_pct': min(x_positions),
        'x_max_pct': max(x_positions),
        'total_length_pct': sum(lengths),
        'max_length_pct': max(lengths),
        'num_segments': len(lines),
        'lines': lines
    }


def detect_title_block_hough(
    page_images: List[Image.Image],
    search_region: Tuple[float, float] = (0.60, 0.98),
    min_agreement: int = 2
) -> Optional[float]:
    """
    Find the title block left border using Hough line detection across multiple pages.
    """
    if not HAS_OPENCV:
        return None

    all_line_positions = []

    for img in page_images:
        lines = find_vertical_lines(img, search_region=search_region)
        clusters = cluster_lines_by_x(lines)

        for cluster in clusters:
            if cluster['total_length_pct'] > 0.3:
                all_line_positions.append(cluster['x_avg_pct'])

    if not all_line_positions:
        return None

    positions = np.array(sorted(all_line_positions))

    # Bin positions and find consensus
    bins = np.arange(search_region[0], search_region[1] + 0.02, 0.02)
    hist, bin_edges = np.histogram(positions, bins=bins)

    consistent_positions = []
    for i, count in enumerate(hist):
        if count >= min_agreement:
            bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
            consistent_positions.append((bin_center, count))

    if not consistent_positions:
        return None

    consistent_positions.sort(key=lambda x: x[0])
    return consistent_positions[0][0]


# =============================================================================
# CONSENSUS DETECTOR
# =============================================================================

class TitleBlockDetector:
    """
    Multi-method title block detector using consensus voting.

    Combines CV transition detection and Hough line detection
    to robustly find the title block left boundary.

    Usage:
        detector = TitleBlockDetector()
        result = detector.detect(page_images)
        x1 = result['x1']  # Left boundary as fraction (0.0-1.0)
    """

    def __init__(
        self,
        search_region: Tuple[float, float] = (0.60, 0.98),
        default_x1: float = 0.85
    ):
        """
        Initialize the detector.

        Args:
            search_region: Region to search for title block (start, end as fractions)
            default_x1: Fallback value if all methods fail
        """
        self.search_region = search_region
        self.default_x1 = default_x1
        self._learned_template = None

    def detect(
        self,
        page_images: List[Image.Image],
        strategy: str = 'balanced'
    ) -> Dict:
        """
        Detect title block using multi-method consensus.

        Args:
            page_images: List of PIL Images (sample pages from PDF)
            strategy: Detection strategy
                - 'balanced': Use median consensus
                - 'conservative': Use maximum (tightest boundary)
                - 'aggressive': Use minimum (widest boundary)

        Returns:
            Dict with detection results:
                - x1: Left boundary as fraction (0.0-1.0)
                - width_pct: Title block width as fraction
                - method: Detection method used
                - confidence: Confidence score (0.0-1.0)
                - estimates: Individual method estimates
        """
        if not page_images:
            return self._default_result("no_images")

        estimates = {}
        estimates_list = []

        # Method 1: CV Transition (strict AND)
        try:
            cv_result = detect_title_block_cv(page_images, use_majority=False)
            if cv_result and cv_result['x1']:
                estimates['cv_transition'] = cv_result['x1']
                estimates_list.append(cv_result['x1'])
        except Exception as e:
            logger.debug(f"CV transition detection failed: {e}")
            estimates['cv_transition_error'] = str(e)

        # Method 2: CV Majority (60% agreement)
        try:
            cv_majority = detect_title_block_cv(page_images, use_majority=True, agreement_ratio=0.6)
            if cv_majority and cv_majority['x1']:
                estimates['cv_majority'] = cv_majority['x1']
                estimates_list.append(cv_majority['x1'])
        except Exception as e:
            logger.debug(f"CV majority detection failed: {e}")
            estimates['cv_majority_error'] = str(e)

        # Method 3: Hough Line Detection
        try:
            hough_x1 = detect_title_block_hough(
                page_images,
                search_region=self.search_region,
                min_agreement=2
            )
            if hough_x1:
                estimates['hough_lines'] = hough_x1
                estimates_list.append(hough_x1)
        except Exception as e:
            logger.debug(f"Hough line detection failed: {e}")
            estimates['hough_lines_error'] = str(e)

        # Calculate consensus
        if not estimates_list:
            return self._default_result("all_methods_failed")

        if len(estimates_list) == 1:
            x1 = estimates_list[0]
            method_name = [k for k, v in estimates.items() if v == x1 and not k.endswith('_error')][0]
            return {
                'x1': x1,
                'width_pct': 1.0 - x1,
                'method': method_name,
                'confidence': 0.5,
                'estimates': estimates
            }

        # Multiple methods - filter outliers first
        estimates_array = np.array(estimates_list)

        # Remove outliers: values more than 0.10 from median
        initial_median = np.median(estimates_array)
        filtered = [e for e in estimates_list if abs(e - initial_median) < 0.10]

        if not filtered:
            filtered = estimates_list

        estimates_array = np.array(filtered)

        if strategy == 'balanced':
            x1 = float(np.median(estimates_array))
            method = 'consensus_median'
        elif strategy == 'conservative':
            x1 = float(np.max(estimates_array))
            method = 'consensus_conservative'
        elif strategy == 'aggressive':
            x1 = float(np.min(estimates_array))
            method = 'consensus_aggressive'
        else:
            x1 = float(np.median(estimates_array))
            method = 'consensus_median'

        spread = float(np.max(estimates_array) - np.min(estimates_array))

        # Confidence based on agreement
        if spread < 0.02:
            confidence = 0.95
        elif spread < 0.05:
            confidence = 0.80
        elif spread < 0.10:
            confidence = 0.60
        else:
            confidence = 0.40

        return {
            'x1': x1,
            'width_pct': 1.0 - x1,
            'method': method,
            'confidence': confidence,
            'estimates': estimates,
            'spread': spread,
            'num_methods': len(estimates_list)
        }

    def detect_with_learning(
        self,
        page_images: List[Image.Image],
        strategy: str = 'balanced'
    ) -> Dict:
        """
        Detect and learn the title block template for this PDF.
        """
        result = self.detect(page_images, strategy=strategy)

        self._learned_template = {
            'x1': result['x1'],
            'method': result['method'],
            'confidence': result['confidence']
        }

        return result

    def apply_learned(self, page_image: Image.Image) -> Dict:
        """
        Apply the learned template to a new page (fast, no detection).
        """
        if self._learned_template is None:
            return self.detect([page_image])

        return {
            'x1': self._learned_template['x1'],
            'width_pct': 1.0 - self._learned_template['x1'],
            'method': 'learned',
            'confidence': self._learned_template['confidence']
        }

    def crop_title_block(
        self,
        page_image: Image.Image,
        detection_result: Optional[Dict] = None
    ) -> Image.Image:
        """
        Crop the title block from a page image.
        """
        if detection_result is None:
            detection_result = self.detect([page_image])

        x1 = detection_result['x1']
        width, height = page_image.size

        x1_px = int(x1 * width)
        return page_image.crop((x1_px, 0, width, height))

    def crop_drawing_area(
        self,
        page_image: Image.Image,
        detection_result: Optional[Dict] = None
    ) -> Image.Image:
        """
        Crop just the drawing area (excluding title block).
        """
        if detection_result is None:
            detection_result = self.detect([page_image])

        x1 = detection_result['x1']
        width, height = page_image.size

        x1_px = int(x1 * width)
        return page_image.crop((0, 0, x1_px, height))

    def _default_result(self, reason: str) -> Dict:
        """Return a default result when detection fails."""
        return {
            'x1': self.default_x1,
            'width_pct': 1.0 - self.default_x1,
            'method': f'default_{reason}',
            'confidence': 0.0,
            'estimates': {}
        }

    def visualize_detection(
        self,
        page_image: Image.Image,
        result: Dict,
        output_path: Optional[Path] = None
    ) -> Image.Image:
        """
        Create visualization showing detected boundary and method estimates.
        """
        viz_img = page_image.convert('RGB')
        width, height = viz_img.size

        draw = ImageDraw.Draw(viz_img)

        # Draw each method's estimate with different colors
        colors = {
            'cv_transition': 'blue',
            'cv_majority': 'cyan',
            'hough_lines': 'orange'
        }

        y_offset = 20
        for method_name, color in colors.items():
            if method_name in result.get('estimates', {}):
                x1 = result['estimates'][method_name]
                x_px = int(x1 * width)

                # Dashed line for individual estimates
                for y in range(0, height, 20):
                    draw.line([(x_px, y), (x_px, min(y + 10, height))], fill=color, width=2)

                draw.text((x_px + 5, y_offset), f"{method_name}: {x1:.3f}", fill=color)
                y_offset += 25

        # Solid green line for final consensus
        consensus_x = int(result['x1'] * width)
        draw.line([(consensus_x, 0), (consensus_x, height)], fill='green', width=4)

        spread = result.get('spread', 0)
        draw.text((consensus_x + 5, y_offset + 10),
                 f"FINAL: {result['x1']:.3f} (spread={spread:.3f})",
                 fill='green')

        if output_path:
            viz_img.save(output_path)

        return viz_img


def detect_title_block(page_images: List[Image.Image]) -> float:
    """
    Convenience function to detect title block left boundary.

    Returns:
        x1 as fraction (0.0-1.0)
    """
    detector = TitleBlockDetector()
    result = detector.detect(page_images)
    return result['x1']
