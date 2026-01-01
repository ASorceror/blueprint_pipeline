"""
Command Line Interface Module

Parses command-line arguments for the blueprint pipeline.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from .constants import (
    DEFAULT_RENDER_DPI,
    DEFAULT_GAP_TOLERANCE_INCHES,
    DEFAULT_CEILING_HEIGHT_FT,
)


def parse_page_range(page_str: str, max_pages: int) -> List[int]:
    """
    Parse page range string to list of page numbers.

    Examples:
        "1-5" -> [0, 1, 2, 3, 4]
        "1,3,5" -> [0, 2, 4]
        "all" -> [0, 1, ..., max_pages-1]

    Args:
        page_str: Page specification string
        max_pages: Maximum page count

    Returns:
        List of 0-indexed page numbers
    """
    if page_str.lower() == "all":
        return list(range(max_pages))

    pages = set()

    for part in page_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            start = int(start) - 1  # Convert to 0-indexed
            end = int(end) - 1
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part) - 1)  # Convert to 0-indexed

    # Filter valid pages
    valid_pages = sorted([p for p in pages if 0 <= p < max_pages])
    return valid_pages


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for the pipeline."""
    parser = argparse.ArgumentParser(
        prog="blueprint_pipeline",
        description="Extract room measurements from commercial blueprint PDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.cli -i plans.pdf -o ./output
  python -m src.cli -i plans.pdf -o ./output --scale "1/8 inch = 1 foot"
  python -m src.cli -i plans.pdf -o ./output --pages 1-5 --verbose
        """
    )

    # Required arguments
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input PDF file path"
    )

    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output directory path"
    )

    # Optional arguments
    parser.add_argument(
        "--pages",
        default="all",
        help="Page range to process (e.g., '1-5', '1,3,5', default: all)"
    )

    parser.add_argument(
        "--units",
        choices=["imperial", "metric"],
        default="imperial",
        help="Output units (default: imperial)"
    )

    parser.add_argument(
        "--scale",
        help="Manual scale override (e.g., '1/8 inch = 1 foot')"
    )

    parser.add_argument(
        "--calib",
        help="Two-point calibration ('x1,y1:x2,y2=10ft')"
    )

    parser.add_argument(
        "--default-height",
        type=float,
        default=DEFAULT_CEILING_HEIGHT_FT,
        help=f"Default ceiling height if not detected (default: {DEFAULT_CEILING_HEIGHT_FT} ft)"
    )

    parser.add_argument(
        "--door-gap",
        type=float,
        default=DEFAULT_GAP_TOLERANCE_INCHES,
        help=f"Gap bridging tolerance in inches (default: {DEFAULT_GAP_TOLERANCE_INCHES})"
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_RENDER_DPI,
        help=f"Render DPI for raster processing (default: {DEFAULT_RENDER_DPI})"
    )

    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR (faster, may lose room labels)"
    )

    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Skip annotated PDF output"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate images for debugging"
    )

    # Detection mode options
    detection_group = parser.add_argument_group('detection mode')

    detection_group.add_argument(
        "--label-driven",
        action="store_true",
        default=True,
        help="Use label-driven detection (default, NEW approach)"
    )

    detection_group.add_argument(
        "--geometry-only",
        action="store_true",
        help="Use geometry-only detection (OLD approach, disables label-driven)"
    )

    detection_group.add_argument(
        "--label-confidence",
        type=float,
        default=0.5,
        help="Minimum label confidence threshold (default: 0.5)"
    )

    detection_group.add_argument(
        "--boundary-completeness",
        type=float,
        default=0.5,
        help="Minimum boundary wall coverage (default: 0.5)"
    )

    # Filter options
    filter_group = parser.add_argument_group('segment filtering')

    filter_group.add_argument(
        "--no-filters",
        action="store_true",
        help="Disable all segment filtering"
    )

    filter_group.add_argument(
        "--no-title-block-filter",
        action="store_true",
        help="Disable title block area filtering"
    )

    filter_group.add_argument(
        "--no-hatching-filter",
        action="store_true",
        help="Disable hatching pattern filtering"
    )

    filter_group.add_argument(
        "--no-dimension-filter",
        action="store_true",
        help="Disable dimension line filtering"
    )

    filter_group.add_argument(
        "--no-annotation-filter",
        action="store_true",
        help="Disable annotation line filtering"
    )

    return parser


def validate_args(args: argparse.Namespace) -> Tuple[bool, str]:
    """
    Validate parsed arguments.

    Args:
        args: Parsed arguments

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        return False, f"Input file not found: {args.input}"

    if not input_path.suffix.lower() == ".pdf":
        return False, f"Input file must be a PDF: {args.input}"

    # Check/create output directory
    output_path = Path(args.output)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return False, f"Cannot create output directory: {e}"

    # Validate DPI
    if args.dpi < 72 or args.dpi > 600:
        return False, f"DPI must be between 72 and 600: {args.dpi}"

    # Validate default height
    if args.default_height < 5 or args.default_height > 100:
        return False, f"Default height must be between 5 and 100: {args.default_height}"

    return True, ""


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse and validate command-line arguments.

    Args:
        args: Optional list of arguments (uses sys.argv if None)

    Returns:
        Parsed and validated arguments
    """
    parser = create_parser()
    parsed = parser.parse_args(args)

    is_valid, error_msg = validate_args(parsed)
    if not is_valid:
        parser.error(error_msg)

    return parsed


def main():
    """Main entry point for CLI."""
    args = parse_args()

    # Import pipeline and run
    from .pipeline import run_pipeline

    try:
        run_pipeline(args)
    except KeyboardInterrupt:
        print("\nProcessing cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
