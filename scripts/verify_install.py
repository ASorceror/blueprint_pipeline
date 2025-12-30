#!/usr/bin/env python
"""
Blueprint Pipeline - Installation Verification Script

Run this script to verify all dependencies are correctly installed.
"""

import sys
from pathlib import Path

# Add src to path for constants import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check_package(name: str, import_name: str = None, version_attr: str = "__version__") -> tuple[bool, str]:
    """Check if a package is installed and return version."""
    import_name = import_name or name
    try:
        module = __import__(import_name)
        version = getattr(module, version_attr, "unknown")
        return True, str(version)
    except ImportError as e:
        return False, str(e)


def check_tesseract() -> tuple[bool, str]:
    """Check if Tesseract executable is available."""
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        return True, str(version)
    except Exception as e:
        return False, f"Tesseract not installed or not in PATH: {e}"


def check_paddleocr() -> tuple[bool, str]:
    """Check if PaddleOCR is working."""
    try:
        from paddleocr import PaddleOCR
        return True, "available"
    except ImportError as e:
        return False, str(e)


def check_constants() -> tuple[bool, str]:
    """Check if constants module loads correctly."""
    try:
        from constants import (
            MIN_DRAWINGS_FOR_VECTOR,
            DEFAULT_COMMERCIAL_SCALE_FACTOR,
            DEFAULT_CEILING_HEIGHT_FT,
            Confidence,
            ProcessingPath,
        )
        return True, f"loaded ({MIN_DRAWINGS_FOR_VECTOR=}, {DEFAULT_COMMERCIAL_SCALE_FACTOR=})"
    except ImportError as e:
        return False, str(e)


def check_settings() -> tuple[bool, str]:
    """Check if settings.yaml loads correctly."""
    try:
        import yaml
        settings_path = Path(__file__).parent.parent / "config" / "settings.yaml"
        if not settings_path.exists():
            return False, "settings.yaml not found"
        with open(settings_path) as f:
            settings = yaml.safe_load(f)
        sections = list(settings.keys())
        return True, f"sections: {', '.join(sections)}"
    except Exception as e:
        return False, str(e)


def main():
    print("=" * 60)
    print("Blueprint Pipeline - Installation Verification")
    print("=" * 60)
    print()

    results = []

    # Core packages
    print("Core Dependencies:")
    print("-" * 40)

    packages = [
        ("pymupdf", "pymupdf", "__version__"),
        ("shapely", "shapely", "__version__"),
        ("opencv", "cv2", "__version__"),
        ("numpy", "numpy", "__version__"),
        ("pandas", "pandas", "__version__"),
        ("pyyaml", "yaml", "__version__"),
    ]

    for name, import_name, version_attr in packages:
        ok, info = check_package(name, import_name, version_attr)
        status = "PASS" if ok else "FAIL"
        print(f"  {name:25} [{status}] {info}")
        results.append((name, ok))

    print()
    print("OCR Engines:")
    print("-" * 40)

    # PaddleOCR
    ok, info = check_paddleocr()
    status = "PASS" if ok else "FAIL"
    print(f"  {'paddleocr':25} [{status}] {info}")
    results.append(("paddleocr", ok))

    # Tesseract (optional)
    ok, info = check_tesseract()
    status = "PASS" if ok else "WARN"  # Tesseract is optional
    print(f"  {'tesseract':25} [{status}] {info}")
    # Don't add to required results - it's optional

    print()
    print("Configuration:")
    print("-" * 40)

    # Constants
    ok, info = check_constants()
    status = "PASS" if ok else "FAIL"
    print(f"  {'constants.py':25} [{status}] {info}")
    results.append(("constants", ok))

    # Settings
    ok, info = check_settings()
    status = "PASS" if ok else "FAIL"
    print(f"  {'settings.yaml':25} [{status}] {info}")
    results.append(("settings", ok))

    print()
    print("=" * 60)

    # Summary
    passed = sum(1 for _, ok in results if ok)
    total = len(results)

    if passed == total:
        print(f"ALL CHECKS PASSED ({passed}/{total})")
        print("Environment is ready for blueprint processing.")
        return 0
    else:
        failed = [name for name, ok in results if not ok]
        print(f"SOME CHECKS FAILED ({passed}/{total})")
        print(f"Failed: {', '.join(failed)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
