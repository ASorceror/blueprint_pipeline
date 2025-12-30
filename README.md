# Blueprint Measurement Pipeline

A tool for extracting room measurements from commercial construction blueprints.

## Features

- Extracts rooms from vector and raster PDF blueprints
- Detects scale automatically from dimension strings and scale notations
- Matches room labels using embedded text and OCR
- Extracts ceiling heights from RCP (Reflected Ceiling Plan) sheets
- Calculates floor area, perimeter, wall area, and ceiling area
- Outputs CSV, JSON, and annotated PDF with toggleable room layer

## Requirements

- Python 3.10 or 3.11 (NOT 3.12)
- Windows 10/11 with NVIDIA GPU (optional, for faster OCR)

## Installation

1. Create and activate virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Verify installation:
   ```
   python scripts/verify_install.py
   ```

## Usage

Basic usage:
```
python -m blueprint_pipeline.cli -i plans.pdf -o output/
```

With manual scale:
```
python -m blueprint_pipeline.cli -i plans.pdf -o output/ --scale "1/8 inch = 1 foot"
```

See `python -m blueprint_pipeline.cli --help` for all options.

## Output Files

- `*_rooms.csv` - Room measurements in CSV format
- `*_rooms.json` - Detailed room data with metadata
- `*_annotated.pdf` - PDF with room outlines on toggleable layer
- `*.log` - Processing log

## Version

2.0.0
