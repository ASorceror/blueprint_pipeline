@echo off
REM Blueprint Pipeline - Windows Batch Runner
REM
REM Usage examples:
REM   run_pipeline.bat -i plans.pdf -o output/
REM   run_pipeline.bat -i plans.pdf -o output/ --scale "1/8 inch = 1 foot"
REM   run_pipeline.bat -i plans.pdf -o output/ --pages 1-5 --no-ocr
REM

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..

REM Activate virtual environment
call "%PROJECT_DIR%\venv\Scripts\activate.bat"

REM Run the pipeline with all passed arguments
python -m blueprint_pipeline.cli %*

REM Deactivate virtual environment
deactivate
