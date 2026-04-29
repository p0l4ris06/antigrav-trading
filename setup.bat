@echo off
REM ANTIGRAV TRADING Backend Setup Script
REM ==========================================

echo.
echo [STEP 1] Checking Python version...
python --version
echo.

echo [STEP 2] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    exit /b 1
)
echo Virtual environment created successfully
echo.

echo [STEP 3] Activating virtual environment...
call venv\Scripts\activate.bat
echo Virtual environment activated
echo.

echo [STEP 4] Installing dependencies from pyproject.toml...
pip install -e .
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    exit /b 1
)
echo Dependencies installed successfully
echo.

echo [STEP 5] Starting ANTIGRAV TRADING server...
echo Starting server on http://127.0.0.1:8000
echo Press Ctrl+C to stop the server
echo.
python -m uvicorn core.gateway:app --host 127.0.0.1 --port 8000 --reload
