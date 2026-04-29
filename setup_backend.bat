@echo off
setlocal enabledelayedexpansion

echo ========================================
echo ANTIGRAV TRADING - Backend Setup
echo ========================================
echo.

cd /d "C:\Users\Wren C\Documents\ANTIGRAV TRADING"

echo [1/5] Checking Python version...
python --version
if !errorlevel! neq 0 (
    echo ERROR: Python not found or not in PATH
    exit /b 1
)
echo.

echo [2/5] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists, skipping creation
) else (
    python -m venv venv
    if !errorlevel! neq 0 (
        echo ERROR: Failed to create virtual environment
        exit /b 1
    )
    echo Virtual environment created successfully
)
echo.

echo [3/5] Activating virtual environment...
call venv\Scripts\activate.bat
if !errorlevel! neq 0 (
    echo ERROR: Failed to activate virtual environment
    exit /b 1
)
echo Virtual environment activated
echo.

echo [4/5] Installing dependencies from pyproject.toml...
pip install -e .
if !errorlevel! neq 0 (
    echo ERROR: Failed to install dependencies
    exit /b 1
)
echo Dependencies installed successfully
echo.

echo [5/5] Starting Uvicorn server...
echo Server starting on http://127.0.0.1:8000
echo Press Ctrl+C to stop the server
echo.
echo === BACKEND SERVER ===
python -m uvicorn antigravity.gateway.server:app --host 127.0.0.1 --port 8000 --reload

endlocal
