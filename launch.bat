@echo off
echo ==========================================
echo  ANTIGRAVITY — LAUNCH
echo ==========================================
echo.
echo [1] Starting gateway server on http://localhost:8000
echo [2] Dashboard served at http://localhost:8000/
echo [3] Press Ctrl+C to stop
echo.
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else (
    call venv\Scripts\activate.bat
)
python -m uvicorn antigravity.gateway.server:app --host 0.0.0.0 --port 8000
