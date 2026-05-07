@echo off
echo Starting live daemon in dry-run mode...
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else (
    call venv\Scripts\activate.bat
)
python live_daemon.py --dry-run --exchange binance
