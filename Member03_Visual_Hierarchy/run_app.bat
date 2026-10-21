@echo off
echo ======================================================================
echo Starting Visual Hierarchy & Attention Predictor Web App...
echo ======================================================================

cd /d "%~dp0"
call .venv\Scripts\activate.bat

echo Opening browser at http://127.0.0.1:8000 ...
start http://127.0.0.1:8000

python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
pause
