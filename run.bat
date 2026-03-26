@echo off
echo ============================================
echo   Interview Proctoring Service
echo ============================================
echo.

:: Use venv python directly
set "VENV_PY=venv_311\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo [!] Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

echo Starting server on http://localhost:5000 ...
echo Press Ctrl+C to stop.
echo.
%VENV_PY% server.py
pause