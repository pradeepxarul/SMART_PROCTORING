@echo off
echo ============================================
echo   Interview Proctoring Service
echo ============================================
echo.

:: Activate virtual environment
if exist "venv_311\Scripts\activate.bat" (
    call venv_311\Scripts\activate.bat
) else (
    echo [!] Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

echo Starting server on http://localhost:5000 ...
echo Press Ctrl+C to stop.
echo.
python server.py
pause