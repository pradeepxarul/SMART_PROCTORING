@echo off
echo ============================================
echo   Interview Proctoring Service - Setup
echo ============================================
echo.

:: Skip Python version check - use available Python
echo [1/4] Skipping Python version check...

:: Create virtual environment
echo [2/4] Creating virtual environment...
if not exist "venv_311" (
    python -m venv venv_311
    echo     Created venv_311
) else (
    echo     venv_311 already exists, skipping.
)

:: Activate and install
echo [3/4] Installing dependencies...
call venv_311\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel >nul 2>&1

:: Install dlib from wheel if available
if exist "dlib-19.24.1-cp311-cp311-win_amd64.whl" (
    echo     Installing dlib from local wheel...
    pip install dlib-19.24.1-cp311-cp311-win_amd64.whl >nul 2>&1
)

pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [!] Dependency installation failed. Check errors above.
    pause
    exit /b 1
)

:: Download YOLO model if missing
echo [4/4] Checking YOLOv8 model...
if not exist "yolov8n.pt" (
    echo     Downloading yolov8n.pt...
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
) else (
    echo     yolov8n.pt found.
)

echo.
echo ============================================
echo   Setup Complete!
echo   Run 'run.bat' to start the server.
echo ============================================
pause
