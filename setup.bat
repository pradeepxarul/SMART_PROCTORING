@echo off
echo ============================================
echo   Interview Proctoring Service - Setup
echo ============================================
echo.

:: ---- Resolve venv (use py launcher to target Python 3.11) ----
set "VENV_DIR=venv_311"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo [1/4] Creating virtual environment with Python 3.11...
    py -3.11 -m venv %VENV_DIR%
    if %ERRORLEVEL% neq 0 (
        echo [!] Failed. Make sure Python 3.11 is installed.
        echo     Download from https://www.python.org/downloads/release/python-3119/
        pause
        exit /b 1
    )
    echo     Created %VENV_DIR%
) else (
    echo [1/4] %VENV_DIR% already exists.
)

:: ---- Upgrade pip, pin setuptools (82+ breaks face_recognition) ----
echo [2/4] Upgrading pip...
%VENV_PY% -m pip install --upgrade pip wheel >nul 2>&1
%VENV_PY% -m pip install "setuptools<81" >nul 2>&1

:: ---- Install dlib from local wheel ----
echo [3/4] Installing dependencies...
if exist "dlib-19.24.1-cp311-cp311-win_amd64.whl" (
    echo     Installing dlib from local wheel...
    %VENV_PY% -m pip install dlib-19.24.1-cp311-cp311-win_amd64.whl >nul 2>&1
)

:: ---- Install everything from requirements.txt ----
%VENV_PY% -m pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [!] Dependency installation failed. Check errors above.
    pause
    exit /b 1
)

:: ---- YOLO model ----
echo [4/4] Checking YOLOv8 model...
if not exist "yolov8n.pt" (
    echo     Downloading yolov8n.pt...
    %VENV_PY% -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
) else (
    echo     yolov8n.pt found.
)

echo.
echo ============================================
echo   Setup Complete!
echo   Run 'run.bat' to start the server.
echo ============================================
pause
