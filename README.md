# 🛡️ Interview Proctoring System

An advanced, intelligent web-based interview proctoring system that monitors candidates in real-time. It ensures interview integrity by tracking head movements, gaze directions, background individuals, and continuous identity verification.

## ✨ Key Features
- **Identity Verification:** Captures the candidate's face during enrollment and continuously matches it during the interview.
- **Head Pose Estimation:** Detects if the candidate is looking away (up, down, left, right) using 3D facial landmarks and `solvePnP`.
- **Gaze Tracking:** Monitors eye movement to detect if the candidate is reading from a hidden screen.
- **Multiple People Detection:** Uses YOLOv8 object detection to ensure only one person is in the frame.
- **Absence Detection:** Flags when the candidate leaves the camera view.
- **Strike System:** Employs a lenient grace period but strictly enforces a maximum strike limit before terminating the interview.

## 🛠️ Technology Stack
- **Backend:** Python, Flask, Flask-CORS
- **Frontend:** HTML5, CSS3 (Glassmorphism Premium UI), Vanilla JavaScript
- **Computer Vision & AI Models:**
  - **MediaPipe Face Mesh:** For precise 468-point facial landmark detection (used in head pose and gaze calculation).
  - **YOLOv8 (Ultralytics):** Highly accurate model for detecting multiple people in the background.
  - **Face Recognition (`dlib` based):** For encoding and verifying the candidate's identity.
  - **OpenCV:** Core image processing operations and `solvePnP` geometry.

## 📂 Project Structure
```text
OPEN_CV_FACE_RECOGNITION/
├── server.py               # Main Flask application and AI logic
├── login.html              # Enrollment and webcam setup page
├── webcam.html             # The actual proctored interview dashboard
├── unauthorized.html       # Page shown when a user exceeds max strikes
├── thankyou.html           # Page shown on successful interview completion
├── setup.bat               # Windows batch script for automated environment setup
├── run.bat                 # Windows batch script to launch the server
├── requirements.txt        # Python dependency list
├── yolov8n.pt              # YOLOv8 nano model weights (downloaded automatically)
└── dlib-19.24.1-...whl     # Precompiled dlib wheel for Python 3.11 (Windows)
```

## 🚀 Installation & Setup (Windows)

1. **Prerequisites:** Ensure you have **Python 3.11** installed.
2. **Run the Setup Script:** Double-click on `setup.bat`. This script will automatically:
   - Create a virtual environment (`venv_311`)
   - Install the pre-compiled `dlib` wheel
   - Install all Python dependencies from `requirements.txt`
   - Download the YOLOv8 model (`yolov8n.pt`)

## 🎮 Running the Application

1. Double-click on `run.bat`.
2. The Flask server will start on `http://localhost:5000` or `http://0.0.0.0:5000`.
3. Open your web browser and navigate to `http://localhost:5000`.

## 🔄 Application Flow

### 1. Enrollment (`login.html`)
- The candidate allows camera access.
- A snapshot is captured and sent to the `/enroll` endpoint.
- The server ensures exactly **one face** is present.
- A 128-dimensional face embedding is generated and stored **in-memory**.

### 2. Proctored Session (`webcam.html`)
- The UI initializes a session timer and strike counter.
- Every 1 second, a frame is sent to the `/analyze` endpoint via a canvas blob.
- **Grace Period:** The first 5 seconds do not accumulate strikes to allow the candidate to settle.
- Depending on the candidate's behavior, strikes are accumulated.
- The UI displays live badges (Normal, Head Turned, Multiple People, unknown Identity, etc.) via toast notifications.

### 3. Termination
- **Violations:** When strikes reach `MAX_STRIKES` (15 by default), the interview is abruptly ended, and the user is redirected to `unauthorized.html` (Integrity Violation Detected).
- **Completion:** If the 60-minute timer runs out or the candidate manually clicks "End Interview", they are redirected to `thankyou.html`.

## 🧠 How the AI Engine Works (`server.py`)

- **YOLOv8 Thread:** Analyzes the full frame to count the number of humans (`class=0`). If `count > 1`, a strike is issued.
- **MediaPipe Face Mesh:** Extracts 468 3D landmarks.
    - **Head Pose:** Maps 6 specific 2D facial landmarks (nose, chin, eyes, mouth edges) to a generic 3D human head model. Using OpenCV's `solvePnP`, we calculate pitch, yaw, and roll to know exactly where the head is pointing.
    - **Gaze Tracking:** Calculates the distance between the iris landmarks and the eye corners. A strict ratio determines if the eyes dart left or right.
- **Violations Window (`VIOLATION_WINDOW`):** A buffer of 2.0 seconds is used. A candidate must be looking away continuously for 2 seconds before a strike is issued. This prevents false positives from rapid eye blinks or swift head adjustments.
- **Identity Check:** The frame is heavily compared against the in-memory embedding generated during login using a distance tolerance of `0.6`. If the face does not match, an identity mismatch violation occurs.
