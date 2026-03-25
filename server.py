from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import cv2
import face_recognition
import os
import logging
import sys
import time
import math
from ultralytics import YOLO
import mediapipe as mp
from collections import deque

# ──────────────────────────────────────────────────
# Configuration & Logging
# ──────────────────────────────────────────────────
def setup_server_logging():
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logger = logging.getLogger('proctoring_server')
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler('logs/server.log')
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger = setup_server_logging()
app = Flask(__name__)
CORS(app)

# ──────────────────────────────────────────────────
# AI Models
# ──────────────────────────────────────────────────
try:
    yolo_model = YOLO("yolov8n.pt")
    logger.info("YOLOv8 model loaded")
except Exception as e:
    logger.error(f"YOLOv8 load failed: {e}")
    yolo_model = None

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ──────────────────────────────────────────────────
# Global State
# ──────────────────────────────────────────────────
user_embeddings = {}          # user_id -> list of np arrays (multi-shot)
sessions = {}
VIOLATION_WINDOW = 2.0        # seconds of continuous violation before a strike
IDENTITY_CHECK_INTERVAL = 3   # run identity check every N frames
FACE_MATCH_THRESHOLD = 0.45   # lower = stricter (default was 0.6)
MAX_STRIKES = 15

# YOLO class IDs for prohibited objects
# 0=person, 67=cell phone, 73=book, 63=laptop, 64=mouse, 65=remote, 66=keyboard
PROHIBITED_OBJECTS = {67: "Cell Phone", 73: "Book", 63: "Laptop"}

# 3D model points for solvePnP head pose
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype="double")


def _euler_from_rotation(R):
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        roll  = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = math.atan2(R[1, 0], R[0, 0])
    else:
        roll  = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = 0
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


def get_session(session_id):
    if session_id not in sessions:
        sessions[session_id] = {
            "strikes": 0,
            "start_time": time.time(),
            "timers": {
                "gaze": None,
                "head": None,
                "multiple_people": None,
                "no_face": None,
                "identity": None,
                "prohibited_object": None,
            },
            "gaze_history": deque(maxlen=5),
            "frame_count": 0,
            "last_identity_result": True,   # cached: True = verified
            "last_face_distance": 0.0,      # cached distance for UI
        }
    return sessions[session_id]


# ──────────────────────────────────────────────────
# Routes – Static Pages
# ──────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('.', 'login.html')

@app.route('/login.html')
def login_page():
    return send_from_directory('.', 'login.html')

@app.route('/webcam.html')
def webcam_page():
    return send_from_directory('.', 'webcam.html')

@app.route('/thankyou.html')
def thankyou_page():
    return send_from_directory('.', 'thankyou.html')

@app.route('/unauthorized.html')
def unauthorized_page():
    return send_from_directory('.', 'unauthorized.html')


# ──────────────────────────────────────────────────
# Enroll – capture face embedding(s) in-memory
# Supports multi-shot: call /enroll multiple times
# to add more reference embeddings for better accuracy.
# ──────────────────────────────────────────────────
@app.route('/enroll', methods=['POST'])
def enroll():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image uploaded'}), 400
    try:
        file = request.files['image']
        user_id = request.form.get('user_id', 'Candidate')
        append = request.form.get('append', 'false') == 'true'

        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb, model='hog')
        if not locs:
            return jsonify({'success': False,
                            'error': 'No face detected. Please look directly at the camera.'}), 400
        if len(locs) > 1:
            return jsonify({'success': False,
                            'error': 'Multiple faces detected. Only one person should be visible.'}), 400

        encs = face_recognition.face_encodings(rgb, locs, num_jitters=3)
        if not encs:
            return jsonify({'success': False, 'error': 'Could not encode face'}), 400

        if append and user_id in user_embeddings:
            user_embeddings[user_id].append(encs[0])
            count = len(user_embeddings[user_id])
            logger.info(f"Added enrollment #{count} for user: {user_id}")
        else:
            user_embeddings[user_id] = [encs[0]]
            # Reset sessions on fresh enrollment
            keys_to_remove = [k for k in sessions if k.startswith(user_id)]
            for k in keys_to_remove:
                del sessions[k]
            if 'default_session' in sessions:
                del sessions['default_session']
            logger.info(f"Enrolled user: {user_id} (fresh, sessions reset)")

        return jsonify({
            'success': True,
            'enrollment_count': len(user_embeddings[user_id])
        })
    except Exception as e:
        logger.error(f"Enroll error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ──────────────────────────────────────────────────
# Analyse – the core proctoring endpoint
# ──────────────────────────────────────────────────
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    session_id = request.form.get('session_id', 'default_session')
    user_id    = request.form.get('user_id', 'Candidate')
    s = get_session(session_id)
    now = time.time()
    s["frame_count"] += 1
    frame_num = s["frame_count"]

    try:
        # Grace period: skip violations for first 5 seconds of session
        grace_period = (now - s["start_time"]) < 5.0

        npimg = np.frombuffer(request.files['image'].read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        h, w  = frame.shape[:2]

        # ═══ 1. YOLO DETECTION (persons + prohibited objects) ═══
        person_count = 0
        detected_objects = []
        if yolo_model:
            # Detect persons (0) + cell phone (67) + book (73) + laptop (63)
            detect_classes = [0] + list(PROHIBITED_OBJECTS.keys())
            res = yolo_model(frame, imgsz=640, conf=0.5, classes=detect_classes, verbose=False)
            for box in res[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if cls_id == 0:
                    person_count += 1
                elif cls_id in PROHIBITED_OBJECTS and conf >= 0.45:
                    detected_objects.append(PROHIBITED_OBJECTS[cls_id])

        # ═══ 2. FACE MESH ═══
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_res = face_mesh.process(rgb)

        status     = "Normal"
        violations = []
        is_verified = s["last_identity_result"]  # use cached by default
        identity_mismatch = False
        face_distance_val = s["last_face_distance"]
        yaw_val    = 0.0
        pitch_val  = 0.0
        gaze_val   = 0.5

        if not mp_res.multi_face_landmarks:
            status = "No Face Detected"
            # Timed violation for no face
            if s["timers"]["no_face"] is None:
                s["timers"]["no_face"] = now
            elif now - s["timers"]["no_face"] >= VIOLATION_WINDOW:
                violations.append("no_face")
                s["timers"]["no_face"] = None
        else:
            s["timers"]["no_face"] = None
            lm = mp_res.multi_face_landmarks[0].landmark

            # ═══ 3. HEAD POSE (solvePnP + Euler) ═══
            focal = float(w)
            cam = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]], dtype="double")
            pts = np.array([
                (lm[1].x*w,   lm[1].y*h),
                (lm[152].x*w, lm[152].y*h),
                (lm[33].x*w,  lm[33].y*h),
                (lm[263].x*w, lm[263].y*h),
                (lm[61].x*w,  lm[61].y*h),
                (lm[291].x*w, lm[291].y*h),
            ], dtype="double")
            ok, rvec, tvec = cv2.solvePnP(
                MODEL_POINTS, pts, cam, np.zeros((4, 1)),
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if ok:
                rmat, _ = cv2.Rodrigues(rvec)
                pitch_val, yaw_val, _ = _euler_from_rotation(rmat)

            # ═══ 4. GAZE ═══
            nose    = lm[1]
            l_cheek = lm[234]
            r_cheek = lm[454]
            nose_x  = nose.x * w
            left_x  = l_cheek.x * w
            right_x = r_cheek.x * w
            left_d  = nose_x - left_x
            right_d = right_x - nose_x
            lm_yaw  = 0
            if (left_d + right_d) != 0:
                lm_yaw = (right_d - left_d) / (left_d + right_d) * 45

            # Pitch from landmark ratio
            top_y    = lm[10].y
            bottom_y = lm[152].y
            nose_y   = lm[1].y
            face_h   = bottom_y - top_y
            pitch_ratio = (nose_y - top_y) / face_h if face_h > 0 else 0.5

            # Iris-based gaze ratio
            try:
                rh_outer = lm[33].x;  rh_inner = lm[133].x; rh_iris = lm[468].x
                lh_inner = lm[362].x; lh_outer = lm[263].x; lh_iris = lm[473].x
                rh_w = rh_inner - rh_outer
                lh_w = lh_outer - lh_inner
                gaze_val = 0.5
                if rh_w != 0 and lh_w != 0:
                    rh_gaze = (rh_iris - rh_outer) / rh_w
                    lh_gaze = (lh_iris - lh_inner) / lh_w
                    gaze_val = (rh_gaze + lh_gaze) / 2.0
            except:
                gaze_val = 0.5

            # ═══ 5. DEVIATION LOGIC ═══
            head_violation = False
            gaze_violation = False

            if lm_yaw < -15:
                status = "Head Turned Left"; head_violation = True
            elif lm_yaw > 15:
                status = "Head Turned Right"; head_violation = True
            elif pitch_ratio < 0.38:
                status = "Head Tilted Up"; head_violation = True
            elif pitch_ratio > 0.62:
                status = "Head Tilted Down"; head_violation = True
            elif gaze_val < 0.38:
                status = "Eyes Looking Right"; gaze_violation = True
            elif gaze_val > 0.62:
                status = "Eyes Looking Left"; gaze_violation = True

            # Timed head violation
            if head_violation:
                if s["timers"]["head"] is None:
                    s["timers"]["head"] = now
                elif now - s["timers"]["head"] >= VIOLATION_WINDOW:
                    violations.append("head_deviation")
                    s["timers"]["head"] = None
            else:
                s["timers"]["head"] = None

            # Timed gaze violation
            if gaze_violation:
                if s["timers"]["gaze"] is None:
                    s["timers"]["gaze"] = now
                elif now - s["timers"]["gaze"] >= VIOLATION_WINDOW:
                    violations.append("gaze_deviation")
                    s["timers"]["gaze"] = None
            else:
                s["timers"]["gaze"] = None

        # ═══ 6. IDENTITY CHECK (every Nth frame, cached) ═══
        if user_id in user_embeddings and (frame_num % IDENTITY_CHECK_INTERVAL == 0):
            face_locs = face_recognition.face_locations(rgb, model='hog')
            if face_locs:
                enc = face_recognition.face_encodings(rgb, face_locs)
                if enc:
                    # Compare against ALL enrolled embeddings
                    enrolled = user_embeddings[user_id]
                    distances = face_recognition.face_distance(enrolled, enc[0])
                    min_distance = float(np.min(distances))
                    match = min_distance <= FACE_MATCH_THRESHOLD

                    s["last_identity_result"] = match
                    s["last_face_distance"] = min_distance
                    face_distance_val = min_distance

                    logger.info(
                        f"[Identity] user={user_id} frame={frame_num} "
                        f"dist={min_distance:.4f} threshold={FACE_MATCH_THRESHOLD} "
                        f"match={match} (checked against {len(enrolled)} embedding(s))"
                    )

                    if not match:
                        identity_mismatch = True
                        is_verified = False
                        status = "Unknown Person"
                else:
                    # Could not encode face in frame
                    s["last_identity_result"] = True
            else:
                # No face found by face_recognition (use cached)
                pass
        else:
            # Use cached identity result
            if not s["last_identity_result"]:
                identity_mismatch = True
                is_verified = False
                if status == "Normal":
                    status = "Unknown Person"

        # Timed identity violation
        if identity_mismatch:
            if s["timers"]["identity"] is None:
                s["timers"]["identity"] = now
            elif now - s["timers"]["identity"] >= VIOLATION_WINDOW:
                violations.append("identity_mismatch")
                s["timers"]["identity"] = None
        else:
            s["timers"]["identity"] = None

        # ═══ 7. MULTIPLE PEOPLE ═══
        if person_count > 1:
            status = "Multiple People Detected"
            if s["timers"]["multiple_people"] is None:
                s["timers"]["multiple_people"] = now
            elif now - s["timers"]["multiple_people"] >= VIOLATION_WINDOW:
                violations.append("multiple_people")
                s["timers"]["multiple_people"] = None
        else:
            s["timers"]["multiple_people"] = None

        # ═══ 8. PROHIBITED OBJECTS ═══
        if detected_objects:
            obj_str = ", ".join(sorted(set(detected_objects)))
            status = f"Prohibited: {obj_str}"
            if s["timers"]["prohibited_object"] is None:
                s["timers"]["prohibited_object"] = now
            elif now - s["timers"]["prohibited_object"] >= VIOLATION_WINDOW:
                violations.append("prohibited_object")
                s["timers"]["prohibited_object"] = None
        else:
            s["timers"]["prohibited_object"] = None

        # Update strikes (skip during grace period)
        if violations and not grace_period:
            s["strikes"] += len(violations)
        elif grace_period:
            violations = []
            status = "Normal"

        terminate = s["strikes"] >= MAX_STRIKES

        return jsonify({
            'status': status,
            'violations': violations,
            'strikes': s["strikes"],
            'max_strikes': MAX_STRIKES,
            'person_count': person_count,
            'is_verified': is_verified,
            'face_distance': round(face_distance_val, 4),
            'detected_objects': list(set(detected_objects)),
            'terminate': terminate,
        })
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    logger.info("Starting Interview Proctoring Server on port 5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)