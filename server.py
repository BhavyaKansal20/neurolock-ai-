"""
NeuroLock AI v2 — WebSocket + REST Server
==========================================
- WebSocket for real-time frame streaming (replaces HTTP polling)
- REST API for students, sessions, reports
- Classroom mode with multi-student tracking
- Grad-CAM endpoint
- Any camera source: USB / IP / RTSP / Phone

Run: python server.py
"""

import os, sys, json, time, base64, logging, uuid
from datetime import datetime
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from model.ensemble import EnsemblePredictor
from utils.face_detector import FaceDetector
from utils.face_recognizer import FaceRecognizer
from utils.gradcam import GradCAM
from utils import database as db
from classroom.session import ClassroomSession

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger('neurolock')

# ── Config ────────────────────────────────────────────────────
HOST       = os.getenv('HOST', '0.0.0.0')
PORT       = int(os.getenv('PORT', 5000))
SECRET_KEY = os.getenv('SECRET_KEY', 'neurolock-dev-key')
MODEL_PATH = os.getenv('MODEL_PATH', 'exports/neurolock_model.h5')
MOB_PATH   = os.getenv('MOBILENET_PATH', 'exports/mobilenet_model.h5')
CONF_THRESH= float(os.getenv('CONFIDENCE_THRESHOLD', 0.45))
DB_PATH    = os.getenv('DATABASE_PATH', 'data/neurolock.db')

# ── Flask + SocketIO ──────────────────────────────────────────
app = Flask(__name__, static_folder='frontend', static_url_path='')
app.config['SECRET_KEY'] = SECRET_KEY
CORS(app, resources={r'/api/*': {'origins': '*'}})
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='eventlet',
                    max_http_buffer_size=10_000_000)

# ── Global objects ────────────────────────────────────────────
predictor:   EnsemblePredictor = None
detector:    FaceDetector      = None
recognizer:  FaceRecognizer    = None
gradcam_obj: GradCAM           = None
active_session: ClassroomSession = None

SERVER_START  = datetime.now()
FRAME_COUNT   = 0
FACE_COUNT    = 0


# ══════════════════════════════════════════════════════════════
# Startup
# ══════════════════════════════════════════════════════════════
def startup():
    global predictor, detector, recognizer, gradcam_obj

    db.init_db(DB_PATH)

    log.info("Loading face detector...")
    detector = FaceDetector(confidence_threshold=0.45, use_dnn=True)

    log.info("Loading face recognizer...")
    recognizer = FaceRecognizer(tolerance=0.5)
    log.info(f"  {recognizer.student_count} students in database")

    log.info("Loading emotion models...")
    xception_path  = MODEL_PATH if os.path.exists(MODEL_PATH) else None
    mobilenet_path = MOB_PATH   if os.path.exists(MOB_PATH)   else None

    if not xception_path and not mobilenet_path:
        log.error("No trained model found! Run: python train.py --dataset archive")
        sys.exit(1)

    predictor = EnsemblePredictor(
        xception_path=xception_path,
        mobilenet_path=mobilenet_path,
        use_tta=True
    )

    # Set up Grad-CAM on first available model
    if predictor.models:
        _, first_model, _ = predictor.models[0]
        gradcam_obj = GradCAM(first_model)
        log.info("  Grad-CAM ready")

    log.info("Server ready!")


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════
def decode_image(b64: str) -> np.ndarray:
    if b64.startswith('data:'):
        b64 = b64.split(',', 1)[1]
    img_bytes = base64.b64decode(b64)
    pil_img   = Image.open(BytesIO(img_bytes)).convert('RGB')
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def process_frame(bgr: np.ndarray, include_gradcam: bool = False,
                  session: ClassroomSession = None) -> dict:
    global FRAME_COUNT, FACE_COUNT
    FRAME_COUNT += 1

    boxes = detector.detect(bgr, max_faces=15)
    results = []

    for box in boxes:
        crop = detector.crop_face(bgr, box, margin=0.08)
        if crop.size == 0:
            continue

        pred = predictor.predict(crop)
        if pred['confidence'] < CONF_THRESH:
            continue

        FACE_COUNT += 1
        x, y, w, h = box

        # Face recognition
        student = recognizer.identify(crop)
        student_id   = student['id']   if student else f'unknown_{FACE_COUNT}'
        student_name = student['name'] if student else 'Unknown'

        face_data = {
            'box':           {'x': x, 'y': y, 'w': w, 'h': h},
            'emotions':      pred['emotions'],
            'dominant':      pred['dominant'],
            'confidence':    pred['confidence'],
            'student_id':    student_id,
            'student_name':  student_name,
            'is_registered': student is not None,
        }

        # Grad-CAM (only if requested, adds latency)
        if include_gradcam and gradcam_obj:
            try:
                from model.architecture import INPUT_SHAPE
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (INPUT_SHAPE[1], INPUT_SHAPE[0]))
                tensor = resized.astype(np.float32) / 255.0
                tensor = tensor.reshape(1, INPUT_SHAPE[0], INPUT_SHAPE[1], 1)
                hm = gradcam_obj.compute(tensor)
                overlay = gradcam_obj.overlay(
                    cv2.resize(crop, (128, 128)), hm
                )
                face_data['gradcam_b64'] = gradcam_obj.to_b64(overlay)
            except Exception as e:
                log.debug(f"Grad-CAM error: {e}")

        # Log to classroom session
        if session and session.active:
            session.log(pred, student_id, student_name)
            if student_id not in session.snapshots and len(session.snapshots) < 50:
                session.save_snapshot(student_id, crop)

        results.append(face_data)

    return {'faces': results, 'face_count': len(results)}


# ══════════════════════════════════════════════════════════════
# WebSocket Events
# ══════════════════════════════════════════════════════════════

@socketio.on('connect')
def on_connect():
    log.info(f"Client connected: {request.sid}")
    emit('server_info', {
        'status': 'connected',
        'models': len(predictor.models) if predictor else 0,
        'students': recognizer.student_count if recognizer else 0,
        'session_active': active_session is not None and (active_session.active if active_session else False),
    })


@socketio.on('disconnect')
def on_disconnect():
    log.info(f"Client disconnected: {request.sid}")


@socketio.on('frame')
def on_frame(data):
    """
    Receive a video frame, run detection, emit results.
    data: {image: base64, gradcam: bool}
    """
    try:
        bgr = decode_image(data['image'])
        include_gc = data.get('gradcam', False)
        result = process_frame(bgr, include_gradcam=include_gc,
                                session=active_session)
        emit('detection_result', result)
    except Exception as e:
        log.exception("Frame processing error")
        emit('error', {'message': str(e)})


@socketio.on('set_phase')
def on_set_phase(data):
    """Change classroom session phase: before/during/after"""
    global active_session
    if not active_session:
        emit('error', {'message': 'No active session'})
        return
    phase = data.get('phase')
    active_session.set_phase(phase)
    emit('phase_changed', {'phase': phase})
    socketio.emit('phase_changed', {'phase': phase})


@socketio.on('request_stats')
def on_request_stats():
    if active_session:
        emit('session_stats', active_session.get_live_stats())


# ══════════════════════════════════════════════════════════════
# REST API
# ══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/api/status')
def api_status():
    uptime = int((datetime.now() - SERVER_START).total_seconds())
    return jsonify({
        'status':          'online',
        'models_loaded':   predictor is not None,
        'model_count':     len(predictor.models) if predictor else 0,
        'face_detector':   detector.mode if detector else 'none',
        'students':        recognizer.student_count if recognizer else 0,
        'session_active':  active_session is not None and (active_session.active if active_session else False),
        'uptime_s':        uptime,
        'frames_processed':FRAME_COUNT,
        'faces_analyzed':  FACE_COUNT,
    })


# ── Students ──────────────────────────────────────────────────

@app.route('/api/students', methods=['GET'])
def api_get_students():
    students = db.get_students(DB_PATH)
    for s in students:
        s['image_b64'] = recognizer.face_image_b64(s['id']) or ''
    return jsonify(students)


@app.route('/api/students', methods=['POST'])
def api_register_student():
    data = request.get_json() or {}
    image_b64   = data.get('image')
    student_id  = data.get('id') or str(uuid.uuid4())[:8]
    name        = data.get('name', 'Student')
    class_name  = data.get('class_name', '')
    roll_no     = data.get('roll_no', '')

    if not image_b64:
        return jsonify({'error': 'Image required'}), 400

    bgr = decode_image(image_b64)
    boxes = detector.detect(bgr, max_faces=1)

    if not boxes:
        return jsonify({'error': 'No face detected in image'}), 422

    crop = detector.crop_face(bgr, boxes[0], margin=0.1)
    success = recognizer.register_student(
        crop, student_id, name,
        extra={'class_name': class_name, 'roll_no': roll_no}
    )

    if not success:
        return jsonify({'error': 'Could not extract face encoding'}), 422

    db.add_student(student_id, name, class_name, roll_no,
                   os.path.join('data/faces', f'{student_id}.jpg'), DB_PATH)

    return jsonify({'success': True, 'id': student_id, 'name': name})


@app.route('/api/students/<sid>', methods=['DELETE'])
def api_delete_student(sid):
    recognizer.remove_student(sid)
    db.delete_student(sid, DB_PATH)
    return jsonify({'success': True})


# ── Sessions ──────────────────────────────────────────────────

@app.route('/api/sessions', methods=['GET'])
def api_get_sessions():
    return jsonify(db.get_sessions(DB_PATH))


@app.route('/api/sessions', methods=['POST'])
def api_create_session():
    global active_session
    data = request.get_json() or {}

    if active_session and active_session.active:
        return jsonify({'error': 'A session is already active. End it first.'}), 409

    active_session = ClassroomSession(
        name     = data.get('name', f'Session {datetime.now().strftime("%H:%M")}'),
        teacher  = data.get('teacher', ''),
        subject  = data.get('subject', ''),
        location = data.get('location', ''),
    )

    socketio.emit('session_started', {
        'session_id': active_session.session_id,
        'name':       active_session.name,
        'phase':      'before',
    })

    return jsonify({'success': True, 'session_id': active_session.session_id})


@app.route('/api/sessions/end', methods=['POST'])
def api_end_session():
    global active_session
    if not active_session:
        return jsonify({'error': 'No active session'}), 404

    report = active_session.end()
    sid = active_session.session_id
    active_session = None

    socketio.emit('session_ended', {'session_id': sid, 'report': report})
    return jsonify({'success': True, 'session_id': sid, 'report': report})


@app.route('/api/sessions/<sid>/report')
def api_get_report(sid):
    report = db.get_report(sid, db_path=DB_PATH)
    if not report:
        return jsonify({'error': 'Report not found'}), 404
    return jsonify(report)


@app.route('/api/sessions/<sid>/logs')
def api_get_logs(sid):
    student_id = request.args.get('student_id')
    phase      = request.args.get('phase')
    logs = db.get_logs(sid, student_id=student_id, phase=phase, db_path=DB_PATH)
    return jsonify(logs)


# ── Camera sources ────────────────────────────────────────────

@app.route('/api/cameras')
def api_cameras():
    """Detect available camera indices."""
    available = []
    for i in range(6):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append({'index': i, 'label': f'Camera {i}'})
            cap.release()
    return jsonify({'cameras': available})


# ── Predict (REST fallback) ───────────────────────────────────

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json() or {}
    if not data.get('image'):
        return jsonify({'error': 'No image provided'}), 400
    bgr = decode_image(data['image'])
    result = process_frame(bgr, include_gradcam=data.get('gradcam', False))
    return jsonify(result)


# ══════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--host',  default=HOST)
    p.add_argument('--port',  type=int, default=PORT)
    p.add_argument('--debug', action='store_true')
    args = p.parse_args()

    startup()

    print(f"""
╔══════════════════════════════════════════════════════════╗
║          NeuroLock AI v2 — Server Running                ║
╠══════════════════════════════════════════════════════════╣
║  Frontend   →  http://localhost:{args.port:<27}║
║  API Status →  http://localhost:{args.port}/api/status    ║
║  WebSocket  →  ws://localhost:{args.port}                 ║
╚══════════════════════════════════════════════════════════╝
""")

    socketio.run(app, host=args.host, port=args.port,
                 debug=args.debug, use_reloader=False)
