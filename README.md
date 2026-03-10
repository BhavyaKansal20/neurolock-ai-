# NeuroLock AI v2 — Complete Setup Guide

```
███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗ ██╗      ██████╗  ██████╗██╗  ██╗
████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗██║     ██╔═══██╗██╔════╝██║ ██╔╝
██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║██║     ██║   ██║██║     █████╔╝
██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║██║     ██║   ██║██║     ██╔═██╗
██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝███████╗╚██████╔╝╚██████╗██║  ██╗
                   AI v2.0 — EMOTION INTELLIGENCE SYSTEM
```

## What's New in v2

| Feature | v1 | v2 |
|---|---|---|
| Face Detector | Haar Cascade (bad) | DNN ResNet-10 SSD (accurate) |
| Emotion Model | Mini-Xception (57%) | Ensemble + TTA (70%+) |
| Real-time | HTTP polling | WebSocket stream |
| Face Recognition | ❌ | ✅ dlib HOG |
| Classroom Mode | ❌ | ✅ Multi-student, 3-phase |
| Reports | ❌ | ✅ Per-student PDF-ready |
| Database | ❌ | ✅ SQLite |
| Grad-CAM | ❌ | ✅ Heatmap overlay |
| Docker | ❌ | ✅ |
| Camera Sources | Webcam only | Webcam, USB, IP/RTSP, Phone |

---

## Quick Start (Local)

### 1. Install dependencies

```bash
# macOS (M-series)
pip install tensorflow-macos tensorflow-metal
pip install -r requirements.txt

# Linux / Windows
pip install -r requirements.txt
```

> **Note:** `face-recognition` requires cmake + dlib. If it fails:
> ```bash
> brew install cmake   # macOS
> pip install dlib face-recognition
> ```

### 2. Train models (skip if you have existing models)

```bash
python train.py --dataset archive --model both
```

This trains both Mini-Xception AND MobileNetV2. Takes ~30-60 min.
Outputs: `exports/neurolock_model.h5` and `exports/mobilenet_model.h5`

### 3. Configure

```bash
cp .env.example .env
# Edit .env if needed (port, model paths, etc.)
```

### 4. Run server

```bash
python server.py --port 5001    # use 5001 if AirPlay uses 5000 on macOS
```

Open: **http://localhost:5001**

---

## Quick Start (Docker)

```bash
# Copy trained models first
# Then:
docker-compose up --build

# Open: http://localhost:5000
```

---

## Camera Sources

| Source | How to use |
|---|---|
| Built-in Webcam | Select "Webcam" in UI |
| External USB Camera | Select "External Camera" |
| Phone (DroidCam) | Install DroidCam app → enter IP URL |
| IP Camera | Enter RTSP URL: `rtsp://user:pass@IP:554/stream` |
| WiFi Camera | HTTP stream: `http://192.168.1.x:8080/video` |
| Bluetooth Camera | Appears as USB device once paired |

---

## Classroom Mode — Step by Step

1. **Register students** (Students tab → Register Student)
   - Upload clear face photo, enter name + roll number

2. **Start session** (Classroom tab → New Session)
   - Enter teacher name, subject, room

3. **Run phases:**
   - Click **Before** → Capture 2-3 min baseline
   - Click **During** → Run throughout the lesson
   - Click **After** → Capture 2-3 min post-lesson

4. **End session** → Auto-generates report

5. **View report** (Analytics tab) with:
   - Per-student engagement % 
   - Comprehension score
   - Trend: improved / stable / declined
   - Recommendations

---

## API Reference

### WebSocket Events

| Event | Direction | Payload |
|---|---|---|
| `frame` | Client → Server | `{image: base64, gradcam: bool}` |
| `detection_result` | Server → Client | `{faces: [...], face_count: N}` |
| `set_phase` | Client → Server | `{phase: "before"/"during"/"after"}` |
| `phase_changed` | Server → Client | `{phase: string}` |

### REST Endpoints

```
GET  /api/status              # Server health
GET  /api/students            # List all students
POST /api/students            # Register student {id, name, class_name, roll_no, image}
DEL  /api/students/:id        # Remove student
GET  /api/sessions            # List sessions
POST /api/sessions            # Create session {name, teacher, subject, location}
POST /api/sessions/end        # End active session + generate report
GET  /api/sessions/:id/report # Get session report
GET  /api/cameras             # Detect available cameras
POST /api/predict             # REST fallback {image: base64, gradcam: bool}
```

---

## Model Architecture

### Ensemble Pipeline

```
Input Frame (any resolution)
      ↓
DNN Face Detector (ResNet-10 SSD)
      ↓ (per face)
┌─────────────────────────────────────┐
│  Mini-Xception  (48×48)             │
│  MobileNetV2    (96×96, ImageNet)   │
│  + TTA (5 augmentations each)       │
│  → Average predictions              │
└─────────────────────────────────────┘
      ↓
Face Recognition (dlib HOG 128-d)
      ↓
Classroom Logger → SQLite DB
      ↓
WebSocket → Frontend
```

---

## Scenarios Beyond Classroom

This system works for any scenario with people and cameras:

- **Retail Analytics** — Customer emotion while browsing products
- **HR / Interviews** — Candidate stress/engagement analysis  
- **Mental Health** — Real-time mood tracking with consent
- **Security** — Stress/threat detection at access points
- **Content Testing** — Viewer reaction to ads/videos
- **Hospital** — Patient anxiety monitoring
- **Online Exams** — Detect confusion/stress in students

---

## Government / Commercial Use

For production deployment:
1. Enable HTTPS (use nginx reverse proxy)
2. Replace SQLite with PostgreSQL
3. Add user authentication (Flask-JWT)
4. Add consent/privacy notices per jurisdiction
5. Use GPU server for inference (10x faster)
6. Enable RTSP streams for existing CCTV infrastructure

---

## Troubleshooting

**face-recognition install fails**
```bash
brew install cmake boost boost-python3    # macOS
sudo apt install cmake libboost-all-dev   # Ubuntu
pip install dlib face-recognition
```

**Port 5000 blocked (macOS AirPlay)**
```bash
python server.py --port 5001
```

**Low accuracy on dark/side faces**
- Use DNN detector (default) — handles angles better than Haar
- Ensure good lighting
- Lower `CONFIDENCE_THRESHOLD` to 0.35 in `.env`

**No GPU detected**
```bash
# macOS M-series
pip install tensorflow-macos tensorflow-metal
# NVIDIA
pip install tensorflow[and-cuda]
```
# neurolock-ai-
# neurolock-ai-
