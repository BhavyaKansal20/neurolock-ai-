
<div align="center">

```
███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗ ██╗      ██████╗  ██████╗██╗  ██╗
████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗██║     ██╔═══██╗██╔════╝██║ ██╔╝
██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║██║     ██║   ██║██║     █████╔╝ 
██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║██║     ██║   ██║██║     ██╔═██╗ 
██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝███████╗╚██████╔╝╚██████╗██║  ██╗
╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝
```

### **Emotion Intelligence System — v2.0**
*Real-time facial emotion detection, student recognition, and classroom analytics*

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange?style=flat-square&logo=tensorflow)
![WebSocket](https://img.shields.io/badge/WebSocket-Flask--SocketIO-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

</div>

---

## ✨ What's New in v2

| | Feature | Detail |
|---|---|---|
| 🧠 | **MobileNetV2 Transfer Learning** | ImageNet pretrained → 68–72% accuracy (was 57%) |
| 🎯 | **Ensemble + TTA** | 2 models × 5 augmentations averaged per frame |
| 👁️ | **DNN Face Detector** | ResNet-10 SSD — no false positives on text/objects |
| 👤 | **Face Recognition** | dlib 128-d encoding — identifies registered students by name |
| 🎓 | **Classroom Mode** | 3-phase session (Before → During → After class) |
| 📊 | **Smart Reports** | Per-student engagement %, comprehension score, trend analysis |
| 🔌 | **WebSocket Streaming** | Real-time, replaces slow HTTP polling |
| 🔥 | **Grad-CAM** | Visual heatmap showing *where* the model is looking |
| 🗄️ | **SQLite Database** | All students, sessions, logs persisted automatically |
| 🐳 | **Docker Ready** | `docker-compose up` — one command deploy |
| 📷 | **Any Camera** | USB · IP · RTSP · WiFi · Phone · Bluetooth |

---

## 🚀 Quick Start

```bash
# 1. Install
pip install tensorflow-macos tensorflow-metal   # macOS M-series
pip install -r requirements.txt

# 2. Train models
python train.py --dataset archive --model both

# 3. Run
python server.py --port 5001

# → Open http://localhost:5001
```

**Docker:**
```bash
cp .env.example .env
docker-compose up --build
# → Open http://localhost:5000
```

---

## 🎓 Classroom Mode — How It Works

```
┌──────────────┐    ┌────────────────┐    ┌──────────────┐
│    BEFORE    │ →  │     DURING     │ →  │    AFTER     │
│  Baseline    │    │  Live Tracking │    │  Assessment  │
│  capture     │    │  per student   │    │  + Report    │
└──────────────┘    └────────────────┘    └──────────────┘
```

1. **Register students** — upload photo + name + roll number
2. **Start session** — enter teacher, subject, room
3. **Run phases** — click Before / During / After at the right time
4. **End session** — auto-generates full report with:
   - Engagement % per student per phase
   - Comprehension score (after vs before delta)
   - Trend: `improved` / `stable` / `declined`
   - Personalized recommendation per student
   - Class-level summary

---

## 📷 Camera Sources

| Source | Setup |
|---|---|
| Built-in webcam | Select in UI — just works |
| External USB camera | Plug in → select "External Camera" |
| Phone (DroidCam) | Install app → use IP URL |
| IP / Security Camera | `rtsp://user:pass@192.168.x.x:554/stream` |
| WiFi Camera | `http://192.168.x.x:8080/video` |

---

## 🌐 Use Cases

> This system is camera-agnostic and scenario-agnostic. Pair it with any webcam for:

```
🏫  Classroom engagement tracking      🏥  Patient anxiety monitoring
🛍️  Retail customer sentiment          🎯  HR interview analysis  
📺  Ad / content reaction testing      🔐  Access-point stress detection
💻  Online exam focus monitoring       🏋️  Athletic performance states
```

---

## 🛠️ API Reference

**WebSocket events:**
```
frame              →  Send base64 frame for analysis
detection_result   ←  Receive face boxes + emotions + student name
set_phase          →  Switch session phase (before/during/after)
```

**REST endpoints:**
```
GET  /api/status             Server health + model info
GET  /api/students           All registered students
POST /api/students           Register new student
POST /api/sessions           Start classroom session
POST /api/sessions/end       End session + generate report
GET  /api/sessions/:id/report   Fetch saved report
GET  /api/cameras            Detect available cameras
```

---

## 📁 Project Structure

```
neurolock-v2/
├── server.py               WebSocket + REST server
├── train.py                Train MobileNetV2 + Mini-Xception
├── model/
│   ├── mobilenet_model.py  MobileNetV2 transfer learning
│   ├── ensemble.py         Multi-model + TTA predictor
│   └── architecture.py     Mini-Xception (fast/lightweight)
├── utils/
│   ├── face_detector.py    DNN ResNet face detector
│   ├── face_recognizer.py  dlib face recognition
│   ├── gradcam.py          Grad-CAM heatmap
│   └── database.py         SQLite layer
├── classroom/
│   └── session.py          Session manager + report engine
├── frontend/               Premium dark-tech UI
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## ⚙️ Configuration (`.env`)

```env
PORT=5000
CONFIDENCE_THRESHOLD=0.45      # Lower = detect more faces
USE_ENSEMBLE=true              # Both models averaged
FACE_DETECTOR=dnn              # dnn (recommended) | haar
FACE_RECOGNITION_TOLERANCE=0.5 # Lower = stricter matching
PROCESS_FPS=10                 # Inference FPS (camera runs at 30)
```

---

<div align="center">

**Built for education, deployable for government & commercial use.**

*No cloud. No external APIs. Runs fully on-device.*

</div>
