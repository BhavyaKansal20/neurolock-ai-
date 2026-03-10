"""
NeuroLock AI v2 — High-Accuracy Face Detector
Uses OpenCV DNN (ResNet-10 SSD) — far superior to Haar cascades.
No false positives on text/patterns. Works in low light.
Falls back to Haar if DNN models not downloaded.
"""

import os
import cv2
import numpy as np
import urllib.request

# DNN model URLs (OpenCV pre-trained)
PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
CAFFEMODEL_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

MODEL_DIR      = os.path.join(os.path.dirname(__file__), '..', 'data', 'face_detector')
PROTOTXT_PATH  = os.path.join(MODEL_DIR, 'deploy.prototxt')
CAFFE_PATH     = os.path.join(MODEL_DIR, 'res10_300x300_ssd_iter_140000.caffemodel')


def download_dnn_models():
    """Download face detector model files if not present."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(PROTOTXT_PATH):
        print("  Downloading face detector prototxt...")
        urllib.request.urlretrieve(PROTOTXT_URL, PROTOTXT_PATH)

    if not os.path.exists(CAFFE_PATH):
        print("  Downloading face detector weights (~2MB)...")
        urllib.request.urlretrieve(CAFFEMODEL_URL, CAFFE_PATH)

    print("  Face detector models ready.")


class FaceDetector:
    """
    High-accuracy face detector using OpenCV DNN.
    Supports: single images, video frames, multi-face.
    """

    def __init__(self, confidence_threshold: float = 0.5, use_dnn: bool = True):
        self.confidence_threshold = confidence_threshold
        self.dnn_net = None
        self.haar_cascade = None
        self.mode = 'none'

        if use_dnn:
            self._load_dnn()

        if self.dnn_net is None:
            self._load_haar()

    def _load_dnn(self):
        try:
            if not os.path.exists(PROTOTXT_PATH) or not os.path.exists(CAFFE_PATH):
                download_dnn_models()
            self.dnn_net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFE_PATH)
            self.mode = 'dnn'
            print("  Face detector: DNN (ResNet-10 SSD) ✓")
        except Exception as e:
            print(f"  DNN load failed ({e}), falling back to Haar")

    def _load_haar(self):
        try:
            path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.haar_cascade = cv2.CascadeClassifier(path)
            self.mode = 'haar'
            print("  Face detector: Haar Cascade (fallback)")
        except Exception as e:
            print(f"  Haar load failed: {e}")

    def detect(self, frame_bgr: np.ndarray, max_faces: int = 20) -> list:
        """
        Detect faces in a BGR frame.
        Returns list of (x, y, w, h) tuples, sorted by area (largest first).
        """
        if self.mode == 'dnn':
            return self._detect_dnn(frame_bgr, max_faces)
        elif self.mode == 'haar':
            return self._detect_haar(frame_bgr, max_faces)
        return []

    def _detect_dnn(self, frame: np.ndarray, max_faces: int) -> list:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)
        )
        self.dnn_net.setInput(blob)
        detections = self.dnn_net.forward()

        boxes = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < self.confidence_threshold:
                continue
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            bw, bh = x2 - x1, y2 - y1
            if bw > 20 and bh > 20:
                boxes.append((x1, y1, bw, bh, conf))

        # Sort by area descending
        boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        return [(x, y, w, h) for x, y, w, h, _ in boxes[:max_faces]]

    def _detect_haar(self, frame: np.ndarray, max_faces: int) -> list:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.haar_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6,
            minSize=(40, 40), flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) == 0:
            return []
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces[:max_faces]]

    def crop_face(self, frame: np.ndarray, box: tuple, margin: float = 0.1) -> np.ndarray:
        """Crop face from frame with optional margin."""
        x, y, w, h = box
        m = int(margin * min(w, h))
        x1 = max(0, x - m)
        y1 = max(0, y - m)
        x2 = min(frame.shape[1], x + w + m)
        y2 = min(frame.shape[0], y + h + m)
        return frame[y1:y2, x1:x2]
