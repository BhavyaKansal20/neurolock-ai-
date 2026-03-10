"""
NeuroLock AI v2 — Face Recognizer
Identifies students by face. Register once, recognize forever.
Uses face_recognition library (dlib HOG + ResNet model).
"""

import os
import json
import pickle
import base64
import numpy as np
import cv2
from datetime import datetime
from typing import Optional

try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except ImportError:
    FACE_REC_AVAILABLE = False
    print("  WARNING: face_recognition not installed. Run: pip install face-recognition")
    print("  Student identification will be disabled.")


FACES_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data', 'faces')
ENCODINGS_DB = os.path.join(FACES_DIR, 'encodings.pkl')
METADATA_DB  = os.path.join(FACES_DIR, 'metadata.json')


class FaceRecognizer:
    """
    Manages student face database and performs recognition.
    """

    def __init__(self, faces_dir: str = FACES_DIR, tolerance: float = 0.5):
        self.faces_dir   = faces_dir
        self.tolerance   = tolerance
        self.encodings   = []   # list of 128-d vectors
        self.student_ids = []   # parallel list of student IDs
        self.metadata    = {}   # student_id → {name, class, image_path, ...}
        self.available   = FACE_REC_AVAILABLE

        os.makedirs(faces_dir, exist_ok=True)
        self._load_db()

    def _load_db(self):
        if os.path.exists(ENCODINGS_DB):
            with open(ENCODINGS_DB, 'rb') as f:
                data = pickle.load(f)
                self.encodings   = data.get('encodings', [])
                self.student_ids = data.get('ids', [])
        if os.path.exists(METADATA_DB):
            with open(METADATA_DB, 'r') as f:
                self.metadata = json.load(f)

    def _save_db(self):
        with open(ENCODINGS_DB, 'wb') as f:
            pickle.dump({'encodings': self.encodings, 'ids': self.student_ids}, f)
        with open(METADATA_DB, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def register_student(self, face_bgr: np.ndarray, student_id: str,
                          name: str, extra: dict = None) -> bool:
        """
        Register a student with their face image.
        Returns True on success.
        """
        if not self.available:
            return False

        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(rgb)

        if not encs:
            return False

        encoding = encs[0]

        # Remove existing if already registered
        if student_id in self.student_ids:
            idx = self.student_ids.index(student_id)
            self.encodings.pop(idx)
            self.student_ids.pop(idx)

        self.encodings.append(encoding)
        self.student_ids.append(student_id)

        # Save face image
        img_path = os.path.join(self.faces_dir, f'{student_id}.jpg')
        cv2.imwrite(img_path, face_bgr)

        self.metadata[student_id] = {
            'id':         student_id,
            'name':       name,
            'image_path': img_path,
            'registered': datetime.now().isoformat(),
            **(extra or {})
        }
        self._save_db()
        return True

    def identify(self, face_bgr: np.ndarray) -> Optional[dict]:
        """
        Identify a face from the database.
        Returns student metadata dict or None if unknown.
        """
        if not self.available or not self.encodings:
            return None

        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(rgb)
        if not encs:
            return None

        enc = encs[0]
        distances = face_recognition.face_distance(self.encodings, enc)

        if len(distances) == 0:
            return None

        best_idx = int(np.argmin(distances))
        best_dist = float(distances[best_idx])

        if best_dist > self.tolerance:
            return None

        sid = self.student_ids[best_idx]
        meta = self.metadata.get(sid, {}).copy()
        meta['match_confidence'] = round(1 - best_dist, 3)
        return meta

    def get_all_students(self) -> list:
        return list(self.metadata.values())

    def remove_student(self, student_id: str) -> bool:
        if student_id not in self.student_ids:
            return False
        idx = self.student_ids.index(student_id)
        self.encodings.pop(idx)
        self.student_ids.pop(idx)
        self.metadata.pop(student_id, None)
        self._save_db()
        return True

    def face_image_b64(self, student_id: str) -> Optional[str]:
        """Return base64 encoded face image for frontend."""
        meta = self.metadata.get(student_id)
        if not meta:
            return None
        img_path = meta.get('image_path')
        if img_path and os.path.exists(img_path):
            with open(img_path, 'rb') as f:
                return base64.b64encode(f.read()).decode()
        return None

    @property
    def student_count(self) -> int:
        return len(self.metadata)
