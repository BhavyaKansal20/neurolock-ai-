"""
NeuroLock AI v2 — Ensemble Predictor + Test Time Augmentation
==============================================================
FIXES:
1. _load_model_safe() function add kiya — .keras aur .h5 dono handle karta hai
2. Legacy .h5 ke liye safe_mode=False use karta hai (Lambda layer wale purane models)
3. .keras format prefer karta hai (new training ke baad)
"""

import numpy as np
import cv2
import os
import tensorflow as tf
from model.architecture import EMOTION_LABELS, INPUT_SHAPE
from model.mobilenet_model import INPUT_SHAPE_MOBILENET


class EnsemblePredictor:
    """
    Combines two models with TTA for best accuracy.
    Falls back to single model if only one is available.
    """

    def __init__(self, xception_path: str = None, mobilenet_path: str = None,
                 use_tta: bool = True, tta_steps: int = 5):
        self.models = []
        self.model_names = []
        self.use_tta = use_tta
        self.tta_steps = tta_steps
        self.emotion_labels = EMOTION_LABELS

        if xception_path and os.path.exists(xception_path):
            m = self._load_model_safe(xception_path)
            self.models.append(('xception', m, INPUT_SHAPE))
            print(f"  ✓ Loaded Mini-Xception: {xception_path}")

        if mobilenet_path and os.path.exists(mobilenet_path):
            m = self._load_model_safe(mobilenet_path)
            self.models.append(('mobilenet', m, INPUT_SHAPE_MOBILENET))
            print(f"  ✓ Loaded MobileNetV2:   {mobilenet_path}")

        if not self.models:
            raise ValueError("No models found! Train first: python train.py --dataset archive")

        print(f"  Ensemble: {len(self.models)} model(s), TTA={'on' if use_tta else 'off'}")

        # Warm up — ek baar dummy predict karo taaki GPU/CPU ready ho
        dummy_bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        self.predict(dummy_bgr)

    def _load_model_safe(self, model_path: str):
        """
        Model load karo safely — dono formats handle karta hai.
        
        .keras file: seedha load — no issues
        .h5 file (purana format): pehle normal try karo, agar Lambda layer error
                                   aaye toh safe_mode=False se load karo
        safe_mode=False ka matlab: Lambda layers bhi load hongi (trusted source)
        """
        # Pehle .keras version dhundo — agar exist karta hai toh prefer karo
        keras_path = model_path.replace('.h5', '.keras')
        if os.path.exists(keras_path):
            print(f"    Loading .keras format: {keras_path}")
            return tf.keras.models.load_model(keras_path)

        # .h5 file — safe load try karo pehle
        try:
            return tf.keras.models.load_model(model_path)
        except ValueError as e:
            if 'Lambda' in str(e) or 'safe_mode' in str(e):
                # Lambda layer wala purana model hai — safe_mode=False se load karo
                print(f"    Legacy Lambda model detected — loading with safe_mode=False")
                tf.keras.config.enable_unsafe_deserialization()
                model = tf.keras.models.load_model(model_path)
                tf.keras.config.disable_unsafe_deserialization()
                return model
            raise  # Koi aur error hai toh upar throw karo

    def _preprocess(self, face_bgr: np.ndarray, target_size: tuple) -> np.ndarray:
        # BGR image → grayscale → resize → normalize → reshape
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        h, w = target_size[0], target_size[1]
        resized = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
        norm = resized.astype(np.float32) / 255.0
        return norm.reshape(1, h, w, 1)

    def _tta_augment(self, face_bgr: np.ndarray, target_size: tuple) -> list:
        """TTA variants: original + flip + brightness changes + rotation."""
        variants = []

        # Original face
        variants.append(self._preprocess(face_bgr, target_size))

        # Horizontally flipped — mirror image
        flipped = cv2.flip(face_bgr, 1)
        variants.append(self._preprocess(flipped, target_size))

        # Thoda bright karo — +10% brightness
        bright = cv2.convertScaleAbs(face_bgr, alpha=1.1, beta=10)
        variants.append(self._preprocess(bright, target_size))

        # Thoda dark karo — -10% brightness
        dark = cv2.convertScaleAbs(face_bgr, alpha=0.9, beta=-10)
        variants.append(self._preprocess(dark, target_size))

        # 5 degree rotate — slight angle variation
        M = cv2.getRotationMatrix2D((face_bgr.shape[1]//2, face_bgr.shape[0]//2), 5, 1)
        rotated = cv2.warpAffine(face_bgr, M, (face_bgr.shape[1], face_bgr.shape[0]))
        variants.append(self._preprocess(rotated, target_size))

        return variants[:self.tta_steps]

    def predict(self, face_bgr: np.ndarray) -> dict:
        """
        Ensemble prediction face crop pe (BGR numpy array).
        Returns dict: emotions, dominant, confidence, model_count
        """
        all_probs = []

        for name, model, shape in self.models:
            if self.use_tta:
                # TTA: 5 variants predict karo, average nikalo
                variants = self._tta_augment(face_bgr, shape)
                probs_list = [model.predict(v, verbose=0)[0] for v in variants]
                avg_probs = np.mean(probs_list, axis=0)
            else:
                # Seedha predict
                tensor = self._preprocess(face_bgr, shape)
                avg_probs = model.predict(tensor, verbose=0)[0]

            all_probs.append(avg_probs)

        # Dono models ka average (equal weight ensemble)
        final_probs = np.mean(all_probs, axis=0)

        emotions = {label: float(round(prob, 4))
                    for label, prob in zip(self.emotion_labels, final_probs)}
        dominant = max(emotions, key=emotions.get)

        return {
            'emotions':    emotions,
            'dominant':    dominant,
            'confidence':  round(float(emotions[dominant]), 4),
            'model_count': len(self.models),
        }
