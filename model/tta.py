"""
NeuroLock AI v2 — Test Time Augmentation (TTA)
================================================
At inference time, run N augmented versions of the face crop
and average the predictions. Free +2-3% accuracy boost.
"""

import cv2
import numpy as np


def _augment_face(face_gray: np.ndarray) -> list:
    """
    Generate augmented versions of a 48x48 grayscale face.
    Returns list of (1, 48, 48, 1) float32 arrays.
    """
    H, W = face_gray.shape
    variants = []

    # Original
    variants.append(face_gray)

    # Horizontal flip
    variants.append(cv2.flip(face_gray, 1))

    # Slight brightness boost
    bright = np.clip(face_gray * 1.15, 0, 1.0).astype(np.float32)
    variants.append(bright)

    # Slight brightness drop
    dark = np.clip(face_gray * 0.85, 0, 1.0).astype(np.float32)
    variants.append(dark)

    # Small rotation +5°
    M = cv2.getRotationMatrix2D((W / 2, H / 2), 5, 1.0)
    rot_pos = cv2.warpAffine(face_gray, M, (W, H))
    variants.append(rot_pos)

    # Small rotation -5°
    M = cv2.getRotationMatrix2D((W / 2, H / 2), -5, 1.0)
    rot_neg = cv2.warpAffine(face_gray, M, (W, H))
    variants.append(rot_neg)

    return [v.reshape(1, H, W, 1) for v in variants]


def tta_predict(model, face_tensor: np.ndarray, steps: int = 5) -> np.ndarray:
    """
    Run TTA on a single face.
    
    Args:
        model      : Keras model
        face_tensor: (1, 48, 48, 1) float32
        steps      : how many augmentations to use (max 6)
    
    Returns:
        Averaged probability array (7,)
    """
    face_gray = face_tensor[0, :, :, 0]
    variants  = _augment_face(face_gray)[:steps]

    preds = [model.predict(v, verbose=0)[0] for v in variants]
    return np.mean(preds, axis=0)


def tta_predict_mobilenet(model, face_48_gray: np.ndarray, steps: int = 5) -> np.ndarray:
    """TTA for MobileNetV2 (96x96 RGB input)."""
    face = face_48_gray[0, :, :, 0]
    variants = _augment_face(face)[:steps]

    preds = []
    for v in variants:
        face_96 = cv2.resize(v[0, :, :, 0], (96, 96), interpolation=cv2.INTER_CUBIC)
        inp = face_96.reshape(1, 96, 96, 1).astype(np.float32)
        preds.append(model.predict(inp, verbose=0)[0])

    return np.mean(preds, axis=0)
