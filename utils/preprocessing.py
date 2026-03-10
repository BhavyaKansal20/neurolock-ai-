"""
NeuroLock AI — Preprocessing Utilities
Data loading, augmentation, and face preprocessing pipelines.
"""

import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from model.architecture import INPUT_SHAPE, EMOTION_LABELS


# ── Dataset Config ────────────────────────────────────────────────────────────
IMG_SIZE   = INPUT_SHAPE[0]   # 48
BATCH_SIZE = 64
SEED       = 42


def get_data_generators(dataset_path: str, batch_size: int = BATCH_SIZE):
    """
    Build train / validation / test ImageDataGenerators.

    Expected dataset folder layout (from FER Kaggle dataset):
        <dataset_path>/
            train/
                angry/   disgust/  fear/  happy/  neutral/  sad/  surprise/
            test/
                angry/   disgust/  fear/  happy/  neutral/  sad/  surprise/

    Returns: (train_gen, val_gen, test_gen, class_weights)
    """
    train_dir = os.path.join(dataset_path, 'train')
    test_dir  = os.path.join(dataset_path, 'test')

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Train directory not found: {train_dir}\n"
            "Download the dataset with:\n"
            "  python scripts/download_dataset.py"
        )

    # ── Augmentation for training ─────────────────────────────────────────────
    train_aug = ImageDataGenerator(
        rescale           = 1.0 / 255.0,
        rotation_range    = 15,
        width_shift_range = 0.15,
        height_shift_range= 0.15,
        shear_range       = 0.1,
        zoom_range        = 0.15,
        horizontal_flip   = True,
        fill_mode         = 'nearest',
        validation_split  = 0.1,      # 10% of train for validation
    )

    # ── Test / inference (no augmentation) ────────────────────────────────────
    test_aug = ImageDataGenerator(rescale=1.0 / 255.0)

    common_kwargs = dict(
        target_size  = (IMG_SIZE, IMG_SIZE),
        color_mode   = 'grayscale',
        class_mode   = 'categorical',
        batch_size   = batch_size,
        seed         = SEED,
        shuffle      = True,
    )

    train_gen = train_aug.flow_from_directory(
        train_dir, subset='training', **common_kwargs
    )
    val_gen = train_aug.flow_from_directory(
        train_dir, subset='validation', **{**common_kwargs, 'shuffle': False}
    )
    test_gen = test_aug.flow_from_directory(
        test_dir, **{**common_kwargs, 'shuffle': False}
    )

    # ── Class weights (dataset is imbalanced) ─────────────────────────────────
    labels = train_gen.classes
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weight_dict = dict(enumerate(class_weights))

    print("\n── Dataset loaded ──────────────────────────────────────────")
    print(f"  Train samples      : {train_gen.samples:,}")
    print(f"  Validation samples : {val_gen.samples:,}")
    print(f"  Test samples       : {test_gen.samples:,}")
    print(f"  Classes            : {list(train_gen.class_indices.keys())}")
    print(f"  Class weights      : {class_weight_dict}")
    print("────────────────────────────────────────────────────────────\n")

    return train_gen, val_gen, test_gen, class_weight_dict


# ── Inference Preprocessing ───────────────────────────────────────────────────

def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    """
    Preprocess a face crop (BGR numpy array) for model inference.
    Returns shape (1, 48, 48, 1) float32 array in [0, 1].
    """
    gray  = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    norm  = resized.astype(np.float32) / 255.0
    return norm.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def preprocess_frame(frame_bgr: np.ndarray):
    """
    Given a full camera frame, detect all faces and return:
        - list of face crops (preprocessed tensors)
        - list of bounding boxes (x, y, w, h) in original frame coords
    """
    face_cascade = _get_face_cascade()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor  = 1.1,
        minNeighbors = 5,
        minSize      = (30, 30),
        flags        = cv2.CASCADE_SCALE_IMAGE
    )

    crops, boxes = [], []
    for (x, y, w, h) in faces:
        # Add a small margin
        margin = int(0.1 * w)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame_bgr.shape[1], x + w + margin)
        y2 = min(frame_bgr.shape[0], y + h + margin)
        face_crop = frame_bgr[y1:y2, x1:x2]
        crops.append(preprocess_face(face_crop))
        boxes.append((x1, y1, x2 - x1, y2 - y1))

    return crops, boxes


_cascade_cache = None

def _get_face_cascade():
    global _cascade_cache
    if _cascade_cache is None:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        _cascade_cache = cv2.CascadeClassifier(cascade_path)
    return _cascade_cache
