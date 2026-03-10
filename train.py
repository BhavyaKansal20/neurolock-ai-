"""
NeuroLock AI v2 — Training Script
==================================
Trains BOTH Mini-Xception and MobileNetV2 models.
MobileNetV2: Transfer learning, expected 68-72% accuracy.
Mini-Xception: Lightweight fast model for low-power devices.

Usage:
  python train.py --dataset archive            # Train both models
  python train.py --dataset archive --model xception
  python train.py --dataset archive --model mobilenet
"""

import os, sys, json, argparse
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)
from sklearn.utils.class_weight import compute_class_weight
from model.architecture  import build_mini_xception, EMOTION_LABELS, NUM_CLASSES, INPUT_SHAPE
from model.mobilenet_model import build_mobilenet_v2, unfreeze_top_layers, INPUT_SHAPE_MOBILENET
from utils.preprocessing import get_data_generators

os.makedirs('checkpoints', exist_ok=True)
os.makedirs('exports',     exist_ok=True)
os.makedirs('logs',        exist_ok=True)


def get_class_weights(dataset_path: str, split: str = 'train') -> dict:
    """Compute class weights to handle imbalanced classes (sad/fearful)."""
    split_path = os.path.join(dataset_path, split)
    counts = []
    for label in EMOTION_LABELS:
        d = os.path.join(split_path, label)
        counts.append(len(os.listdir(d)) if os.path.isdir(d) else 0)
    total = sum(counts)
    if total == 0:
        return {i: 1.0 for i in range(NUM_CLASSES)}

    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(NUM_CLASSES),
        y=np.repeat(np.arange(NUM_CLASSES), counts)
    )
    w_dict = {i: float(round(w, 4)) for i, w in enumerate(weights)}
    print("\n  Class weights (to fix Sad/Fearful imbalance):")
    for i, label in enumerate(EMOTION_LABELS):
        print(f"    {label:12s}: {w_dict[i]:.3f}  ({counts[i]} images)")
    return w_dict


def train_xception(dataset: str, run_id: str, epochs: int = 80):
    print("\n" + "="*60)
    print("  TRAINING: Mini-Xception")
    print("="*60)

    train_gen, val_gen, _, class_weights = get_data_generators(dataset, batch_size=64)
    model = build_mini_xception()
    model.summary()

    ckpt = f'checkpoints/xception_{run_id}_best.keras'
    cbs  = [
        EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(ckpt, monitor='val_accuracy', save_best_only=True, verbose=1),
        TensorBoard(log_dir=f'logs/xception_{run_id}'),
    ]

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    hist = model.fit(
        train_gen, validation_data=val_gen, epochs=epochs,
        callbacks=cbs, class_weight=class_weights, verbose=1
    )

    out = f'exports/neurolock_model.keras'
    model.save(out)
    print(f"\n  ✓ Mini-Xception saved: {out}")
    save_meta(run_id, 'xception', hist, ckpt, out)
    return model, hist


def train_mobilenet(dataset: str, run_id: str, epochs_frozen: int = 20,
                    epochs_finetune: int = 50):
    print("\n" + "="*60)
    print("  TRAINING: MobileNetV2 (Transfer Learning)")
    print("="*60)

    # MobileNetV2 needs 96x96 — get_data_generators uses 48x48, so build 96x96 generators separately
    from tensorflow.keras.preprocessing.image import ImageDataGenerator as _IDG
    _aug96 = _IDG(
        rescale=1.0/255.0, rotation_range=15, width_shift_range=0.15,
        height_shift_range=0.15, shear_range=0.1, zoom_range=0.15,
        horizontal_flip=True, fill_mode='nearest', validation_split=0.1,
    )
    _kw96 = dict(target_size=(96, 96), color_mode='grayscale',
                 class_mode='categorical', batch_size=32, seed=42)
    train_gen = _aug96.flow_from_directory(
        os.path.join(dataset, 'train'), subset='training', shuffle=True, **_kw96)
    val_gen = _aug96.flow_from_directory(
        os.path.join(dataset, 'train'), subset='validation', shuffle=False, **_kw96)
    # Get class weights from standard generator
    _, _, _, class_weights = get_data_generators(dataset, batch_size=32)
    model, base_model = build_mobilenet_v2()
    model.summary()

    ckpt = f'checkpoints/mobilenet_{run_id}_best.keras'
    cbs  = [
        EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=4, min_lr=1e-7, verbose=1),
        ModelCheckpoint(ckpt, monitor='val_accuracy', save_best_only=True, verbose=1),
        TensorBoard(log_dir=f'logs/mobilenet_{run_id}'),
    ]

    # Phase 1 — Train head only (frozen base)
    print(f"\n  Phase 1: Training head ({epochs_frozen} epochs, base frozen)")
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        train_gen, validation_data=val_gen, epochs=epochs_frozen,
        callbacks=cbs[:2], class_weight=class_weights, verbose=1
    )

    # Phase 2 — Fine-tune top layers
    print(f"\n  Phase 2: Fine-tuning top 40 layers ({epochs_finetune} epochs)")
    unfreeze_top_layers(base_model, num_layers=40)
    model.compile(
        optimizer=Adam(learning_rate=1e-5),   # Much lower LR for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    hist = model.fit(
        train_gen, validation_data=val_gen, epochs=epochs_finetune,
        callbacks=cbs, class_weight=class_weights, verbose=1
    )

    out = 'exports/mobilenet_model.keras'
    model.save(out)
    print(f"\n  ✓ MobileNetV2 saved: {out}")

    # Test set pe final evaluation karo
    print("\n  Evaluating on test set...")
    _test96 = _IDG(rescale=1.0/255.0)
    test_gen = _test96.flow_from_directory(
        os.path.join(dataset, 'test'), **{**_kw96, 'shuffle': False})
    if test_gen.n > 0:
        test_loss, test_acc = model.evaluate(test_gen, verbose=0)
        print(f"  ✓ TEST ACCURACY: {test_acc*100:.2f}%")
    else:
        print("  (No test/ folder found, skipping test eval)")

    save_meta(run_id, 'mobilenet', hist, ckpt, out)
    return model, hist


def save_meta(run_id, model_type, hist, ckpt_path, export_path):
    val_acc = max(hist.history.get('val_accuracy', [0]))
    meta = {
        'run_id':       run_id,
        'model_type':   model_type,
        'best_val_acc': round(float(val_acc), 4),
        'trained_at':   datetime.now().isoformat(),
        'checkpoint':   ckpt_path,
        'export':       export_path,
        'emotions':     EMOTION_LABELS,
        'num_classes':  NUM_CLASSES,
    }
    fname = f'exports/{model_type}_meta_{run_id}.json'
    with open(fname, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  ✓ Metadata saved: {fname}")
    print(f"\n  Best val accuracy: {val_acc*100:.2f}%")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, help='Path to dataset (contains train/ and test/)')
    p.add_argument('--model',   choices=['xception','mobilenet','both'], default='both')
    p.add_argument('--epochs',  type=int, default=80)
    args = p.parse_args()

    if not os.path.isdir(args.dataset):
        print(f"Dataset not found: {args.dataset}")
        sys.exit(1)

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"\nNeuroLock AI v2 Training | Run: {run_id}")
    print(f"Dataset: {args.dataset} | Model: {args.model}")

    if args.model in ('xception', 'both'):
        train_xception(args.dataset, run_id, args.epochs)

    if args.model in ('mobilenet', 'both'):
        train_mobilenet(args.dataset, run_id)

    print("\n" + "="*60)
    print("  Training complete! Next step:")
    print("  python server.py --port 5001")
    print("="*60)
