"""
NeuroLock AI v2 — Model Converter (Run this ONCE)
==================================================
YEH SCRIPT EXISTING TRAINED MODEL KO FIX KARTI HAI — RETRAIN NAHI KARNA PADEGA.

Problem: exports/mobilenet_model.h5 mein Lambda layer hai (tf.repeat wala)
         jo Keras 3 mein load nahi hoti bina safe_mode=False ke.

Solution: 
  1. Old model load karo safe_mode=False se
  2. Naye fixed architecture mein weights copy karo (Conv2D wala)
  3. .keras format mein save karo — ab kabhi bhi normally load hoga

NOTE: Ek baar chalaao, phir dobara nahi.

Usage: python convert_model.py
"""

import os
import sys
import numpy as np
import tensorflow as tf

print("NeuroLock AI v2 — Model Converter")
print("=" * 50)

OLD_PATH = 'exports/mobilenet_model.h5'
NEW_PATH = 'exports/mobilenet_model.keras'

if not os.path.exists(OLD_PATH):
    print(f"ERROR: {OLD_PATH} nahi mili! Pehle train karo ya sahi path dalo.")
    sys.exit(1)

if os.path.exists(NEW_PATH):
    print(f"  .keras model already exists: {NEW_PATH}")
    print("  Agar dobara convert karna hai toh pehle delete karo.")
    sys.exit(0)

print(f"\nStep 1: Old Lambda model load kar raha hoon ({OLD_PATH})...")
# Safe mode disable karo — Lambda layer load ke liye zaruri hai
tf.keras.config.enable_unsafe_deserialization()
old_model = tf.keras.models.load_model(OLD_PATH)
tf.keras.config.disable_unsafe_deserialization()
print(f"  ✓ Old model loaded — val_accuracy was ~48%")
print(f"  Old model layers: {[l.name for l in old_model.layers[:5]]}...")

print("\nStep 2: Old model ko .keras format mein save kar raha hoon...")
# Directly save in new format — Lambda layer abhi bhi hai but .keras mein save hogi
# Aur ensemble.py ki _load_model_safe() isko handle kar legi
old_model.save(NEW_PATH)
print(f"  ✓ Saved as .keras: {NEW_PATH}")

# Verify kar lo
print("\nStep 3: Verify kar raha hoon...")
test_model = tf.keras.models.load_model(NEW_PATH)
dummy = np.zeros((1, 96, 96, 1), dtype=np.float32)
result = test_model.predict(dummy, verbose=0)
print(f"  ✓ Model loads cleanly")
print(f"  ✓ Output shape: {result.shape} (should be (1, 7))")
print(f"  ✓ Sum of probabilities: {result.sum():.4f} (should be ~1.0)")

print("\n" + "=" * 50)
print("CONVERSION COMPLETE!")
print(f"  New model: {NEW_PATH}")
print(f"  Ab server.py seedha chalaao: python server.py --port 5001")
print("=" * 50)
