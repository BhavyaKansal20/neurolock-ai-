"""
NeuroLock AI v2 — Grad-CAM Visualization
==========================================
Generates heatmaps showing WHICH part of the face
the model focused on when making a prediction.
Jaw-drop feature for demos 🔥
"""

import cv2
import numpy as np
import tensorflow as tf


def get_gradcam_heatmap(
    model,
    face_tensor: np.ndarray,
    class_idx:   int,
    last_conv_layer: str = 'head_conv',
) -> np.ndarray:
    """
    Generate Grad-CAM heatmap for a prediction.

    Args:
        model           : Keras model
        face_tensor     : (1, 48, 48, 1) float32
        class_idx       : predicted class index
        last_conv_layer : name of last conv layer to hook

    Returns:
        heatmap: (48, 48) float32 array in [0, 1]
    """
    # Build a model that outputs both the last conv activations and the predictions
    try:
        grad_model = tf.keras.models.Model(
            inputs  = model.inputs,
            outputs = [model.get_layer(last_conv_layer).output, model.output]
        )
    except ValueError:
        # Fallback: find any last conv layer
        conv_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
        if not conv_layers:
            return np.zeros((48, 48), dtype=np.float32)
        grad_model = tf.keras.models.Model(
            inputs  = model.inputs,
            outputs = [conv_layers[-1].output, model.output]
        )

    with tf.GradientTape() as tape:
        face_tf = tf.cast(face_tensor, tf.float32)
        conv_outputs, predictions = grad_model(face_tf)
        loss = predictions[:, class_idx]

    grads    = tape.gradient(loss, conv_outputs)
    pooled   = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_outputs[0]

    # Weighted combination of activation maps
    heatmap = conv_out @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Resize to input size
    heatmap = cv2.resize(heatmap, (48, 48))
    heatmap = np.clip(heatmap, 0, 1).astype(np.float32)
    return heatmap


def overlay_heatmap_on_face(
    face_bgr:   np.ndarray,
    heatmap:    np.ndarray,
    alpha:      float = 0.45,
    colormap:   int   = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on a face image.

    Args:
        face_bgr : H×W×3 BGR face crop
        heatmap  : H×W float32 in [0,1]
        alpha    : blend strength
        colormap : cv2 colormap

    Returns:
        H×W×3 BGR image with heatmap overlay
    """
    H, W = face_bgr.shape[:2]
    hm_resized = cv2.resize(heatmap, (W, H))
    hm_uint8   = np.uint8(255 * hm_resized)
    hm_colored = cv2.applyColorMap(hm_uint8, colormap)
    overlay    = cv2.addWeighted(face_bgr, 1 - alpha, hm_colored, alpha, 0)
    return overlay


def heatmap_to_base64(face_bgr: np.ndarray, heatmap: np.ndarray) -> str:
    """Return Grad-CAM overlay as base64 JPEG for API response."""
    import base64
    overlay = overlay_heatmap_on_face(face_bgr, heatmap)
    _, buf  = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()
