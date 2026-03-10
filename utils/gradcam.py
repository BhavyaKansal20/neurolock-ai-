"""
NeuroLock AI v2 — Grad-CAM Visualization
Shows WHERE the model looks to make emotion predictions.
"""

import numpy as np
import cv2
import tensorflow as tf


class GradCAM:
    def __init__(self, model: tf.keras.Model, layer_name: str = None):
        self.model = model
        self.layer_name = layer_name or self._find_last_conv()

    def _find_last_conv(self) -> str:
        for layer in reversed(self.model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D,
                                   tf.keras.layers.SeparableConv2D)):
                return layer.name
        return self.model.layers[-3].name

    def compute(self, input_tensor: np.ndarray, class_idx: int = None) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.
        input_tensor: shape (1, H, W, 1)
        Returns: heatmap (H, W) float32 in [0, 1]
        """
        grad_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=[
                self.model.get_layer(self.layer_name).output,
                self.model.output
            ]
        )

        with tf.GradientTape() as tape:
            inputs = tf.cast(input_tensor, tf.float32)
            conv_outputs, predictions = grad_model(inputs)
            if class_idx is None:
                class_idx = int(tf.argmax(predictions[0]))
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.nn.relu(heatmap)

        heatmap = heatmap.numpy()
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap.astype(np.float32)

    def overlay(self, face_bgr: np.ndarray, heatmap: np.ndarray,
                alpha: float = 0.45) -> np.ndarray:
        """Overlay heatmap on face image. Returns BGR image."""
        h, w = face_bgr.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(face_bgr, 1 - alpha, colormap, alpha, 0)
        return overlay

    def to_b64(self, overlay_bgr: np.ndarray) -> str:
        """Encode overlay image to base64 string for frontend."""
        import base64
        _, buf = cv2.imencode('.jpg', overlay_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode()
