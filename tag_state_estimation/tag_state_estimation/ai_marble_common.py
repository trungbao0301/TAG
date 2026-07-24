"""Shared preprocessing and decoding for the learned marble detector."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MarbleDetection:
    """One full-frame marble detection in source-image pixel coordinates."""

    visible: bool
    x_px: float
    y_px: float
    confidence: float


def sigmoid(value):
    value = np.asarray(value, dtype=np.float32)
    result = np.empty_like(value)
    positive = value >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def decode_heatmap(
    logits, image_width, image_height, threshold=0.90, valid_roi=None
):
    """Decode a 1x1xHxW or HxW heatmap into a source-image position."""
    heatmap = np.asarray(logits, dtype=np.float32).squeeze()
    if heatmap.ndim != 2:
        raise ValueError(f"Expected a 2-D heatmap, got shape {heatmap.shape}")
    if valid_roi is not None:
        x_min, y_min, x_max, y_max = map(float, valid_roi)
        if not (0.0 <= x_min < x_max <= 1.0 and 0.0 <= y_min < y_max <= 1.0):
            raise ValueError(f"Invalid normalized ROI: {valid_roi}")
        yy_all, xx_all = np.mgrid[: heatmap.shape[0], : heatmap.shape[1]]
        x_norm = xx_all / float(heatmap.shape[1])
        y_norm = yy_all / float(heatmap.shape[0])
        allowed = (
            (x_norm >= x_min)
            & (x_norm <= x_max)
            & (y_norm >= y_min)
            & (y_norm <= y_max)
        )
        heatmap = np.where(allowed, heatmap, -1.0e9)
    probabilities = sigmoid(heatmap)
    flat_index = int(np.argmax(probabilities))
    y_cell, x_cell = np.unravel_index(flat_index, probabilities.shape)
    confidence = float(probabilities[y_cell, x_cell])
    # Labels are encoded as x / image_width * heatmap_width, so heatmap index
    # zero maps to source pixel zero (there is no half-cell offset). Recover
    # sub-cell precision with a local soft-argmax around the strongest cell.
    radius = 3
    y0 = max(0, y_cell - radius)
    y1 = min(heatmap.shape[0], y_cell + radius + 1)
    x0 = max(0, x_cell - radius)
    x1 = min(heatmap.shape[1], x_cell + radius + 1)
    patch = heatmap[y0:y1, x0:x1]
    weights = np.exp(patch - float(np.max(patch)))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    x_index = float(np.sum(weights * xx) / np.sum(weights))
    y_index = float(np.sum(weights * yy) / np.sum(weights))
    x_px = x_index * float(image_width) / probabilities.shape[1]
    y_px = y_index * float(image_height) / probabilities.shape[0]
    return MarbleDetection(confidence >= threshold, x_px, y_px, confidence)


class OnnxMarbleDetector:
    """OpenCV-DNN runner for the ONNX heatmap model produced by the trainer."""

    def __init__(
        self,
        model_path,
        input_width=320,
        input_height=200,
        confidence_threshold=0.90,
        backend="cpu",
        valid_roi=None,
    ):
        self.input_width = int(input_width)
        self.input_height = int(input_height)
        self.confidence_threshold = float(confidence_threshold)
        self.valid_roi = valid_roi
        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        if backend == "cuda":
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect(self, frame):
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0 / 255.0,
            size=(self.input_width, self.input_height),
            mean=(0.0, 0.0, 0.0),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        logits = self.net.forward()
        return decode_heatmap(
            logits,
            image_width=width,
            image_height=height,
            threshold=self.confidence_threshold,
            valid_roi=self.valid_roi,
        )
