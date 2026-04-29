# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from PIL import Image


class ImageDetector:
    # Fake image detector (trained EfficientNetB0).
    def __init__(self, model_path: str):
        self._model_path = model_path
        self.model = None
        self.input_size = (224, 224)
        self._model_label = "EfficientNetB0"
        self._load()

    def _load(self) -> None:
        self.model = tf.keras.models.load_model(self._model_path)

    def preprocess_image(self, image_path: str):
        img = Image.open(image_path).convert("RGB")
        img = img.resize(self.input_size)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)

    def preprocess_rgb_array(self, rgb: np.ndarray):
        img = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
        img = img.resize(self.input_size)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)

    def _predict_raw(self, batch: np.ndarray) -> float:
        return float(self.model.predict(batch, verbose=0)[0][0])

    def _result_from_sigmoid_real(self, raw: float) -> dict:
        # CIFAKE: FAKE=0, REAL=1; sigmoid output is P(real), app uses p_fake = 1 - raw
        p_fake = 1.0 - raw
        label = "Fake" if p_fake >= 0.5 else "Real"
        confidence = p_fake if label == "Fake" else (1.0 - p_fake)
        return {
            "label": label,
            "confidence": min(1.0, max(0.0, confidence)),
            "model": self._model_label,
        }

    def predict(self, image_path: str) -> dict:
        batch = self.preprocess_image(image_path)
        raw = self._predict_raw(batch)
        return self._result_from_sigmoid_real(raw)

    def predict_rgb(self, rgb: np.ndarray) -> dict:
        """Predict from an RGB array without a file on disk (useful for video frames)."""
        batch = self.preprocess_rgb_array(rgb)
        raw = self._predict_raw(batch)
        return self._result_from_sigmoid_real(raw)

    def is_loaded(self) -> bool:
        return self.model is not None
