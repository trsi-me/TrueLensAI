# -*- coding: utf-8 -*-
import os

import numpy as np
from PIL import Image


class _ImageDetectorKeras:
    def __init__(self, model_path: str):
        self._model_path = model_path
        self.model = None
        self.input_size = (224, 224)
        self._model_label = "EfficientNetB0 (Keras)"
        self._load()

    def _load(self) -> None:
        import tensorflow as tf

        # Some environments have standalone Keras that rejects DepthwiseConv2D(groups=1)
        # serialized by older TF/Keras; strip that key during deserialization.
        def _wrap(layer_cls):
            orig = layer_cls.from_config.__func__

            @classmethod
            def from_config_fixed(cls, config):
                cfg = dict(config)
                cfg.pop("groups", None)
                return orig(cls, cfg)

            layer_cls.from_config = from_config_fixed

        try:
            _wrap(tf.keras.layers.DepthwiseConv2D)
        except Exception:
            pass

        from tensorflow.keras.models import load_model

        self.model = load_model(self._model_path, compile=False)

    def preprocess_image(self, image_path: str):
        img = Image.open(image_path).convert("RGB")
        img = img.resize(self.input_size)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        # Match EfficientNet preprocess_input when trained with it: scale to [-1, 1]
        mode = os.environ.get("TRUELENS_IMAGE_PREPROCESS", "").strip().lower()
        if mode in ("effnet", "efficientnet", "tf"):
            arr = (arr * 2.0) - 1.0
        return np.expand_dims(arr, axis=0)

    def preprocess_rgb_array(self, rgb: np.ndarray):
        img = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
        img = img.resize(self.input_size)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        mode = os.environ.get("TRUELENS_IMAGE_PREPROCESS", "").strip().lower()
        if mode in ("effnet", "efficientnet", "tf"):
            arr = (arr * 2.0) - 1.0
        return np.expand_dims(arr, axis=0)

    def _predict_raw(self, batch: np.ndarray) -> float:
        return float(self.model.predict(batch, verbose=0)[0][0])

    def _result_from_sigmoid_real(self, raw: float) -> dict:
        # raw is a sigmoid output from the model. Some training pipelines use:
        # - raw = P(real)  (CIFAKE default)
        # - raw = P(fake)  (other datasets)
        # Allow flipping at runtime to avoid retraining when labels are inverted.
        invert = os.environ.get("TRUELENS_IMAGE_INVERT", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        p_fake = raw if invert else (1.0 - raw)
        label = "Fake" if p_fake >= 0.5 else "Real"
        confidence = p_fake if label == "Fake" else (1.0 - p_fake)
        return {
            "label": label,
            "confidence": min(1.0, max(0.0, confidence)),
            "model": self._model_label,
            "raw": raw,
        }

    def predict(self, image_path: str) -> dict:
        batch = self.preprocess_image(image_path)
        raw = self._predict_raw(batch)
        return self._result_from_sigmoid_real(raw)

    def predict_rgb(self, rgb: np.ndarray) -> dict:
        batch = self.preprocess_rgb_array(rgb)
        raw = self._predict_raw(batch)
        return self._result_from_sigmoid_real(raw)

    def is_loaded(self) -> bool:
        return self.model is not None


class _ImageDetectorTflite:
    def __init__(self, model_path: str):
        self._model_path = model_path
        self._interpreter = None
        self._in_index = None
        self._out_index = None
        self.input_size = (224, 224)
        self._model_label = "EfficientNetB0 (TFLite)"
        self._load()

    def _load(self) -> None:
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            import tensorflow as tf

            tflite = tf.lite

        self._interpreter = tflite.Interpreter(model_path=self._model_path)
        self._interpreter.allocate_tensors()
        in_d = self._interpreter.get_input_details()[0]
        out_d = self._interpreter.get_output_details()[0]
        self._in_index = in_d["index"]
        self._out_index = out_d["index"]
        self._in_dtype = in_d.get("dtype", np.float32)
        h, w = self.input_size[0], self.input_size[1]
        shape = in_d.get("shape")
        if shape is not None and len(shape) >= 3:
            h, w = int(shape[1]), int(shape[2])
            self.input_size = (h, w)

    def preprocess_image(self, image_path: str):
        img = Image.open(image_path).convert("RGB")
        img = img.resize(self.input_size)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        mode = os.environ.get("TRUELENS_IMAGE_PREPROCESS", "").strip().lower()
        if mode in ("effnet", "efficientnet", "tf"):
            arr = (arr * 2.0) - 1.0
        return np.expand_dims(arr, axis=0).astype(self._in_dtype)

    def preprocess_rgb_array(self, rgb: np.ndarray):
        img = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
        img = img.resize(self.input_size)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        mode = os.environ.get("TRUELENS_IMAGE_PREPROCESS", "").strip().lower()
        if mode in ("effnet", "efficientnet", "tf"):
            arr = (arr * 2.0) - 1.0
        return np.expand_dims(arr, axis=0).astype(self._in_dtype)

    def _predict_raw(self, batch: np.ndarray) -> float:
        self._interpreter.set_tensor(self._in_index, batch)
        self._interpreter.invoke()
        out = self._interpreter.get_tensor(self._out_index)
        out = np.asarray(out, dtype=np.float64).reshape(-1)
        return float(out[0])

    def _result_from_sigmoid_real(self, raw: float) -> dict:
        invert = os.environ.get("TRUELENS_IMAGE_INVERT", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        p_fake = raw if invert else (1.0 - raw)
        label = "Fake" if p_fake >= 0.5 else "Real"
        confidence = p_fake if label == "Fake" else (1.0 - p_fake)
        return {
            "label": label,
            "confidence": min(1.0, max(0.0, confidence)),
            "model": self._model_label,
            "raw": raw,
        }

    def predict(self, image_path: str) -> dict:
        batch = self.preprocess_image(image_path)
        raw = self._predict_raw(batch)
        return self._result_from_sigmoid_real(raw)

    def predict_rgb(self, rgb: np.ndarray) -> dict:
        batch = self.preprocess_rgb_array(rgb)
        raw = self._predict_raw(batch)
        return self._result_from_sigmoid_real(raw)

    def is_loaded(self) -> bool:
        return self._interpreter is not None


def ImageDetector(model_path: str):
    if model_path.lower().endswith(".tflite"):
        return _ImageDetectorTflite(model_path)
    return _ImageDetectorKeras(model_path)
