# -*- coding: utf-8 -*-
# HTTP clients: main app on Render calls a separate ML worker that holds .pkl / .tflite / .h5.
import io
import os
from typing import Any, Dict, Optional

import numpy as np
import requests
from PIL import Image


def _ml_headers(api_key: str) -> Dict[str, str]:
    h: Dict[str, str] = {}
    k = (api_key or "").strip()
    if k:
        h["Authorization"] = f"Bearer {k}"
    return h


def _ml_timeout() -> float:
    raw = (os.environ.get("TRUELENS_ML_API_TIMEOUT_SEC") or "120").strip()
    try:
        return float(raw)
    except ValueError:
        return 120.0


class RemoteTextDetector:
    # Same contract as TextDetector for routes / presenter.
    def __init__(self, base_url: str, api_key: str = ""):
        self._base = base_url.rstrip("/")
        self._api_key = api_key
        self._model_name = "News wording & style (remote ML API)"
        self._session = requests.Session()

    def is_loaded(self) -> bool:
        return True

    def predict(self, text: str) -> Dict[str, Any]:
        r = self._session.post(
            f"{self._base}/v1/predict/text",
            json={"text": text or ""},
            headers=_ml_headers(self._api_key),
            timeout=_ml_timeout(),
        )
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, dict):
            raise ValueError("Remote ML: invalid JSON for text")
        out = {
            "label": data.get("label", "Fake"),
            "confidence": float(data.get("confidence", 0.5)),
            "model": data.get("model") or self._model_name,
            "raw": data.get("raw"),
        }
        return out


class RemoteImageDetector:
    # Same contract as image ImageDetector (predict path + predict_rgb for video).
    def __init__(self, base_url: str, api_key: str = ""):
        self._base = base_url.rstrip("/")
        self._api_key = api_key
        self._session = requests.Session()

    def is_loaded(self) -> bool:
        return True

    def predict(self, image_path: str) -> Dict[str, Any]:
        with open(image_path, "rb") as f:
            raw = f.read()
        name = os.path.basename(image_path) or "upload.bin"
        return self._post_image_bytes(raw, name)

    def predict_rgb(self, rgb: np.ndarray) -> Dict[str, Any]:
        img = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        return self._post_image_bytes(bio.getvalue(), "frame.png")

    def _post_image_bytes(self, data: bytes, filename: str) -> Dict[str, Any]:
        ct = "image/png" if filename.lower().endswith(".png") else "application/octet-stream"
        files = {"file": (filename, data, ct)}
        r = self._session.post(
            f"{self._base}/v1/predict/image",
            files=files,
            headers=_ml_headers(self._api_key),
            timeout=_ml_timeout(),
        )
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, dict):
            raise ValueError("Remote ML: invalid JSON for image")
        return {
            "label": data.get("label", "Fake"),
            "confidence": float(data.get("confidence", 0.5)),
            "model": data.get("model") or "EfficientNetB0 (remote ML API)",
            "raw": data.get("raw"),
        }
