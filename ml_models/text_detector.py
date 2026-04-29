# -*- coding: utf-8 -*-
import re
import unicodedata

import joblib
import numpy as np


class TextDetector:
    # Fake-news detector: trained classifier + TF-IDF vectorizer.
    def __init__(self, model_path: str, vectorizer_path: str):
        self._model_path = model_path
        self._vectorizer_path = vectorizer_path
        self._model = None
        self._vectorizer = None
        self._model_name = "PassiveAggressiveClassifier + TF-IDF"
        self._load()

    def _load(self) -> None:
        self._model = joblib.load(self._model_path)
        self._vectorizer = joblib.load(self._vectorizer_path)

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        t = unicodedata.normalize("NFC", text)
        t = re.sub(r"<[^>]+>", " ", t)
        t = re.sub(r"http\S+", " ", t)
        t = re.sub(r"[^\w\s\u0600-\u06FF]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def predict(self, text: str) -> dict:
        cleaned = self.clean_text(text)
        if not cleaned:
            return {
                "label": "Fake",
                "confidence": 0.5,
                "model": self._model_name,
            }
        X = self._vectorizer.transform([cleaned])
        pred = int(self._model.predict(X)[0])
        confidence = 0.7
        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(X)[0]
            for i, c in enumerate(self._model.classes_):
                if int(c) == pred:
                    confidence = float(proba[i])
                    break
        elif hasattr(self._model, "decision_function"):
            raw = float(self._model.decision_function(X)[0])
            confidence = float(1.0 / (1.0 + np.exp(-np.clip(raw, -30.0, 30.0))))
        label = "Real" if pred == 1 else "Fake"
        return {
            "label": label,
            "confidence": min(1.0, max(0.0, confidence)),
            "model": self._model_name,
        }

    def is_loaded(self) -> bool:
        return self._model is not None and self._vectorizer is not None
