# -*- coding: utf-8 -*-
import os
import re
import unicodedata

import joblib
import numpy as np

from utils.text_sensational import emotion_hit_count


class TextDetector:
    # Fake-news detector: trained classifier + TF-IDF vectorizer.
    def __init__(self, model_path: str, vectorizer_path: str):
        self._model_path = model_path
        self._vectorizer_path = vectorizer_path
        self._model = None
        self._vectorizer = None
        self._model_name = "News wording & style (TF-IDF + linear classifier, WELFake-style)"
        self._fake_class_cache = None
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

    def _resolve_fake_class(self) -> int:
        if self._fake_class_cache is not None:
            return self._fake_class_cache
        manual = (os.environ.get("TRUELENS_TEXT_FAKE_CLASS") or "").strip()
        if manual != "":
            try:
                self._fake_class_cache = int(manual)
                return self._fake_class_cache
            except ValueError:
                pass
        auto_off = (os.environ.get("TRUELENS_TEXT_AUTO_FAKE_CLASS") or "1").strip().lower() in (
            "0",
            "false",
            "no",
            "off",
        )
        if auto_off:
            self._fake_class_cache = 0
            return self._fake_class_cache
        classes = [int(c) for c in getattr(self._model, "classes_", [])]
        if len(classes) != 2 or set(classes) != {0, 1}:
            self._fake_class_cache = 0
            return self._fake_class_cache
        wire = (
            "The federal agency released annual crop estimates Tuesday. Officials said rainfall "
            "in the region was near the ten year average, and commodity futures were unchanged "
            "in after hours trading."
        )
        junk = (
            "BREAKING you wont believe this shocking secret they dont want you to see miracle cure "
            "guaranteed destroy the cover up immediately share before deleted horrible catastrophe urgent"
        )
        Xw = self._vectorizer.transform([self.clean_text(wire)])
        Xj = self._vectorizer.transform([self.clean_text(junk)])
        try:
            c_j = int(self._model.predict(Xj)[0])
            c_w = int(self._model.predict(Xw)[0])
        except Exception:
            c_j, c_w = 0, 1
        if c_j != c_w:
            self._fake_class_cache = c_j
            return self._fake_class_cache
        if hasattr(self._model, "predict_proba"):
            pw = self._model.predict_proba(Xw)[0]
            pj = self._model.predict_proba(Xj)[0]
            pmap_w = {int(c): float(p) for c, p in zip(self._model.classes_, pw)}
            pmap_j = {int(c): float(p) for c, p in zip(self._model.classes_, pj)}
            d0 = pmap_j[0] - pmap_w[0]
            d1 = pmap_j[1] - pmap_w[1]
            if d0 > d1 + 0.005:
                self._fake_class_cache = 0
            elif d1 > d0 + 0.005:
                self._fake_class_cache = 1
            else:
                self._fake_class_cache = 0
            return self._fake_class_cache
        self._fake_class_cache = 0
        return self._fake_class_cache

    def predict(self, text: str) -> dict:
        cleaned = self.clean_text(text)
        if not cleaned:
            return {
                "label": "Fake",
                "confidence": 0.5,
                "model": self._model_name,
            }
        X = self._vectorizer.transform([cleaned])
        invert = os.environ.get("TRUELENS_TEXT_INVERT", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        fake_class = self._resolve_fake_class()
        classes = [int(c) for c in getattr(self._model, "classes_", [])]
        label = "Fake"
        confidence = 0.7
        raw = None
        if hasattr(self._model, "predict_proba") and len(classes) >= 2:
            proba = self._model.predict_proba(X)[0]
            pmap = {int(c): float(p) for c, p in zip(self._model.classes_, proba)}
            if fake_class in pmap:
                p_fake = pmap[fake_class]
                if invert:
                    p_fake = 1.0 - p_fake
                p_fake_model = float(p_fake)
                hw = (os.environ.get("TRUELENS_TEXT_HEURISTIC_WEIGHT") or "0.52").strip()
                try:
                    blend_w = float(hw)
                except ValueError:
                    blend_w = 0.52
                blend_w = min(1.0, max(0.0, blend_w))
                hits = emotion_hit_count(text)
                if blend_w > 0:
                    h_norm = min(1.0, hits / 5.0)
                    p_fake = (1.0 - blend_w) * p_fake + blend_w * h_norm
                strong_on = (os.environ.get("TRUELENS_TEXT_STRONG_HEURISTIC") or "1").strip().lower() not in (
                    "0",
                    "false",
                    "no",
                    "off",
                )
                if strong_on and hits >= 4 and p_fake_model < 0.55:
                    p_fake = max(float(p_fake), 0.58)
                label = "Fake" if p_fake >= 0.5 else "Real"
                confidence = p_fake if label == "Fake" else (1.0 - p_fake)
            else:
                idx = int(np.argmax(proba))
                pred_class = int(self._model.classes_[idx])
                if invert and len(classes) == 2 and set(classes) == {0, 1}:
                    pred_class = 1 - pred_class
                label = "Fake" if pred_class == fake_class else "Real"
                confidence = float(proba[idx])
        else:
            pred = int(self._model.predict(X)[0])
            if invert and len(classes) == 2 and set(classes) == {0, 1}:
                pred = 1 - pred
            elif invert:
                pred = 0 if pred == 1 else 1
            label = "Fake" if pred == fake_class else "Real"
            if hasattr(self._model, "decision_function"):
                raw = float(self._model.decision_function(X)[0])
                confidence = float(1.0 / (1.0 + np.exp(-np.clip(raw, -30.0, 30.0))))
        return {
            "label": label,
            "confidence": min(1.0, max(0.0, confidence)),
            "model": self._model_name,
            "raw": raw,
        }

    def is_loaded(self) -> bool:
        return self._model is not None and self._vectorizer is not None
