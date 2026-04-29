# -*- coding: utf-8 -*-
import gc

import cv2
import numpy as np

from config import Config


class VideoDetector:
    # Video: sample frames and run the image detector on each.
    def __init__(self, image_detector):
        self.image_detector = image_detector

    def extract_frame_rgb_list(self, video_path: str, max_frames: int = None):
        """Returns frames as RGB arrays in memory without writing PNG files to disk."""
        max_f = max_frames or Config.MAX_FRAMES_PER_VIDEO
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], 0.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        duration_sec = total / fps if fps > 0 else 0
        frames_rgb = []
        if total <= 0:
            cap.release()
            return [], 0.0
        if total <= max_f:
            indices = list(range(total))
        else:
            step = total / float(max_f)
            indices = [min(int(i * step), total - 1) for i in range(max_f)]
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_rgb.append(rgb)
        cap.release()
        return frames_rgb, duration_sec

    def predict(self, video_path: str) -> dict:
        frames_rgb, _duration = self.extract_frame_rgb_list(
            video_path, Config.MAX_FRAMES_PER_VIDEO
        )
        if not frames_rgb:
            return {
                "label": "Fake",
                "confidence": 0.5,
                "frames_analyzed": 0,
                "model": "EfficientNetB0 (video)",
                "std": 0.0,
            }
        scores = []
        try:
            for rgb in frames_rgb:
                out = self.image_detector.predict_rgb(rgb)
                fake_prob = (
                    out["confidence"]
                    if out["label"] == "Fake"
                    else (1.0 - out["confidence"])
                )
                scores.append(fake_prob)
        finally:
            frames_rgb.clear()
            gc.collect()

        mean_fake = float(np.mean(scores)) if scores else 0.5
        std_fake = float(np.std(scores)) if len(scores) > 1 else 0.0
        label = "Fake" if mean_fake >= 0.5 else "Real"
        confidence = mean_fake if mean_fake >= 0.5 else (1.0 - mean_fake)
        return {
            "label": label,
            "confidence": min(1.0, max(0.0, confidence)),
            "frames_analyzed": len(scores),
            "model": "EfficientNetB0 (video)",
            "std": std_fake,
        }
