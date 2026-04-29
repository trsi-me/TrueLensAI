# -*- coding: utf-8 -*-
# Dataclass for detection_history rows (optional use in code).
from dataclasses import dataclass
from typing import Optional


@dataclass
class DetectionRecord:
    id: int
    detection_type: str
    input_summary: Optional[str]
    result_label: str
    confidence_score: float
    model_used: Optional[str]
    processing_time_ms: int
    file_name: Optional[str]
    created_at: str


DETECTION_TYPES = ("text", "image", "video")

LABEL_FAKE = "Fake"
LABEL_REAL = "Real"
