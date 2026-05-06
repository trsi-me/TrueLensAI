# -*- coding: utf-8 -*-
# Load ML models once at application startup.
import os
import threading
from typing import Optional

from config import Config

_text_detector = None
_image_detector = None
_video_detector = None
_models_load_thread: Optional[threading.Thread] = None


def init_models() -> None:
    # Load text / image / video detectors when model files exist.
    global _text_detector, _image_detector, _video_detector
    from ml_models.text_detector import TextDetector
    from ml_models.image_detector import ImageDetector
    from ml_models.video_detector import VideoDetector

    _text_detector = None
    if os.path.isfile(Config.TEXT_MODEL_PATH) and os.path.isfile(Config.TFIDF_PATH):
        try:
            _text_detector = TextDetector(Config.TEXT_MODEL_PATH, Config.TFIDF_PATH)
        except Exception:
            _text_detector = None

    _image_detector = None
    if os.path.isfile(Config.IMAGE_MODEL_PATH):
        try:
            _image_detector = ImageDetector(Config.IMAGE_MODEL_PATH)
        except Exception:
            _image_detector = None

    _video_detector = None
    if _image_detector is not None and _image_detector.is_loaded():
        _video_detector = VideoDetector(_image_detector)


def start_models_loading_thread() -> None:
    """Load models in a background thread so Gunicorn can bind and accept traffic immediately."""
    global _models_load_thread
    if _models_load_thread is not None and _models_load_thread.is_alive():
        return

    def _run() -> None:
        try:
            init_models()
        except Exception:
            pass

    _models_load_thread = threading.Thread(
        target=_run, daemon=True, name="truelens-ml-loader"
    )
    _models_load_thread.start()


def get_text_detector():
    return _text_detector


def get_image_detector():
    return _image_detector


def get_video_detector():
    return _video_detector
