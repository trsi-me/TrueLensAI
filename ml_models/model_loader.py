# -*- coding: utf-8 -*-
# Load ML models once at application startup.
import os
import sys
import threading
import traceback
from typing import Optional

from config import Config

_text_detector = None
_image_detector = None
_video_detector = None
_models_load_thread: Optional[threading.Thread] = None


def _looks_like_git_lfs_pointer(path: str) -> bool:
    try:
        if os.path.getsize(path) > 512:
            return False
        with open(path, "rb") as f:
            return f.read(40).startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def init_models() -> None:
    # Load text / image / video detectors when model files exist.
    global _text_detector, _image_detector, _video_detector
    from ml_models.text_detector import TextDetector
    from ml_models.image_detector import ImageDetector
    from ml_models.video_detector import VideoDetector

    _text_detector = None
    if os.path.isfile(Config.TEXT_MODEL_PATH) and os.path.isfile(Config.TFIDF_PATH):
        if _looks_like_git_lfs_pointer(Config.TEXT_MODEL_PATH) or _looks_like_git_lfs_pointer(
            Config.TFIDF_PATH
        ):
            print(
                "TrueLens: text_model.pkl or tfidf_vectorizer.pkl is a Git LFS pointer. "
                "Use render_git_lfs_pull.sh in build or PRETRAINED_MODELS_BASE_URL.",
                file=sys.stderr,
            )
        else:
            try:
                _text_detector = TextDetector(Config.TEXT_MODEL_PATH, Config.TFIDF_PATH)
            except Exception:
                print("TrueLens: failed to load text models.", file=sys.stderr)
                traceback.print_exc()
                _text_detector = None

    _image_detector = None
    if os.path.isfile(Config.IMAGE_MODEL_PATH):
        if _looks_like_git_lfs_pointer(Config.IMAGE_MODEL_PATH):
            print(
                "TrueLens: image_model.h5 is a Git LFS pointer (large file not on disk). "
                "On Render, run bash scripts/render_git_lfs_pull.sh in build, or set "
                "PRETRAINED_MODELS_BASE_URL / IMAGE_MODEL_DOWNLOAD_URL.",
                file=sys.stderr,
            )
        else:
            try:
                _image_detector = ImageDetector(Config.IMAGE_MODEL_PATH)
            except Exception:
                print("TrueLens: failed to load image model.", file=sys.stderr)
                traceback.print_exc()
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
            print("TrueLens: init_models() crashed.", file=sys.stderr)
            traceback.print_exc()

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
