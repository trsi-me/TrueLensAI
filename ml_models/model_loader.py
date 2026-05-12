# -*- coding: utf-8 -*-
# Load ML models once at application startup.
import os
import sys
import threading
import traceback
from typing import List, Optional

from config import Config

_text_detector = None
_image_detector = None
_video_detector = None
_models_load_thread: Optional[threading.Thread] = None


def _unique_paths(*paths: str) -> List[str]:
    seen = set()
    out: List[str] = []
    for p in paths:
        if not p or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _image_tflite_candidates() -> List[str]:
    base = Config.BASE_DIR
    return _unique_paths(
        Config.IMAGE_MODEL_TFLITE_PATH,
        os.path.join(base, "ml_models", "image_model.tflite"),
        os.path.join(base, "ml", "image_model.tflite"),
    )


def _image_h5_candidates() -> List[str]:
    base = Config.BASE_DIR
    return _unique_paths(
        Config.IMAGE_MODEL_PATH,
        os.path.join(base, "ml_models", "image_model.h5"),
        os.path.join(base, "ml", "image_model.h5"),
    )


def _first_real_model_file(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.isfile(p) and not _looks_like_git_lfs_pointer(p):
            return p
    return None


def _resolve_image_model_path() -> Optional[str]:
    # TFLite first unless TRUELENS_IMAGE_PREFER_H5; TRUELENS_IMAGE_ONLY_TFLITE blocks .h5.
    override = (os.environ.get("TRUELENS_IMAGE_MODEL_PATH") or "").strip()
    if override:
        if os.path.isfile(override) and not _looks_like_git_lfs_pointer(override):
            return override
    tflite_only = os.environ.get("TRUELENS_IMAGE_ONLY_TFLITE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    prefer_h5 = os.environ.get("TRUELENS_IMAGE_PREFER_H5", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if prefer_h5:
        return _first_real_model_file(_image_h5_candidates())
    p = _first_real_model_file(_image_tflite_candidates())
    if p is not None:
        return p
    if tflite_only:
        return None
    return _first_real_model_file(_image_h5_candidates())


def _looks_like_git_lfs_pointer(path: str) -> bool:
    try:
        if os.path.getsize(path) > 512:
            return False
        with open(path, "rb") as f:
            return f.read(40).startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def looks_like_git_lfs_pointer(path: str) -> bool:
    return _looks_like_git_lfs_pointer(path)


def resolve_image_model_path() -> Optional[str]:
    return _resolve_image_model_path()


def init_models() -> None:
    # Load text / image / video detectors when model files exist.
    global _text_detector, _image_detector, _video_detector
    from ml_models.text_detector import TextDetector
    from ml_models.video_detector import VideoDetector

    remote_url = (os.environ.get("TRUELENS_ML_API_BASE_URL") or "").strip()
    if remote_url:
        remote_url = remote_url.rstrip("/")
        api_key = (os.environ.get("TRUELENS_ML_API_KEY") or "").strip()
        try:
            from ml_models.remote_detectors import RemoteImageDetector, RemoteTextDetector

            _text_detector = RemoteTextDetector(remote_url, api_key)
            _image_detector = RemoteImageDetector(remote_url, api_key)
            _video_detector = VideoDetector(_image_detector)
        except Exception:
            print("TrueLens: failed to configure remote ML API clients.", file=sys.stderr)
            traceback.print_exc()
            _text_detector = None
            _image_detector = None
            _video_detector = None
        return

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
    img_path = _resolve_image_model_path()
    if img_path is None:
        if os.path.isfile(Config.IMAGE_MODEL_PATH) and _looks_like_git_lfs_pointer(
            Config.IMAGE_MODEL_PATH
        ):
            print(
                "TrueLens: image_model.h5 is a Git LFS pointer (large file not on disk). "
                "On Render, run bash scripts/render_git_lfs_pull.sh in build, or set "
                "PRETRAINED_MODELS_BASE_URL / IMAGE_MODEL_DOWNLOAD_URL, or add image_model.tflite.",
                file=sys.stderr,
            )
        elif os.path.isfile(Config.IMAGE_MODEL_TFLITE_PATH) and _looks_like_git_lfs_pointer(
            Config.IMAGE_MODEL_TFLITE_PATH
        ):
            print(
                "TrueLens: image_model.tflite is a Git LFS pointer; run git lfs pull in build.",
                file=sys.stderr,
            )
    else:
        try:
            from ml_models.image_detector import ImageDetector

            _image_detector = ImageDetector(img_path)
        except Exception:
            print("TrueLens: failed to load image model.", file=sys.stderr)
            traceback.print_exc()
            _image_detector = None

    _video_detector = None
    if _image_detector is not None and _image_detector.is_loaded():
        _video_detector = VideoDetector(_image_detector)


def start_models_loading_thread() -> None:
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


def wait_for_models_init(timeout: Optional[float] = None) -> None:
    # Join background loader (skipped when TRUELENS_EAGER_LOAD_MODELS loaded sync).
    global _models_load_thread
    if timeout is None:
        raw = (os.environ.get("TRUELENS_MODEL_LOAD_TIMEOUT_SEC") or "").strip()
        try:
            timeout = float(raw) if raw else 600.0
        except ValueError:
            timeout = 600.0
    t = _models_load_thread
    if t is not None and t.is_alive():
        t.join(timeout=timeout)


def get_text_detector():
    return _text_detector


def get_image_detector():
    return _image_detector


def get_video_detector():
    return _video_detector
