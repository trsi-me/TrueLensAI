# -*- coding: utf-8 -*-
# Standalone Flask app: loads text + image models on disk and exposes /v1/* for TrueLens main app.
# Deploy on Fly.io, Railway, a VM, or a second Render service with more RAM; set TRUELENS_ML_API_KEY
# on both worker and main app (Authorization: Bearer <key>).
import os
import sys
import tempfile
import traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from flask import Flask, jsonify, request

from config import Config
from ml_models.model_loader import looks_like_git_lfs_pointer, resolve_image_model_path
from ml_models.text_detector import TextDetector

app = Flask(__name__)
_text_detector = None
_image_detector = None


def _auth_ok() -> bool:
    need = (os.environ.get("TRUELENS_ML_API_KEY") or "").strip()
    if not need:
        return True
    auth = (request.headers.get("Authorization") or "").strip()
    if auth == f"Bearer {need}":
        return True
    return (request.headers.get("X-TrueLens-ML-Key") or "").strip() == need


@app.before_request
def _require_auth_for_v1():
    if request.path.startswith("/v1/") and not _auth_ok():
        return jsonify({"error": "Unauthorized"}), 401


def load_models() -> None:
    global _text_detector, _image_detector
    _text_detector = None
    tp, vp = Config.TEXT_MODEL_PATH, Config.TFIDF_PATH
    if os.path.isfile(tp) and os.path.isfile(vp):
        if looks_like_git_lfs_pointer(tp) or looks_like_git_lfs_pointer(vp):
            print("ML worker: text models are Git LFS pointers; place real .pkl files.", file=sys.stderr)
        else:
            try:
                _text_detector = TextDetector(tp, vp)
            except Exception:
                print("ML worker: failed to load text models.", file=sys.stderr)
                traceback.print_exc()
                _text_detector = None

    _image_detector = None
    img_path = resolve_image_model_path()
    if img_path and looks_like_git_lfs_pointer(img_path):
        print("ML worker: image model path is an LFS pointer.", file=sys.stderr)
        img_path = None
    if img_path:
        try:
            from ml_models.image_detector import ImageDetector

            _image_detector = ImageDetector(img_path)
        except Exception:
            print("ML worker: failed to load image model.", file=sys.stderr)
            traceback.print_exc()
            _image_detector = None

    if not (os.environ.get("TRUELENS_ML_API_KEY") or "").strip():
        print(
            "ML worker: TRUELENS_ML_API_KEY is unset - use only on a private network or for local tests.",
            file=sys.stderr,
        )


@app.route("/health")
def health():
    return jsonify(
        {
            "ok": True,
            "text_loaded": bool(_text_detector and _text_detector.is_loaded()),
            "image_loaded": bool(_image_detector and _image_detector.is_loaded()),
        }
    )


@app.route("/v1/predict/text", methods=["POST"])
def predict_text():
    if _text_detector is None or not _text_detector.is_loaded():
        return jsonify({"error": "Text model not loaded on this worker"}), 503
    body = request.get_json(silent=True) or {}
    text = body.get("text") or ""
    try:
        out = _text_detector.predict(text)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(out)


@app.route("/v1/predict/image", methods=["POST"])
def predict_image():
    if _image_detector is None or not _image_detector.is_loaded():
        return jsonify({"error": "Image model not loaded on this worker"}), 503
    f = request.files.get("file") or request.files.get("image")
    if not f or not f.filename:
        return jsonify({"error": "Missing file"}), 400
    path = None
    try:
        suf = os.path.splitext(f.filename)[1] or ".bin"
        fd, path = tempfile.mkstemp(suffix=suf)
        os.close(fd)
        f.save(path)
        out = _image_detector.predict(path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass
    return jsonify(out)


load_models()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, threaded=True)
