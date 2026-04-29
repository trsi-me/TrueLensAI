# -*- coding: utf-8 -*-
import time

from flask import Blueprint, request

from config import Config
from database.db_handler import increment_stat, save_detection
from ml_models.model_loader import get_image_detector
from utils.file_handler import remove_file_safe, save_upload_temp
from utils.validators import validate_image_file

image_bp = Blueprint("image", __name__)


def _json(success: bool, data=None, error=None):
    from flask import jsonify

    return jsonify(
        {"success": success, "data": data if data is not None else {}, "error": error}
    )


@image_bp.route("/image")
def detect_image_page():
    from flask import render_template

    return render_template("detect_image.html")


@image_bp.route("/image/analyze", methods=["POST"])
def analyze_image():
    det = get_image_detector()
    if det is None or not det.is_loaded():
        return _json(
            False,
            None,
            "Image model is not available. Train the model and save image_model.h5 in the specified folder.",
        )
    f = request.files.get("file") or request.files.get("image")
    if not f or not f.filename:
        return _json(False, None, "No file selected.")
    clen = request.content_length or 0
    ok, err = validate_image_file(f.filename, clen)
    if not ok:
        return _json(False, None, err)
    path = None
    safe_name = ""
    try:
        path, safe_name, _ext = save_upload_temp(f, Config.ALLOWED_IMAGE_EXTENSIONS)
        t0 = time.perf_counter()
        out = det.predict(path)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
    except Exception as e:
        if path:
            remove_file_safe(path)
        return _json(False, None, f"Image analysis failed: {str(e)}")
    finally:
        if path:
            remove_file_safe(path)
    increment_stat("total_scans")
    increment_stat("fake_detected" if out["label"] == "Fake" else "real_detected")
    return _json(
        True,
        {
            "label": out["label"],
            "confidence": out["confidence"],
            "model": out.get("model"),
            "processing_time_ms": elapsed_ms,
            "input_summary": safe_name[:200],
            "file_name": safe_name,
        },
        None,
    )


@image_bp.route("/image/save", methods=["POST"])
def save_image_result():
    payload = request.get_json(silent=True) or {}
    label = payload.get("label")
    confidence = payload.get("confidence")
    summary = payload.get("input_summary") or ""
    fname = payload.get("file_name")
    model = payload.get("model")
    time_ms = payload.get("processing_time_ms", 0)
    if label not in ("Fake", "Real"):
        return _json(False, None, "Invalid result.")
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        return _json(False, None, "Invalid confidence value.")
    rid = save_detection(
        "image",
        summary,
        label,
        conf,
        model,
        int(time_ms),
        fname or None,
    )
    return _json(True, {"id": rid}, None)
