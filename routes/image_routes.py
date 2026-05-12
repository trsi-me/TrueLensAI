# -*- coding: utf-8 -*-
import os
import time

from flask import Blueprint, request, session

from config import Config
from database.db_handler import save_detection
from ml_models.model_loader import get_image_detector, wait_for_models_init
from utils.analysis_presenter import (
    apply_uncertain_label,
    explanations_image,
    increment_stats_for_detection,
    jpeg_ela_data_uri,
)
from utils.auth import current_user_id
from utils.file_handler import remove_file_safe, save_upload_temp
from utils.history_auto import auto_save_detection_after_analyze
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


_IMAGE_MODEL_HELP = (
    "Image model is not available. Expected a real (non–Git-LFS-pointer) file at "
    "ml_models/saved_models/image_model.h5 or image_model.tflite, or under ml_models/ or ml/ "
    "with the same names, or set TRUELENS_IMAGE_MODEL_PATH to the full path. "
    "If you use only .h5, unset TRUELENS_IMAGE_ONLY_TFLITE or install TensorFlow (requirements.txt). "
    "If the server just started, wait for loading or set TRUELENS_EAGER_LOAD_MODELS=1. "
    "Or set TRUELENS_ML_API_BASE_URL to a remote ML worker (README §15.3)."
)


@image_bp.route("/image/analyze", methods=["POST"])
def analyze_image():
    wait_for_models_init()
    det = get_image_detector()
    if det is None or not det.is_loaded():
        return _json(False, None, _IMAGE_MODEL_HELP)
    f = request.files.get("file") or request.files.get("image")
    if not f or not f.filename:
        return _json(False, None, "No file selected.")
    clen = request.content_length or 0
    ok, err = validate_image_file(f.filename, clen)
    if not ok:
        return _json(False, None, err)
    path = None
    safe_name = ""
    ela_uri = None
    try:
        path, safe_name, _ext = save_upload_temp(f, Config.ALLOWED_IMAGE_EXTENSIONS)
        ela_uri = jpeg_ela_data_uri(path) if path else None
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
    model_label = out["label"]
    conf = float(out["confidence"])
    display_label, underlying = apply_uncertain_label(model_label, conf)
    ela_ok = bool(ela_uri)
    explanations = explanations_image(display_label, conf, underlying, ela_ok)
    increment_stats_for_detection(display_label)
    data = {
        "label": display_label,
        "model_label": underlying,
        "confidence": conf,
        "model": out.get("model"),
        "processing_time_ms": elapsed_ms,
        "input_summary": safe_name[:200],
        "file_name": safe_name,
        "explanations": explanations,
        "ela_image": ela_uri,
    }
    hid = auto_save_detection_after_analyze(
        "image",
        safe_name[:200],
        display_label,
        conf,
        out.get("model"),
        elapsed_ms,
        safe_name[:200],
    )
    if hid is not None:
        data["history_id"] = hid
    session["truelens_report"] = {
        "kind": "image",
        "label": display_label,
        "model_label": underlying,
        "confidence": conf,
        "model": out.get("model"),
        "processing_time_ms": elapsed_ms,
        "input_summary": safe_name[:200],
        "file_name": safe_name,
        "explanations": explanations,
        "ela_note": "ELA map was shown on the analysis page (not stored in this report)." if ela_ok else "ELA not available for this format.",
    }
    return _json(True, data, None)


@image_bp.route("/image/save", methods=["POST"])
def save_image_result():
    uid = current_user_id()
    if uid is None:
        return _json(False, None, "Please sign in to save results to your history."), 401
    payload = request.get_json(silent=True) or {}
    label = payload.get("label")
    confidence = payload.get("confidence")
    summary = payload.get("input_summary") or ""
    fname = payload.get("file_name")
    model = payload.get("model")
    time_ms = payload.get("processing_time_ms", 0)
    if label not in ("Fake", "Real", "Uncertain"):
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
        user_id=uid,
    )
    return _json(True, {"id": rid}, None)
