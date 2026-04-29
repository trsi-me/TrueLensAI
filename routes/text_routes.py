# -*- coding: utf-8 -*-
import time

from flask import Blueprint, request

from database.db_handler import increment_stat, save_detection
from ml_models.model_loader import get_text_detector
from utils.preprocessor import clip_for_summary
from utils.validators import validate_text_input

text_bp = Blueprint("text", __name__)


def _json(success: bool, data=None, error=None):
    from flask import jsonify

    return jsonify(
        {"success": success, "data": data if data is not None else {}, "error": error}
    )


@text_bp.route("/text")
def detect_text_page():
    from flask import render_template

    return render_template("detect_text.html")


@text_bp.route("/text/analyze", methods=["POST"])
def analyze_text():
    det = get_text_detector()
    if det is None or not det.is_loaded():
        return _json(
            False,
            None,
            "Text model is not available. Train the model and place the files in ml_models/saved_models.",
        )
    payload = request.get_json(silent=True) or {}
    text = payload.get("text") or request.form.get("text") or ""
    ok, err = validate_text_input(text)
    if not ok:
        return _json(False, None, err)
    t0 = time.perf_counter()
    try:
        out = det.predict(text)
    except Exception as e:
        return _json(False, None, f"Prediction failed: {str(e)}")
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    summary = clip_for_summary(text, 200)
    increment_stat("total_scans")
    increment_stat("fake_detected" if out["label"] == "Fake" else "real_detected")
    return _json(
        True,
        {
            "label": out["label"],
            "confidence": out["confidence"],
            "model": out.get("model"),
            "processing_time_ms": elapsed_ms,
            "input_summary": summary,
        },
        None,
    )


@text_bp.route("/text/save", methods=["POST"])
def save_text_result():
    # Persist analysis result to detection_history (after UI shows it).
    payload = request.get_json(silent=True) or {}
    label = payload.get("label")
    confidence = payload.get("confidence")
    summary = payload.get("input_summary") or ""
    model = payload.get("model")
    time_ms = payload.get("processing_time_ms", 0)
    if label not in ("Fake", "Real"):
        return _json(False, None, "Invalid result.")
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        return _json(False, None, "Invalid confidence value.")
    rid = save_detection(
        "text",
        summary,
        label,
        conf,
        model,
        int(time_ms),
        None,
    )
    return _json(True, {"id": rid}, None)
