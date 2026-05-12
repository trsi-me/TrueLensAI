# -*- coding: utf-8 -*-
import os
import time

from flask import Blueprint, request, session

from database.db_handler import save_detection
from ml_models.model_loader import get_text_detector, wait_for_models_init
from utils.analysis_presenter import (
    apply_uncertain_label,
    explanations_text,
    increment_stats_for_detection,
)
from utils.auth import current_user_id
from utils.history_auto import auto_save_detection_after_analyze
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
    wait_for_models_init()
    det = get_text_detector()
    if det is None or not det.is_loaded():
        extra = ""
        if (os.environ.get("TRUELENS_ML_API_BASE_URL") or "").strip():
            extra = (
                " Remote ML is enabled (TRUELENS_ML_API_BASE_URL): ensure the worker is up, "
                "reachable from Render, and TRUELENS_ML_API_KEY matches on both sides."
            )
        return _json(
            False,
            None,
            "Text model is not available. Train the model and place the files in ml_models/saved_models,"
            " or configure a remote ML worker (see README §15.3)."
            + extra,
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
    model_label = out["label"]
    conf = float(out["confidence"])
    display_label, underlying = apply_uncertain_label(model_label, conf)
    explanations = explanations_text(text, display_label, conf, underlying)
    increment_stats_for_detection(display_label)
    data = {
        "label": display_label,
        "model_label": underlying,
        "confidence": conf,
        "model": out.get("model"),
        "processing_time_ms": elapsed_ms,
        "input_summary": summary,
        "explanations": explanations,
    }
    hid = auto_save_detection_after_analyze(
        "text", summary, display_label, conf, out.get("model"), elapsed_ms, None
    )
    if hid is not None:
        data["history_id"] = hid
    session["truelens_report"] = {
        "kind": "text",
        "label": display_label,
        "model_label": underlying,
        "confidence": conf,
        "model": out.get("model"),
        "processing_time_ms": elapsed_ms,
        "input_summary": summary,
        "explanations": explanations,
    }
    return _json(True, data, None)


@text_bp.route("/text/save", methods=["POST"])
def save_text_result():
    uid = current_user_id()
    if uid is None:
        return _json(False, None, "Please sign in to save results to your history."), 401
    payload = request.get_json(silent=True) or {}
    label = payload.get("label")
    confidence = payload.get("confidence")
    summary = payload.get("input_summary") or ""
    model = payload.get("model")
    time_ms = payload.get("processing_time_ms", 0)
    if label not in ("Fake", "Real", "Uncertain"):
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
        user_id=uid,
    )
    return _json(True, {"id": rid}, None)
