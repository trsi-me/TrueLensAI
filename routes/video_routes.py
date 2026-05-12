# -*- coding: utf-8 -*-
import time

from flask import Blueprint, request, session

from config import Config
from database.db_handler import save_detection
from ml_models.model_loader import (
    get_image_detector,
    get_video_detector,
    wait_for_models_init,
)
from utils.analysis_presenter import (
    apply_uncertain_label,
    explanations_video,
    increment_stats_for_detection,
    probe_video_audio_streams,
)
from utils.auth import current_user_id
from utils.file_handler import remove_file_safe, save_upload_temp
from utils.history_auto import auto_save_detection_after_analyze
from utils.validators import validate_video_file

video_bp = Blueprint("video", __name__)


def _json(success: bool, data=None, error=None):
    from flask import jsonify

    return jsonify(
        {"success": success, "data": data if data is not None else {}, "error": error}
    )


@video_bp.route("/video")
def detect_video_page():
    from flask import render_template

    return render_template("detect_video.html")


@video_bp.route("/video/analyze", methods=["POST"])
def analyze_video():
    wait_for_models_init()
    vdet = get_video_detector()
    img_det = get_image_detector()
    if img_det is None or not img_det.is_loaded() or vdet is None:
        return _json(
            False,
            None,
            "Video model is not available (depends on the image model). Train the image model first.",
        )
    f = request.files.get("file") or request.files.get("video")
    if not f or not f.filename:
        return _json(False, None, "No file selected.")
    clen = request.content_length or 0
    ok, err = validate_video_file(f.filename, clen)
    if not ok:
        return _json(False, None, err)
    path = None
    safe_name = ""
    audio_info: dict = {}
    try:
        path, safe_name, _ext = save_upload_temp(f, Config.ALLOWED_VIDEO_EXTENSIONS)
        audio_info = probe_video_audio_streams(path)
        t0 = time.perf_counter()
        out = vdet.predict(path)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
    except Exception as e:
        if path:
            remove_file_safe(path)
        errno = getattr(e, "errno", None)
        msg_lower = str(e).lower()
        if errno == 28 or "no space left on device" in msg_lower:
            return _json(
                False,
                None,
                "Disk is full. Free up space (e.g. empty the recycle bin or temp folder) and try again.",
            )
        return _json(False, None, f"Video analysis failed: {str(e)}")
    finally:
        if path:
            remove_file_safe(path)
    model_label = out["label"]
    conf = float(out["confidence"])
    display_label, underlying = apply_uncertain_label(model_label, conf)
    frames = int(out.get("frames_analyzed", 0))
    std = float(out.get("std", 0.0))
    explanations = explanations_video(
        display_label, conf, underlying, frames, std, audio_info
    )
    increment_stats_for_detection(display_label)
    if audio_info.get("ffprobe_available"):
        if audio_info.get("has_audio") is True:
            audio_summary = "ffprobe: audio stream present in container."
        elif audio_info.get("has_audio") is False:
            audio_summary = "ffprobe: no audio stream in container."
        else:
            audio_summary = "ffprobe: could not determine audio presence."
    else:
        audio_summary = audio_info.get("note") or "Audio metadata not available."
    data = {
        "label": display_label,
        "model_label": underlying,
        "confidence": conf,
        "model": out.get("model"),
        "processing_time_ms": elapsed_ms,
        "frames_analyzed": frames,
        "std": std,
        "input_summary": safe_name[:200],
        "file_name": safe_name,
        "explanations": explanations,
        "audio": audio_info,
        "audio_summary": audio_summary,
    }
    hid = auto_save_detection_after_analyze(
        "video",
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
        "kind": "video",
        "label": display_label,
        "model_label": underlying,
        "confidence": conf,
        "model": out.get("model"),
        "processing_time_ms": elapsed_ms,
        "frames_analyzed": frames,
        "std": std,
        "input_summary": safe_name[:200],
        "file_name": safe_name,
        "explanations": explanations,
        "audio_summary": audio_summary,
    }
    return _json(True, data, None)


@video_bp.route("/video/save", methods=["POST"])
def save_video_result():
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
        "video",
        summary,
        label,
        conf,
        model,
        int(time_ms),
        fname or None,
        user_id=uid,
    )
    return _json(True, {"id": rid}, None)
