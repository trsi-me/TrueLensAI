# -*- coding: utf-8 -*-
import base64
import io
import os
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageChops, ImageEnhance

from config import Config
from utils.text_sensational import emotion_hit_count


def uncertain_band() -> Tuple[float, float]:
    lo = float(os.environ.get("TRUELENS_UNCERTAIN_LOW", str(Config.UNCERTAIN_CONFIDENCE_LOW)))
    hi = float(os.environ.get("TRUELENS_UNCERTAIN_HIGH", str(Config.UNCERTAIN_CONFIDENCE_HIGH)))
    if lo >= hi:
        lo, hi = 0.42, 0.58
    return lo, hi


def apply_uncertain_label(model_label: str, confidence: float) -> Tuple[str, str]:
    lo, hi = uncertain_band()
    ml = model_label if model_label in ("Fake", "Real") else "Fake"
    conf = float(confidence)
    if lo <= conf <= hi:
        return "Uncertain", ml
    return ml, ml


def explanations_text(text: str, display_label: str, confidence: float, model_label: str) -> List[str]:
    t = (text or "").lower()
    lines: List[str] = []
    lo, hi = uncertain_band()
    if display_label == "Uncertain":
        lines.append(
            f"The authenticity score sits on the fence (confidence {confidence:.0%} in the "
            f"{lo:.0%}–{hi:.0%} uncertain band). Treat the article as inconclusive from wording alone."
        )
    hits = emotion_hit_count(text)
    if hits >= 2:
        lines.append(
            "Several sensational or emotionally loaded phrases read like tabloid or manipulative "
            "headline style—often seen in unreliable or exaggerated news copy."
        )
    elif hits == 1:
        lines.append(
            "Some emotionally loaded wording in the piece (weak signal on its own; combine with other checks)."
        )
    url_count = len(re.findall(r"https?://", text or ""))
    if url_count == 0 and len((text or "").split()) > 80:
        lines.append(
            "Long article body with no inline links—many trustworthy reports still cite sources; "
            "missing URLs are only a soft hint about form, not proof of falsehood."
        )
    if re.search(r"[!?]{3,}", text or ""):
        lines.append(
            "Repeated punctuation (!!!, ???) often matches shouty or sensational headline tone rather than wire-style reporting."
        )
    caps_words = re.findall(r"\b[A-Z]{4,}\b", text or "")
    if len(caps_words) >= 3:
        lines.append(
            "Multiple ALL-CAPS tokens suggest attention-grabbing formatting more typical of viral posts than calm newsroom style."
        )
    if display_label != "Uncertain":
        lines.append(
            f"Based on vocabulary and phrasing patterns (trained on labeled news), the estimate leans "
            f"toward “{display_label}” (raw side: {model_label}) with score {confidence:.0%}. "
            "This reflects writing style, not verified facts."
        )
    else:
        lines.append(
            f"Before the uncertain band: raw vote {model_label} at {confidence:.0%}—ambiguous for this article’s wording."
        )
    if not lines:
        lines.append(
            "No strong stylistic red flags beyond the score—still verify claims, dates, and primary sources outside this tool."
        )
    lines.append(
        "Important: this is not live fact-checking; it does not read the internet or confirm events—"
        "only patterns from the text similar to fake vs authentic news in the training data."
    )
    return lines[:8]


def explanations_image(
    display_label: str, confidence: float, model_label: str, ela_available: bool
) -> List[str]:
    lines: List[str] = []
    lo, hi = uncertain_band()
    if display_label == "Uncertain":
        lines.append(
            f"Confidence {confidence:.0%} is in the uncertain band ({lo:.0%}–{hi:.0%}); avoid a hard verdict without expert review."
        )
    lines.append("The CNN examines global image statistics (textures, edges) typical of edited or synthetic content.")
    if ela_available:
        lines.append(
            "JPEG-style Error Level Analysis (ELA) map below highlights regions with unusual compression — "
            "often worth inspecting manually (not a definitive proof on its own). Applies to .jpg, .jpeg, .jfif, .jpe."
        )
    else:
        lines.append("ELA map is only generated for JPEG uploads (.jpg, .jpeg, .jfif, .jpe); other formats skip this forensic view.")
    if display_label not in ("Uncertain",):
        lines.append(f"Model vote: {model_label} at {confidence:.0%} confidence.")
    else:
        lines.append(f"Raw model side: {model_label} at {confidence:.0%} before applying the uncertain band.")
    return lines[:8]


def explanations_video(
    display_label: str,
    confidence: float,
    model_label: str,
    frames: int,
    std: float,
    audio: Dict[str, Any],
) -> List[str]:
    lines: List[str] = []
    lo, hi = uncertain_band()
    if display_label == "Uncertain":
        lines.append(
            f"Frame-aggregated score is ambiguous (confidence {confidence:.0%} in {lo:.0%}–{hi:.0%} band)."
        )
    lines.append(f"Analyzed {frames} sampled frame(s); score dispersion (std) ≈ {float(std):.3f}.")
    lines.append("Video verdict follows the same image detector on frames — lip-sync / voice deepfakes are not scored separately.")
    if audio.get("ffprobe_available"):
        if audio.get("has_audio") is True:
            lines.append("An audio stream is present; automatic voice authenticity was not evaluated (add dedicated audio models in future work).")
        elif audio.get("has_audio") is False:
            lines.append("No audio stream detected in the container — voice-based deepfake checks are not applicable.")
    else:
        lines.append(audio.get("note") or "Install FFmpeg/ffprobe on the server to log basic audio-stream metadata alongside video.")
    if display_label not in ("Uncertain",):
        lines.append(f"Aggregated label: {model_label} at {confidence:.0%}.")
    else:
        lines.append(f"Raw aggregated side: {model_label} at {confidence:.0%} before uncertain banding.")
    return lines[:10]


def jpeg_ela_data_uri(image_path: str, quality: int = 90) -> Optional[str]:
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in (".jpg", ".jpeg", ".jfif", ".jpe"):
        return None
    try:
        im = Image.open(image_path).convert("RGB")
    except OSError:
        return None
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")
    resaved = resaved.resize(im.size)
    ela = ImageChops.difference(im, resaved)
    extrema = ela.getextrema()
    max_diff = max(e[1] for e in extrema) or 1
    scale = 255.0 / max_diff
    ela = ImageEnhance.Brightness(ela).enhance(scale)
    out = io.BytesIO()
    ela.save(out, format="PNG")
    b64 = base64.standard_b64encode(out.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def probe_video_audio_streams(video_path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "ffprobe_available": False,
        "has_audio": None,
        "note": "",
    }
    exe = shutil.which("ffprobe")
    if not exe:
        out["note"] = "ffprobe not found in PATH — install FFmpeg to enable audio stream metadata."
        return out
    try:
        proc = subprocess.run(
            [
                exe,
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        out["note"] = f"ffprobe failed: {e}"
        return out
    out["ffprobe_available"] = True
    txt = (proc.stdout or "").strip()
    if proc.returncode != 0 and not txt:
        out["has_audio"] = None
        out["note"] = (proc.stderr or "ffprobe returned no audio stream info.")[:200]
        return out
    out["has_audio"] = bool(txt)
    out["note"] = "Audio stream listing from ffprobe (codec presence only)."
    return out


def increment_stats_for_detection(display_label: str) -> None:
    from database.db_handler import increment_stat

    increment_stat("total_scans")
    if display_label == "Uncertain":
        increment_stat("uncertain_detected")
    elif display_label == "Fake":
        increment_stat("fake_detected")
    else:
        increment_stat("real_detected")
