# -*- coding: utf-8 -*-
import re
from typing import Optional

from config import Config

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def validate_email(email: str) -> tuple:
    if email is None:
        return False, "Email is required."
    s = email.strip().lower()
    if len(s) < 5 or len(s) > 254:
        return False, "Invalid email address."
    if not _EMAIL_RE.match(s):
        return False, "Invalid email address."
    return True, ""


def validate_password_for_register(password: str) -> tuple:
    if password is None or len(password) < 8:
        return False, "Password must be at least 8 characters."
    if len(password) > 256:
        return False, "Password is too long."
    return True, ""


def validate_display_name(name: Optional[str]) -> tuple:
    if name is None or not str(name).strip():
        return True, ""
    s = str(name).strip()
    if len(s) > 120:
        return False, "Display name is too long (max 120 characters)."
    return True, ""


def validate_text_input(text: str) -> tuple:
    # Returns (ok, error_message). Expects pasted news article / wire copy for authenticity-style scoring.
    if text is None:
        return False, "Paste the news article to analyze (headline + lead paragraph is best)."
    s = text.strip()
    if len(s) < 20:
        return False, "Please paste at least 20 characters of the news piece (headline and first sentences work best)."
    if len(s) > 50000:
        return False, "Article is too long (maximum 50000 characters)."
    return True, ""


def validate_image_file(filename: str, content_length: int) -> tuple:
    if not filename:
        return False, "No image selected."
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in Config.ALLOWED_IMAGE_EXTENSIONS:
        return (
            False,
            "Image format not supported. Use common raster types (e.g. PNG, JPEG, WebP, GIF, BMP, TIFF, ICO, HEIC) — not vector-only files like SVG.",
        )
    if content_length and content_length > Config.MAX_IMAGE_UPLOAD:
        return False, "Image size exceeds 20 MB."
    return True, ""


def validate_video_file(filename: str, content_length: int) -> tuple:
    if not filename:
        return False, "No video file selected."
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in Config.ALLOWED_VIDEO_EXTENSIONS:
        return False, "Video format not supported."
    if content_length and content_length > Config.MAX_VIDEO_UPLOAD:
        return False, "Video size exceeds 100 MB."
    return True, ""
