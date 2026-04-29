# -*- coding: utf-8 -*-
from config import Config


def validate_text_input(text: str) -> tuple:
    # Returns (ok, error_message).
    if text is None:
        return False, "Text is empty."
    s = text.strip()
    if len(s) < 20:
        return False, "Please enter a news text of at least 20 characters."
    if len(s) > 50000:
        return False, "Text is too long (maximum 50000 characters)."
    return True, ""


def validate_image_file(filename: str, content_length: int) -> tuple:
    if not filename:
        return False, "No image selected."
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in Config.ALLOWED_IMAGE_EXTENSIONS:
        return False, "Image format not supported."
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
