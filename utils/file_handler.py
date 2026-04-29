# -*- coding: utf-8 -*-
import os
import uuid

from werkzeug.utils import secure_filename

from config import Config


def ensure_upload_dir() -> None:
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)


def save_upload_temp(file_storage, allowed_ext: set) -> tuple:
    # Save upload to temp path; returns (full_path, safe_name, ext).
    ensure_upload_dir()
    if not file_storage or not file_storage.filename:
        raise ValueError("No file selected.")
    orig = file_storage.filename
    safe = secure_filename(orig)
    if not safe:
        safe = "upload"
    ext = safe.rsplit(".", 1)[-1].lower() if "." in safe else ""
    if ext not in allowed_ext:
        raise ValueError("File type not allowed.")
    unique = f"{uuid.uuid4().hex}_{safe}"
    path = os.path.join(Config.UPLOAD_FOLDER, unique)
    file_storage.save(path)
    return path, safe, ext


def remove_file_safe(path: str) -> None:
    if path and os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass
