# -*- coding: utf-8 -*-
import os
import uuid

from config import Config


def ensure_upload_dir() -> None:
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)


def _upload_basename(filename: str) -> str:
    # Normalize slashes and drop any path segments (defense in depth).
    return os.path.basename((filename or "").replace("\\", "/"))


def save_upload_temp(file_storage, allowed_ext) -> tuple:
    # Save upload to temp path; returns (full_path, display_name, ext).
    # Extension is taken from the *original* filename. Werkzeug's secure_filename
    # strips non-ASCII (e.g. Arabic), which made names like "شعار.png" fail as "not allowed".
    ensure_upload_dir()
    if not file_storage or not file_storage.filename:
        raise ValueError("No file selected.")
    orig = file_storage.filename
    base = _upload_basename(orig)
    base = base.replace("\x00", "").strip()
    if not base:
        raise ValueError("No file selected.")
    ext = base.rsplit(".", 1)[-1].lower() if "." in base else ""
    if ext not in allowed_ext:
        raise ValueError("File type not allowed.")
    disk_name = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(Config.UPLOAD_FOLDER, disk_name)
    display = base[:200] if len(base) <= 200 else base[:197] + "..."
    file_storage.save(path)
    return path, display, ext


def remove_file_safe(path: str) -> None:
    if path and os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass
