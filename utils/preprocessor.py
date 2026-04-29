# -*- coding: utf-8 -*-
# Light text helpers (complements text_detector.clean_text).
import re
import unicodedata


def normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def clip_for_summary(text: str, max_len: int = 200) -> str:
    t = normalize_whitespace(text)
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"
