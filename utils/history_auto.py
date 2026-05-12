# -*- coding: utf-8 -*-
import os
import sys
import traceback
from typing import Optional

from database.db_handler import save_detection
from utils.auth import current_user_id


def auto_save_detection_after_analyze(
    detection_type: str,
    input_summary: Optional[str],
    label: str,
    confidence: float,
    model: Optional[str],
    time_ms: int,
    filename: Optional[str] = None,
) -> Optional[int]:
    # TRUELENS_DISABLE_AUTO_HISTORY=1 → no insert; user uses Save only.
    if (os.environ.get("TRUELENS_DISABLE_AUTO_HISTORY") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return None
    uid = current_user_id()
    if uid is None:
        return None
    try:
        return save_detection(
            detection_type,
            input_summary,
            label,
            confidence,
            model,
            int(time_ms),
            filename,
            user_id=uid,
        )
    except Exception:
        print("TrueLens: auto-save to detection_history failed.", file=sys.stderr)
        traceback.print_exc()
        return None
