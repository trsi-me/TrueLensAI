# -*- coding: utf-8 -*-
import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from config import Config


def _get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(Config.DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(Config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def _cursor():
    conn = _get_connection()
    try:
        cur = conn.cursor()
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with _cursor() as cur:
        cur.execute(
            "CREATE TABLE IF NOT EXISTS detection_history ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "detection_type TEXT NOT NULL,"
            "input_summary TEXT,"
            "result_label TEXT NOT NULL,"
            "confidence_score REAL NOT NULL,"
            "model_used TEXT,"
            "processing_time_ms INTEGER,"
            "file_name TEXT,"
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS app_stats ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "stat_key TEXT UNIQUE NOT NULL,"
            "stat_value INTEGER DEFAULT 0,"
            "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        cur.execute(
            "INSERT OR IGNORE INTO app_stats (stat_key, stat_value) VALUES ('total_scans', 0)"
        )
        cur.execute(
            "INSERT OR IGNORE INTO app_stats (stat_key, stat_value) VALUES ('fake_detected', 0)"
        )
        cur.execute(
            "INSERT OR IGNORE INTO app_stats (stat_key, stat_value) VALUES ('real_detected', 0)"
        )


def save_detection(
    detection_type: str,
    input_summary: Optional[str],
    label: str,
    score: float,
    model: Optional[str],
    time_ms: int,
    filename: Optional[str] = None,
) -> int:
    summary = input_summary or ""
    if len(summary) > 200:
        summary = summary[:200]
    with _cursor() as cur:
        cur.execute(
            "INSERT INTO detection_history ("
            "detection_type, input_summary, result_label, confidence_score,"
            "model_used, processing_time_ms, file_name"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                detection_type,
                summary,
                label,
                float(score),
                model,
                int(time_ms),
                filename,
            ),
        )
        return int(cur.lastrowid)


def get_all_history(limit: int = 50) -> List[Dict[str, Any]]:
    with _cursor() as cur:
        cur.execute(
            "SELECT id, detection_type, input_summary, result_label, confidence_score,"
            "model_used, processing_time_ms, file_name, created_at "
            "FROM detection_history ORDER BY created_at DESC, id DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def get_stats() -> Dict[str, int]:
    with _cursor() as cur:
        cur.execute("SELECT stat_key, stat_value FROM app_stats")
        rows = cur.fetchall()
    out = {"total_scans": 0, "fake_detected": 0, "real_detected": 0}
    for r in rows:
        k = r["stat_key"]
        if k in out:
            out[k] = int(r["stat_value"])
    return out


def increment_stat(key: str) -> None:
    allowed = {"total_scans", "fake_detected", "real_detected"}
    if key not in allowed:
        return
    with _cursor() as cur:
        cur.execute(
            "UPDATE app_stats SET stat_value = stat_value + 1, "
            "updated_at = CURRENT_TIMESTAMP WHERE stat_key = ?",
            (key,),
        )


def delete_history_record(record_id: int) -> bool:
    with _cursor() as cur:
        cur.execute("DELETE FROM detection_history WHERE id = ?", (record_id,))
        return cur.rowcount > 0


def clear_all_history() -> None:
    with _cursor() as cur:
        cur.execute("DELETE FROM detection_history")
