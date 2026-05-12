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


def _table_columns(cur, table: str) -> List[str]:
    cur.execute("PRAGMA table_info(%s)" % table.replace('"', ""))
    return [str(r[1]) for r in cur.fetchall()]


def init_db() -> None:
    with _cursor() as cur:
        cur.execute(
            "CREATE TABLE IF NOT EXISTS users ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "email TEXT UNIQUE NOT NULL,"
            "password_hash TEXT NOT NULL,"
            "display_name TEXT,"
            "is_admin INTEGER NOT NULL DEFAULT 0,"
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        ucols = _table_columns(cur, "users")
        if "is_admin" not in ucols:
            cur.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0")
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
        cols = _table_columns(cur, "detection_history")
        if "user_id" not in cols:
            cur.execute(
                "ALTER TABLE detection_history ADD COLUMN user_id INTEGER "
                "REFERENCES users(id) ON DELETE CASCADE"
            )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_detection_history_user_id "
            "ON detection_history(user_id)"
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
        cur.execute(
            "INSERT OR IGNORE INTO app_stats (stat_key, stat_value) VALUES ('uncertain_detected', 0)"
        )
    ensure_default_admin_from_env()


def ensure_default_admin_from_env() -> None:
    # TRUELENS_DEFAULT_ADMIN_EMAIL + TRUELENS_DEFAULT_ADMIN_PASSWORD → create or promote admin (no password reset if user exists).
    email = (os.environ.get("TRUELENS_DEFAULT_ADMIN_EMAIL") or "").strip().lower()
    password = os.environ.get("TRUELENS_DEFAULT_ADMIN_PASSWORD") or ""
    if not email or not password:
        return
    from werkzeug.security import generate_password_hash

    row = get_user_by_email(email)
    if row:
        if not int(row.get("is_admin") or 0):
            with _cursor() as cur:
                cur.execute("UPDATE users SET is_admin = 1 WHERE id = ?", (int(row["id"]),))
        return
    dn = (os.environ.get("TRUELENS_DEFAULT_ADMIN_DISPLAY_NAME") or "").strip()[:120] or None
    if not dn:
        dn = "Administrator"
    h = generate_password_hash(password)
    create_user(email, h, dn, is_admin=1)


def create_user(
    email: str, password_hash: str, display_name: Optional[str], is_admin: int = 0
) -> int:
    em = email.strip().lower()
    dn = (display_name or "").strip()[:120] or None
    ia = 1 if int(is_admin) else 0
    with _cursor() as cur:
        cur.execute(
            "INSERT INTO users (email, password_hash, display_name, is_admin) VALUES (?, ?, ?, ?)",
            (em, password_hash, dn, ia),
        )
        return int(cur.lastrowid)


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    em = email.strip().lower()
    with _cursor() as cur:
        cur.execute(
            "SELECT id, email, password_hash, display_name, is_admin, created_at FROM users WHERE email = ?",
            (em,),
        )
        r = cur.fetchone()
    return dict(r) if r else None


def get_user_public(user_id: int) -> Optional[Dict[str, Any]]:
    with _cursor() as cur:
        cur.execute(
            "SELECT id, email, display_name, is_admin, created_at FROM users WHERE id = ?",
            (user_id,),
        )
        r = cur.fetchone()
    return dict(r) if r else None


def update_user_display_name(user_id: int, display_name: Optional[str]) -> None:
    dn = (display_name or "").strip()[:120] or None
    with _cursor() as cur:
        cur.execute("UPDATE users SET display_name = ? WHERE id = ?", (dn, user_id))


def update_user_password_hash(user_id: int, password_hash: str) -> None:
    with _cursor() as cur:
        cur.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (password_hash, user_id),
        )


def save_detection(
    detection_type: str,
    input_summary: Optional[str],
    label: str,
    score: float,
    model: Optional[str],
    time_ms: int,
    filename: Optional[str] = None,
    user_id: Optional[int] = None,
) -> int:
    summary = input_summary or ""
    if len(summary) > 200:
        summary = summary[:200]
    with _cursor() as cur:
        cur.execute(
            "INSERT INTO detection_history ("
            "detection_type, input_summary, result_label, confidence_score,"
            "model_used, processing_time_ms, file_name, user_id"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                detection_type,
                summary,
                label,
                float(score),
                model,
                int(time_ms),
                filename,
                user_id,
            ),
        )
        return int(cur.lastrowid)


def get_history_for_user(user_id: int, limit: int = 200) -> List[Dict[str, Any]]:
    with _cursor() as cur:
        cur.execute(
            "SELECT id, detection_type, input_summary, result_label, confidence_score,"
            "model_used, processing_time_ms, file_name, created_at, user_id "
            "FROM detection_history WHERE user_id = ? "
            "ORDER BY created_at DESC, id DESC LIMIT ?",
            (user_id, limit),
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def get_all_history(limit: int = 50) -> List[Dict[str, Any]]:
    with _cursor() as cur:
        cur.execute(
            "SELECT id, detection_type, input_summary, result_label, confidence_score,"
            "model_used, processing_time_ms, file_name, created_at, user_id "
            "FROM detection_history ORDER BY created_at DESC, id DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def get_stats() -> Dict[str, int]:
    with _cursor() as cur:
        cur.execute("SELECT stat_key, stat_value FROM app_stats")
        rows = cur.fetchall()
    out = {"total_scans": 0, "fake_detected": 0, "real_detected": 0, "uncertain_detected": 0}
    for r in rows:
        k = r["stat_key"]
        if k in out:
            out[k] = int(r["stat_value"])
    return out


def increment_stat(key: str) -> None:
    allowed = {"total_scans", "fake_detected", "real_detected", "uncertain_detected"}
    if key not in allowed:
        return
    with _cursor() as cur:
        cur.execute(
            "UPDATE app_stats SET stat_value = stat_value + 1, "
            "updated_at = CURRENT_TIMESTAMP WHERE stat_key = ?",
            (key,),
        )


def delete_history_record(record_id: int, user_id: Optional[int] = None) -> bool:
    with _cursor() as cur:
        if user_id is not None:
            cur.execute(
                "DELETE FROM detection_history WHERE id = ? AND user_id = ?",
                (record_id, user_id),
            )
        else:
            cur.execute("DELETE FROM detection_history WHERE id = ?", (record_id,))
        return cur.rowcount > 0


def clear_all_history(user_id: Optional[int] = None) -> None:
    with _cursor() as cur:
        if user_id is not None:
            cur.execute("DELETE FROM detection_history WHERE user_id = ?", (user_id,))
        else:
            cur.execute("DELETE FROM detection_history")


def count_users() -> int:
    with _cursor() as cur:
        cur.execute("SELECT COUNT(*) AS c FROM users")
        return int(cur.fetchone()["c"])


def list_users_admin(limit: int = 500) -> List[Dict[str, Any]]:
    with _cursor() as cur:
        cur.execute(
            "SELECT u.id, u.email, u.display_name, u.is_admin, u.created_at,"
            " (SELECT COUNT(*) FROM detection_history d WHERE d.user_id = u.id) AS scan_count "
            "FROM users u ORDER BY u.id DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]


def get_user_dashboard_counts(user_id: int) -> Dict[str, Any]:
    with _cursor() as cur:
        cur.execute(
            "SELECT detection_type, COUNT(*) AS c FROM detection_history "
            "WHERE user_id = ? GROUP BY detection_type",
            (user_id,),
        )
        rows = cur.fetchall()
    by_type = {"text": 0, "image": 0, "video": 0}
    total = 0
    for r in rows:
        t = r["detection_type"]
        c = int(r["c"])
        total += c
        if t in by_type:
            by_type[t] = c
    last_row = None
    with _cursor() as cur:
        cur.execute(
            "SELECT id, detection_type, result_label, confidence_score, created_at "
            "FROM detection_history WHERE user_id = ? ORDER BY created_at DESC, id DESC LIMIT 1",
            (user_id,),
        )
        last_row = cur.fetchone()
    last = dict(last_row) if last_row else None
    return {"total": total, "by_type": by_type, "last_scan": last}
