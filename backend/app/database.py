from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .constants import DATA_DIR, DB_PATH


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure_database(path: Path = DB_PATH) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rewrite_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_text TEXT NOT NULL,
                raw_output TEXT NOT NULL,
                rewritten_text TEXT NOT NULL,
                platform TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                iterations INTEGER NOT NULL,
                warnings_json TEXT NOT NULL,
                nlp_applied INTEGER NOT NULL,
                nlp_style TEXT,
                diff_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


@contextmanager
def connect(path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    configure_database(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def get_setting(key: str) -> str | None:
    with connect() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_setting(key: str, value: str) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO settings(key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (key, value, utc_now_iso()),
        )


def insert_history(payload: dict) -> int:
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO rewrite_history (
                original_text, raw_output, rewritten_text, platform, provider, model, iterations,
                warnings_json, nlp_applied, nlp_style, diff_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["original_text"],
                payload["raw_output"],
                payload["rewritten_text"],
                payload["platform"],
                payload["provider"],
                payload["model"],
                payload["iterations"],
                json.dumps(payload["warnings"], ensure_ascii=False),
                1 if payload["nlp_applied"] else 0,
                payload.get("nlp_style"),
                json.dumps(payload["diff"], ensure_ascii=False),
                payload["created_at"],
            ),
        )
        return int(cur.lastrowid)


def list_history(limit: int = 50) -> list[sqlite3.Row]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT id, platform, provider, model, original_text, rewritten_text, created_at
            FROM rewrite_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return list(rows)


def get_history(record_id: int) -> sqlite3.Row | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM rewrite_history WHERE id = ?", (record_id,)).fetchone()
    return row


def delete_history(record_id: int) -> bool:
    with connect() as conn:
        cur = conn.execute("DELETE FROM rewrite_history WHERE id = ?", (record_id,))
        return cur.rowcount > 0

