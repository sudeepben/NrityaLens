from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                file_name TEXT NOT NULL,
                dance_form TEXT NOT NULL,
                detected_label TEXT,
                confidence REAL NOT NULL,
                posture_score REAL NOT NULL,
                symmetry_score REAL NOT NULL,
                pose_match_score REAL NOT NULL,
                feedback TEXT NOT NULL
            )
            """
        )


def save_analysis(db_path: Path, result: dict[str, Any]) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO analyses (
                created_at, file_name, dance_form, detected_label, confidence,
                posture_score, symmetry_score, pose_match_score, feedback
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                result["file_name"],
                result["dance_form"],
                result.get("detected_label"),
                float(result["confidence"]),
                float(result["posture_score"]),
                float(result["symmetry_score"]),
                float(result["pose_match_score"]),
                "\n".join(result["feedback"]),
            ),
        )


def recent_analyses(db_path: Path, limit: int = 10) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT created_at, file_name, dance_form, detected_label, confidence,
                   posture_score, symmetry_score, pose_match_score, feedback
            FROM analyses
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]
