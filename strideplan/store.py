"""SQLite log of served rankings, user feedback and completed sessions."""

import json
import sqlite3
import threading
import uuid
from pathlib import Path


ACTIONS = ("accepted", "rejected", "completed")
EFFORTS = ("too_easy", "ok", "too_hard")
EFFORT_TARGET = {"too_easy": 0.0, "ok": 0.5, "too_hard": 1.0}

SCHEMA = """
CREATE TABLE IF NOT EXISTS recommendations (
    rec_id TEXT PRIMARY KEY,
    user_id TEXT,
    created_at REAL NOT NULL,
    model_version TEXT,
    strategy TEXT NOT NULL,
    max_duration_minutes REAL NOT NULL,
    items TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
    rec_id TEXT NOT NULL REFERENCES recommendations(rec_id),
    user_id TEXT,
    template_id INTEGER NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('accepted', 'rejected', 'completed')),
    effort TEXT CHECK (effort IS NULL OR effort IN ('too_easy', 'ok', 'too_hard')),
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    workout_id TEXT NOT NULL,
    start_timestamp REAL NOT NULL,
    duration_seconds REAL NOT NULL,
    avg_speed_mps REAL,
    avg_heart_rate_bpm REAL,
    rec_id TEXT REFERENCES recommendations(rec_id),
    created_at REAL NOT NULL,
    UNIQUE (user_id, workout_id)
);
CREATE INDEX IF NOT EXISTS feedback_user ON feedback(user_id, created_at);
CREATE INDEX IF NOT EXISTS sessions_user ON sessions(user_id, start_timestamp);
"""


class FeedbackError(ValueError):
    """Feedback that does not match a served recommendation."""


class Store:
    """Thread-safe for the FastAPI worker pool. Use ':memory:' for tests and simulations."""

    def __init__(self, path):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path), check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(SCHEMA)
        self.lock = threading.Lock()

    def close(self):
        self.connection.close()

    def log_recommendation(self, user_id, created_at, model_version, strategy, max_duration_minutes, items):
        """items: dicts with position, template_id, score and explored. Returns the new rec_id."""
        rec_id = uuid.uuid4().hex
        with self.lock, self.connection:
            self.connection.execute(
                "INSERT INTO recommendations VALUES (?, ?, ?, ?, ?, ?, ?)",
                (rec_id, user_id, created_at, model_version, strategy, max_duration_minutes,
                 json.dumps(items, allow_nan=False)))
        return rec_id

    def recommendation(self, rec_id):
        with self.lock:
            row = self.connection.execute("SELECT * FROM recommendations WHERE rec_id = ?", (rec_id,)).fetchone()
        return None if row is None else {**dict(row), "items": json.loads(row["items"])}

    def add_feedback(self, rec_id, template_id, action, effort, created_at):
        if action not in ACTIONS:
            raise FeedbackError(f"action must be one of {ACTIONS}")
        if effort is not None and (effort not in EFFORTS or action != "completed"):
            raise FeedbackError("effort is only given for a completed workout, as too_easy, ok or too_hard")
        served = self.recommendation(rec_id)
        if served is None:
            raise FeedbackError("Unknown rec_id")
        if template_id not in {item["template_id"] for item in served["items"]}:
            raise FeedbackError("This template was not part of the recommendation")
        if created_at < served["created_at"]:
            raise FeedbackError("Feedback cannot precede its recommendation")
        with self.lock, self.connection:
            cursor = self.connection.execute(
                "INSERT INTO feedback (rec_id, user_id, template_id, action, effort, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (rec_id, served["user_id"], template_id, action, effort, created_at))
        return cursor.lastrowid

    def add_session(self, user_id, session, created_at, rec_id=None):
        """session uses the processed-CSV field names. A repeated workout_id is ignored."""
        workout_id = session.get("workout_id") or f"logged-{uuid.uuid4().hex}"
        with self.lock, self.connection:
            cursor = self.connection.execute(
                "INSERT OR IGNORE INTO sessions (user_id, workout_id, start_timestamp, duration_seconds, avg_speed_mps, "
                "avg_heart_rate_bpm, rec_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, workout_id, session["start_timestamp"], session["duration_seconds"],
                 session.get("avg_speed_mps"), session.get("avg_heart_rate_bpm"), rec_id, created_at))
        return cursor.rowcount == 1

    def sessions(self, user_id=None):
        query = "SELECT * FROM sessions" + (" WHERE user_id = ?" if user_id is not None else "")
        query += " ORDER BY start_timestamp, workout_id"
        with self.lock:
            rows = self.connection.execute(query, () if user_id is None else (user_id,)).fetchall()
        return [{**dict(row), "split": "logged"} for row in rows]

    def feedback(self, user_id=None, since=None, until=None):
        """Feedback joined with the served model version and the template's served position."""
        clauses, values = [], []
        for clause, value in (("f.user_id = ?", user_id), ("f.created_at >= ?", since), ("f.created_at < ?", until)):
            if value is not None:
                clauses.append(clause)
                values.append(value)
        query = ("SELECT f.*, r.created_at AS served_at, r.model_version, r.items FROM feedback f "
                 "JOIN recommendations r USING (rec_id)")
        query += (" WHERE " + " AND ".join(clauses) if clauses else "") + " ORDER BY f.created_at, f.feedback_id"
        with self.lock:
            rows = self.connection.execute(query, values).fetchall()
        result = []
        for row in rows:
            row = dict(row)
            items = json.loads(row.pop("items"))
            row["position"] = next(item["position"] for item in items if item["template_id"] == row["template_id"])
            result.append(row)
        return result

    def counts(self):
        with self.lock:
            return {table: self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    for table in ("recommendations", "feedback", "sessions")}
