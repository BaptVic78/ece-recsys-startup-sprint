"""SQLite storage for runners, their declared race, logged sessions and every interaction.

`/log_interaction` writes one row in `interactions` per event. A completed session is also
upserted into `sessions`, which is the history the recommender reads. Served recommendations
are stored too, so an interaction can refer to the list it answers.
"""

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

INTERACTION_TYPES = ("completed", "missed", "skipped")
EFFORTS = ("too_easy", "ok", "too_hard")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    race_date TEXT,
    goal_minutes REAL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    user_id TEXT NOT NULL REFERENCES users(user_id),
    date TEXT NOT NULL,
    km REAL NOT NULL CHECK (km > 0),
    minutes REAL NOT NULL CHECK (minutes > 0),
    source TEXT NOT NULL,
    PRIMARY KEY (user_id, date)
);
CREATE TABLE IF NOT EXISTS recommendations (
    rec_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    target_date TEXT NOT NULL,
    model TEXT NOT NULL,
    items TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS interactions (
    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    type TEXT NOT NULL CHECK (type IN ('completed', 'missed', 'skipped')),
    date TEXT NOT NULL,
    rec_id TEXT REFERENCES recommendations(rec_id),
    item INTEGER,
    km REAL,
    minutes REAL,
    effort TEXT CHECK (effort IS NULL OR effort IN ('too_easy', 'ok', 'too_hard')),
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS interactions_user ON interactions(user_id, date);
"""


class InteractionError(ValueError):
    """An interaction that does not match the stored state."""


class Store:
    """Thread-safe for FastAPI's worker threads. Use ':memory:' in tests."""

    def __init__(self, path):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path), check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(SCHEMA)
        self.lock = threading.Lock()

    def upsert_user(self, user_id, race_date=None, goal_minutes=None):
        with self.lock, self.connection:
            self.connection.execute(
                "INSERT INTO users (user_id, race_date, goal_minutes, created_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET race_date = COALESCE(excluded.race_date, race_date), "
                "goal_minutes = COALESCE(excluded.goal_minutes, goal_minutes)",
                (user_id, race_date, goal_minutes, time.time()))
        return self.user(user_id)

    def user(self, user_id):
        row = self.connection.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        return dict(row) if row else None

    def add_sessions(self, user_id, sessions, source):
        with self.lock, self.connection:
            self.connection.executemany(
                "INSERT INTO sessions (user_id, date, km, minutes, source) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(user_id, date) DO UPDATE SET km = excluded.km, minutes = excluded.minutes, "
                "source = excluded.source",
                [(user_id, str(s["date"]), float(s["km"]), float(s["minutes"]), source) for s in sessions])

    def sessions(self, user_id, before=None):
        query = "SELECT date, km, minutes, source FROM sessions WHERE user_id = ?"
        parameters = [user_id]
        if before is not None:
            query += " AND date < ?"
            parameters.append(str(before))
        return [dict(row) for row in self.connection.execute(query + " ORDER BY date", parameters)]

    def save_recommendation(self, user_id, target_date, model, items):
        rec_id = uuid.uuid4().hex
        with self.lock, self.connection:
            self.connection.execute(
                "INSERT INTO recommendations (rec_id, user_id, target_date, model, items, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (rec_id, user_id, str(target_date), model, json.dumps(items), time.time()))
        return rec_id

    def log_interaction(self, user_id, type, date, rec_id=None, item=None, km=None, minutes=None, effort=None):
        if type not in INTERACTION_TYPES:
            raise InteractionError(f"type must be one of {INTERACTION_TYPES}")
        if effort is not None and (effort not in EFFORTS or type != "completed"):
            raise InteractionError("effort is only accepted for a completed session")
        if type == "completed" and (km is None or minutes is None):
            raise InteractionError("a completed session needs km and minutes")
        if self.user(user_id) is None:
            raise InteractionError("unknown user")
        if rec_id is not None:
            row = self.connection.execute("SELECT user_id, items FROM recommendations WHERE rec_id = ?", (rec_id,)).fetchone()
            if row is None or row["user_id"] != user_id:
                raise InteractionError("unknown rec_id for this user")
            if item is not None and item not in json.loads(row["items"]):
                raise InteractionError("item was not in that recommendation")
        with self.lock, self.connection:
            cursor = self.connection.execute(
                "INSERT INTO interactions (user_id, type, date, rec_id, item, km, minutes, effort, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, type, str(date), rec_id, item, km, minutes, effort, time.time()))
            if type == "completed":
                self.connection.execute(
                    "INSERT INTO sessions (user_id, date, km, minutes, source) VALUES (?, ?, ?, ?, 'logged') "
                    "ON CONFLICT(user_id, date) DO UPDATE SET km = excluded.km, minutes = excluded.minutes, source = 'logged'",
                    (user_id, str(date), float(km), float(minutes)))
        return cursor.lastrowid

    def interactions(self, user_id):
        return [dict(row) for row in self.connection.execute(
            "SELECT * FROM interactions WHERE user_id = ? ORDER BY date, interaction_id", (user_id,))]
