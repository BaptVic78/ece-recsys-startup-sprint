"""Turn the session table into model inputs without materialising every padded history.

All sessions live in flat arrays sorted by runner and date. An example is the index of a target
session. Its history is the up-to-`max_history` sessions just before it for the same runner.
Every history session is strictly earlier than its target, because a runner has at most one
session per day.
"""

import numpy as np
import pandas as pd

from marathon.data import INTENSITY_EDGES, DISTANCE_EDGES, MIN_SESSION_KM, NUM_ITEMS

MAX_HISTORY = 32
MIN_HISTORY = 5
RACE_WEEKS = 20
RECOVERY_WEEKS = 8
# Numeric attributes of a session: log distance, clipped intensity, log gap since the previous run.
SESSION_FEATURES = 3
# Bucket sizes for the context embeddings.
RACE_BUCKETS = RACE_WEEKS + 2      # 0 = no race within 20 weeks, 1..21 = weeks to race (1 = race week)
RECOVERY_BUCKETS = RECOVERY_WEEKS + 2  # 0 = no race in the last 8 weeks, 1..9 = weeks since the race
WEEKDAYS = 7


def race_bucket(days_to_race):
    days = np.asarray(days_to_race, dtype=float)
    bucket = np.floor(np.nan_to_num(days, nan=-1) / 7).astype(int) + 1
    return np.where(np.isfinite(days) & (days >= 0) & (days <= 7 * RACE_WEEKS + 6), bucket, 0)


def recovery_bucket(days_since_race):
    days = np.asarray(days_since_race, dtype=float)
    bucket = np.floor(np.nan_to_num(days, nan=-1) / 7).astype(int) + 1
    return np.where(np.isfinite(days) & (days >= 0) & (days <= 7 * RECOVERY_WEEKS + 6), bucket, 0)


def item_attributes():
    """Static attributes of each session type: log of the band's middle distance and its middle intensity."""
    distance_edges = [MIN_SESSION_KM, *DISTANCE_EDGES, 42.0]
    intensity_edges = [0.85, *INTENSITY_EDGES, 1.15]
    rows = []
    for distance_band in range(len(DISTANCE_EDGES) + 1):
        for intensity_band in range(len(INTENSITY_EDGES) + 1):
            km = (distance_edges[distance_band] + distance_edges[distance_band + 1]) / 2
            intensity = (intensity_edges[intensity_band] + intensity_edges[intensity_band + 1]) / 2
            rows.append([np.log1p(km), (intensity - 1) * 10])
    attributes = np.array(rows, dtype=np.float32)
    assert len(attributes) == NUM_ITEMS
    return attributes


class SessionStore:
    """Flat, chronologically sorted session arrays with per-session features and context."""

    def __init__(self, sessions):
        sessions = sessions.sort_values(["athlete", "date"], kind="stable").reset_index(drop=True)
        self.sessions = sessions
        athlete = sessions["athlete"].to_numpy()
        dates = sessions["date"].to_numpy("datetime64[D]")
        new_runner = np.r_[True, athlete[1:] != athlete[:-1]]
        self.runner_start = np.maximum.accumulate(np.where(new_runner, np.arange(len(athlete)), 0))
        self.position = np.arange(len(athlete)) - self.runner_start
        gap = np.r_[0, (dates[1:] - dates[:-1]).astype(int)].astype(float)
        gap[new_runner] = 0
        self.gap_days = gap
        self.item = sessions["item"].to_numpy().astype(np.int64)
        self.features = np.column_stack([
            np.log1p(sessions["km"].to_numpy()),
            np.clip(sessions["intensity"].to_numpy(), 0.5, 1.5) * 10 - 10,
            np.log1p(gap)]).astype(np.float32)
        self.weekday = sessions["date"].dt.day_of_week.to_numpy().astype(np.int64)
        self.race_bucket = race_bucket(sessions["days_to_race"].to_numpy())
        # Days since the most recent earlier-or-same-day race of this runner, computed causally.
        race_index = np.where(sessions["is_race"].to_numpy(), np.arange(len(athlete)), -1)
        last_race = np.maximum.accumulate(race_index)
        valid = (last_race >= 0) & (last_race >= self.runner_start)
        since = np.where(valid, (dates - dates[np.maximum(last_race, 0)]).astype(int), np.nan)
        self.days_since_race = since
        self.recovery_bucket = recovery_bucket(since)
        self.is_race = sessions["is_race"].to_numpy()
        self.split = sessions["split"].to_numpy()

    def targets(self, split, min_history=MIN_HISTORY):
        """Indices of recommendation targets: non-race sessions of a split with enough history."""
        mask = (self.split == split) & ~self.is_race & (self.position >= min_history)
        return np.flatnonzero(mask)

    def target_context(self, targets):
        """Label, race-calendar and recovery buckets of target sessions, without their histories."""
        targets = np.asarray(targets)
        # The race used for recovery must be strictly before the target day, so start from the
        # previous session (targets always have history, so targets - 1 is the same runner).
        since = self.days_since_race[targets - 1] + self.gap_days[targets]
        return {"labels": self.item[targets] - 1, "target_race": self.race_bucket[targets],
                "target_recovery": recovery_bucket(since)}

    def batch(self, targets, max_history=MAX_HISTORY):
        """Gather left-padded histories and the target-day context for a batch of target indices."""
        targets = np.asarray(targets)
        offsets = np.arange(-max_history, 0)
        index = targets[:, None] + offsets[None, :]
        valid = index >= self.runner_start[targets][:, None]
        index = np.where(valid, index, 0)
        context = self.target_context(targets)
        return {
            "items": np.where(valid, self.item[index], 0),
            "features": np.where(valid[..., None], self.features[index], 0).astype(np.float32),
            "race": np.where(valid, self.race_bucket[index], 0),
            "recovery": np.where(valid, self.recovery_bucket[index], 0),
            "weekday": np.where(valid, self.weekday[index], 0),
            "mask": valid,
            "target_weekday": self.weekday[targets],
            "target_gap": np.log1p(self.gap_days[targets]).astype(np.float32),
            **context,
        }


def load_store(path):
    return SessionStore(pd.read_parquet(path))
