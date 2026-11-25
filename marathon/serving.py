"""Predict functions called by the API: rank session types for a day and roll out a weekly plan.

Served histories go through `annotate_sessions` and `SessionStore`, the same code that built the
training data, so a logged session is encoded exactly as in training. Runners with fewer than
MIN_HISTORY usable sessions get the training popularity of their preparation phase instead,
because the model was only trained with at least that much history.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from marathon.data import DISTANCE_EDGES, INTENSITY_EDGES, MIN_SESSION_KM, NUM_ITEMS, annotate_sessions, describe_item
from marathon.model import ContextCrossAttentionRecommender
from marathon.sequences import MIN_HISTORY, RACE_BUCKETS, SessionStore, item_attributes, race_bucket

# Representative values used to turn a session type into a concrete prescription.
DISTANCE_POINTS = [3.0, 6.5, 9.0, 11.0, 13.5, 16.5, 19.5, 23.0, 27.5, 32.0, 37.0]
INTENSITY_POINTS = [0.87, 0.925, 0.97, 1.01, 1.055, 1.11]
DEFAULT_PACE = 6.0
assert len(DISTANCE_POINTS) == len(DISTANCE_EDGES) + 1 and len(INTENSITY_POINTS) == len(INTENSITY_EDGES) + 1


def phase_name(days_to_race):
    if days_to_race is None or days_to_race < 0:
        return "no race planned"
    if days_to_race == 0:
        return "race day"
    weeks = days_to_race // 7 + 1
    if weeks <= 3:
        return f"taper, race in {days_to_race} days"
    if weeks <= 10:
        return f"peak training, race in {weeks} weeks"
    if weeks <= 21:
        return f"build-up, race in {weeks} weeks"
    return f"base training, race in {weeks} weeks"


def prescription(item, reference_pace):
    distance_band, intensity_band = divmod(int(item) - 1, len(INTENSITY_EDGES) + 1)
    km = DISTANCE_POINTS[distance_band]
    pace = reference_pace * INTENSITY_POINTS[intensity_band]
    info = describe_item(item)
    range_text = f"{info['km_min']:g}+ km" if info["km_max"] is None else f"{info['km_min']:g}-{info['km_max']:g} km"
    seconds = round(pace * 60)
    return {"item": int(item), "title": f"{km:g} km {info['intensity']} run", "distance_range": range_text,
            "intensity": info["intensity"], "km": km, "pace_min_per_km": round(pace, 2),
            "pace": f"{seconds // 60}:{seconds % 60:02d}/km", "minutes": round(km * pace)}


def export_phase_popularity(sessions_path, model_dir):
    """Cold-start fallback: session-type counts per race-calendar bucket, from training sessions only."""
    sessions = pd.read_parquet(sessions_path, columns=["item", "days_to_race", "is_race", "split"])
    train = sessions[(sessions["split"] == "train") & ~sessions["is_race"]]
    buckets = race_bucket(train["days_to_race"].to_numpy())
    counts = np.zeros((RACE_BUCKETS, NUM_ITEMS), dtype=np.int64)
    np.add.at(counts, (buckets, train["item"].to_numpy() - 1), 1)
    path = Path(model_dir) / "phase_popularity.json"
    path.write_text(json.dumps({"source": "training split only", "counts": counts.tolist()}) + "\n", encoding="utf-8")
    return path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export the cold-start fallback next to the model.")
    parser.add_argument("--data", type=Path, default=Path("data/processed/marathon/sessions.parquet"))
    parser.add_argument("--models", type=Path, default=Path("models/marathon"))
    args = parser.parse_args()
    print(export_phase_popularity(args.data, args.models))


if __name__ == "__main__":
    main()


class Recommender:
    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        self.config = json.loads((model_dir / "full.json").read_text(encoding="utf-8"))
        if self.config["num_items"] != NUM_ITEMS:
            raise ValueError("The saved model does not match the session catalogue")
        self.model = ContextCrossAttentionRecommender(
            NUM_ITEMS, item_attributes(), self.config["max_history"], dim=self.config["dim"],
            heads=self.config["heads"], layers=self.config["layers"], dropout=self.config["dropout"],
            use_race_context=self.config["use_race_context"])
        self.model.load_state_dict(torch.load(model_dir / "full.pt", weights_only=True))
        self.model.eval()
        fallback = json.loads((model_dir / "phase_popularity.json").read_text(encoding="utf-8"))
        self.phase_popularity = np.array(fallback["counts"], dtype=float)
        self.name = f"{self.config['model']}-epoch{self.config['best_epoch']}"
        # One request scores one runner: a single thread is fastest and keeps the API responsive.
        torch.set_num_threads(1)
        self.recommend([{"date": f"2019-01-{day:02d}", "km": 10.0, "minutes": 55.0} for day in range(1, 12)],
                       "2019-01-12")  # warm-up so the first real request is not slower

    def _frame(self, sessions, target_date, race_date):
        """History strictly before target_date plus a placeholder row for the target day."""
        target_date = pd.Timestamp(target_date).normalize()
        rows = [{"athlete": "runner", "date": pd.Timestamp(s["date"]).normalize(), "km": float(s["km"]),
                 "minutes": float(s["minutes"])} for s in sessions]
        rows = [row for row in rows if row["date"] < target_date and row["km"] >= MIN_SESSION_KM and row["minutes"] > 0]
        rows.append({"athlete": "runner", "date": target_date, "km": 10.0, "minutes": 60.0})
        frame = annotate_sessions(pd.DataFrame(rows))
        race = pd.Timestamp(race_date).normalize() if race_date else None
        days = (race - frame["date"]).dt.days if race is not None else pd.Series(np.nan, index=frame.index)
        frame["days_to_race"] = days.where(days >= 0)
        frame["is_race"] = (frame["days_to_race"] == 0) & (frame["km"] >= 41) & (frame["date"] < target_date)
        frame["split"] = "serve"
        return frame

    def rank(self, sessions, target_date, race_date=None):
        """Scores for all session types on target_date and the strategy that produced them."""
        frame = self._frame(sessions, target_date, race_date)
        target = len(frame) - 1
        days_to_race = frame["days_to_race"].iloc[target]
        reference = frame["reference_pace"].iloc[target]
        history = frame.iloc[:target]
        if not np.isfinite(reference):
            reference = float(history["pace"].median()) if len(history) else DEFAULT_PACE
        if target < MIN_HISTORY:
            bucket = int(race_bucket([days_to_race])[0])
            scores = self.phase_popularity[bucket] + self.phase_popularity.sum(axis=0) * 1e-6
            strategy = "phase_popularity"
        else:
            store = SessionStore(frame)
            batch = store.batch(np.array([target]), max_history=self.config["max_history"])
            with torch.no_grad():
                scores = self.model({key: torch.as_tensor(value) for key, value in batch.items()})[0].numpy()
            strategy = "model"
        return scores, {"strategy": strategy, "history_sessions": int(target), "reference_pace": float(reference),
                        "days_to_race": None if not np.isfinite(days_to_race) else int(days_to_race)}

    def recommend(self, sessions, target_date, race_date=None, top_k=10):
        scores, info = self.rank(sessions, target_date, race_date)
        order = np.argsort(-scores, kind="stable")[:top_k]
        items = [{"rank": rank + 1, "score": float(scores[index]), **prescription(index + 1, info["reference_pace"])}
                 for rank, index in enumerate(order)]
        return {"date": str(pd.Timestamp(target_date).date()), "phase": phase_name(info["days_to_race"]),
                **info, "model": self.name, "recommendations": items}

    def plan(self, sessions, run_dates, race_date=None, top_k=3):
        """Roll the model forward: each planned day's top session is assumed done before the next day."""
        history = list(sessions)
        days = []
        for date in sorted(pd.Timestamp(d).normalize() for d in run_dates):
            day = self.recommend(history, date, race_date, top_k)
            days.append(day)
            best = day["recommendations"][0]
            history.append({"date": date, "km": best["km"], "minutes": best["km"] * best["pace_min_per_km"]})
        return days
