"""Predict functions called by the API: rank session types for a day and roll out a weekly plan.

Served histories go through `annotate_sessions` and `SessionStore`, the same code that built the
training data, so a logged session is encoded exactly as in training. Runners with fewer than
MIN_HISTORY usable sessions get the training popularity of their preparation phase instead,
because the model was only trained with at least that much history.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from marathon.data import (DISTANCE_POINTS, INTENSITY_EDGES, INTENSITY_POINTS, MIN_SESSION_KM, NUM_ITEMS, PACE_RANGE,
                           annotate_sessions, describe_item, plausible_pace)
from marathon.model import ContextCrossAttentionRecommender
from marathon.sequences import MIN_HISTORY, RACE_BUCKETS, SessionStore, item_attributes, race_bucket

DEFAULT_PACE = 6.0


def phase_name(days_to_race):
    if days_to_race is None or days_to_race < 0:
        return "no race planned"
    if days_to_race == 0:
        return "race day"
    weeks = days_to_race // 7 + 1
    if weeks <= 3:
        return f"taper, race in {days_to_race} day{'s' if days_to_race > 1 else ''}"
    if weeks <= 10:
        return f"peak training, race in {weeks} weeks"
    if weeks <= 21:
        return f"build-up, race in {weeks} weeks"
    return f"base training, race in {weeks} weeks"


def prescription(item, reference_pace):
    distance_band, intensity_band = divmod(int(item) - 1, len(INTENSITY_EDGES) + 1)
    km = DISTANCE_POINTS[distance_band]
    # Keep every prescription inside the quality rule the API applies when it is logged back:
    # clip the pace, then round the duration to whole minutes that still satisfy the rule.
    pace = min(max(reference_pace * INTENSITY_POINTS[intensity_band], PACE_RANGE[0]), PACE_RANGE[1])
    minutes = min(max(round(km * pace), math.ceil(km * PACE_RANGE[0])), math.floor(km * PACE_RANGE[1]))
    info = describe_item(item)
    range_text = f"{info['km_min']:g}+ km" if info["km_max"] is None else f"{info['km_min']:g}-{info['km_max']:g} km"
    seconds = round(pace * 60)
    return {"item": int(item), "title": f"{km:g} km {info['intensity']} run", "distance_range": range_text,
            "intensity": info["intensity"], "km": km, "pace_min_per_km": round(pace, 2),
            "pace": f"{seconds // 60}:{seconds % 60:02d}/km", "minutes": minutes}


EXPLAIN_TOP = 3


def because_of(weights, frame, positions, valid, top=EXPLAIN_TOP):
    """The `top` history sessions a candidate attended to most, highest weight first.

    `weights` are one candidate's cross-attention weights over the model's history window,
    `positions` the row of each window slot in `frame` (the SessionStore's sorted sessions)
    and `valid` which slots hold a real session.
    Padding slots are never returned, even if their weight ties with a real session.
    """
    slots = [slot for slot in np.argsort(-weights, kind="stable") if valid[slot]][:top]
    result = []
    for slot in slots:
        row = frame.iloc[int(positions[slot])]
        seconds = round(float(row["pace"]) * 60)
        result.append({"date": str(row["date"].date()), "km": round(float(row["km"]), 1),
                       "minutes": round(float(row["minutes"]), 1), "pace": f"{seconds // 60}:{seconds % 60:02d}/km",
                       "weight": round(float(weights[slot]), 3)})
    return result


PLAN_STRATEGIES = ("sample", "greedy")
PLAN_TOP_P = 0.9


def session_probabilities(scores, strategy):
    """Model logits become a softmax; the cold-start fallback's counts become frequencies."""
    scores = np.asarray(scores, dtype=float)
    if strategy == "model":
        weights = np.exp(scores - scores.max())
    else:
        weights = np.clip(scores, 0, None)
    return weights / weights.sum()


def day_rng(seed, date):
    """Generator for one planned day, from the seed and that day's date only.

    A day's draw then changes only when the history before it changes: following the plan
    and asking again from the next day returns the same remaining sessions.
    """
    return np.random.default_rng([int(seed), pd.Timestamp(date).toordinal()])


def nucleus_sample(probabilities, top_p, rng):
    """Draw an index from the smallest highest-probability set whose mass reaches top_p."""
    order = np.argsort(-probabilities, kind="stable")
    cumulative = np.cumsum(probabilities[order])
    keep = order[:int(np.searchsorted(cumulative, top_p) + 1)]
    weights = probabilities[keep] / probabilities[keep].sum()
    return int(rng.choice(keep, p=weights))


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
        # Same quality rule as training: implausible paces never reach the pace reference or the model.
        rows = [row for row in rows if row["date"] < target_date and row["km"] >= MIN_SESSION_KM
                and plausible_pace(row["km"], row["minutes"])]
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
        scores, info, _ = self._rank(sessions, target_date, race_date)
        return scores, info

    def _rank(self, sessions, target_date, race_date=None):
        """`rank`, plus a function giving the `because_of` sessions of a 0-based item index.

        Scores come from the model's default call, so they are exactly those of `rank` and of
        offline evaluation; the attention weights come from a second call used only to explain.
        """
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
            explain = lambda index: []  # noqa: E731 - no model, so no attention to explain with
        else:
            store = SessionStore(frame)
            max_history = self.config["max_history"]
            batch = store.batch(np.array([target]), max_history=max_history)
            inputs = {key: torch.as_tensor(value) for key, value in batch.items()}
            with torch.no_grad():
                scores = self.model(inputs)[0].numpy()
                weights = self.model(inputs, return_attention=True)[1][0].numpy()
            positions = target + np.arange(-max_history, 0)
            valid = batch["mask"][0]
            explain = lambda index: because_of(weights[index], store.sessions, positions, valid)  # noqa: E731
            strategy = "model"
        info = {"strategy": strategy, "history_sessions": int(target), "reference_pace": float(reference),
                "days_to_race": None if not np.isfinite(days_to_race) else int(days_to_race)}
        return scores, info, explain

    def _day(self, scores, info, target_date, top_k, explain=lambda index: []):
        order = np.argsort(-scores, kind="stable")
        probabilities = session_probabilities(scores, info["strategy"])
        items = [{"rank": rank + 1, "score": float(scores[index]), "probability": float(probabilities[index]),
                  **prescription(index + 1, info["reference_pace"]), "because_of": explain(index)}
                 for rank, index in enumerate(order[:top_k])]
        return {"date": str(pd.Timestamp(target_date).date()), "phase": phase_name(info["days_to_race"]),
                **info, "model": self.name, "recommendations": items}, order

    def _planned(self, scores, info, order, target_date, strategy, seed, top_p, explain=lambda index: []):
        """The session planned for one day: the top-ranked type, or a draw from the model's distribution."""
        if strategy not in PLAN_STRATEGIES:
            raise ValueError(f"strategy must be one of {PLAN_STRATEGIES}")
        probabilities = session_probabilities(scores, info["strategy"])
        if strategy == "greedy":
            chosen = int(order[0])
        else:
            chosen = nucleus_sample(probabilities, top_p, day_rng(seed, target_date))
        return {"rank": int(np.flatnonzero(order == chosen)[0]) + 1, "score": float(scores[chosen]),
                "probability": float(probabilities[chosen]), **prescription(chosen + 1, info["reference_pace"]),
                "because_of": explain(chosen)}

    def recommend(self, sessions, target_date, race_date=None, top_k=10, strategy=None, seed=0, top_p=PLAN_TOP_P):
        """Ranked session types for one day. With a `strategy`, also the session `planned` for that day,
        drawn exactly as `plan` draws it, so a day shows the same session in both views."""
        scores, info, explain = self._rank(sessions, target_date, race_date)
        day, order = self._day(scores, info, target_date, top_k, explain)
        if strategy is not None:
            day["planned"] = self._planned(scores, info, order, target_date, strategy, seed, top_p, explain)
            day["plan_strategy"] = strategy
        return day

    def plan(self, sessions, run_dates, race_date=None, top_k=3, strategy="sample", seed=0, top_p=PLAN_TOP_P):
        """Roll the model forward over the run dates; each planned session is assumed done before the next.

        `strategy="sample"` (default) draws each day's session from the model's own distribution,
        restricted to the smallest set of types covering `top_p` of the probability (nucleus
        sampling). Each day has its own generator, seeded by `seed` and the date, so a plan is
        reproducible and stays the same when the runner follows it. `strategy="greedy"` always
        takes the top-ranked type: the most likely session each day, but the most likely session
        rarely changes within a week, so greedy weeks repeat one session.
        """
        if strategy not in PLAN_STRATEGIES:
            raise ValueError(f"strategy must be one of {PLAN_STRATEGIES}")
        history = list(sessions)
        days = []
        for date in sorted(pd.Timestamp(d).normalize() for d in run_dates):
            day = self.recommend(history, date, race_date, top_k, strategy=strategy, seed=seed, top_p=top_p)
            days.append(day)
            history.append({"date": date, "km": day["planned"]["km"], "minutes": day["planned"]["minutes"]})
        return days
