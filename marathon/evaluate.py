"""Full-catalogue ranking metrics, sanity-floor baselines and the innovation metrics.

Every target has one relevant session type: the one the runner actually did. Its rank is the
number of types scored strictly higher, plus the tied types with a smaller ID, so ties never
favour a method. Metrics are averaged over each runner's targets, then over runners.
"""

import numpy as np
import pandas as pd

from marathon.sequences import RACE_BUCKETS

CUTOFFS = (10, 20)
PHASES = {
    "taper (race in 0-3 weeks)": lambda race, recovery: (race >= 1) & (race <= 3),
    "peak (4-10 weeks out)": lambda race, recovery: (race >= 4) & (race <= 10),
    "build-up (11-20 weeks out)": lambda race, recovery: race >= 11,
    "post-race recovery (0-8 weeks after)": lambda race, recovery: (race == 0) & (recovery > 0),
    "no race context": lambda race, recovery: (race == 0) & (recovery == 0),
}


def target_ranks(scores, labels):
    """0-based rank of each label under deterministic tie-breaking by item index."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite")
    target = scores[np.arange(len(labels)), labels][:, None]
    columns = np.arange(scores.shape[1])[None, :]
    return ((scores > target) | ((scores == target) & (columns < labels[:, None]))).sum(axis=1)


def per_target_metrics(ranks):
    ranks = np.asarray(ranks)
    result = {}
    for k in CUTOFFS:
        hit = ranks < k
        result[f"ndcg@{k}"] = np.where(hit, 1 / np.log2(ranks + 2), 0.0)
        result[f"recall@{k}"] = hit.astype(float)
    result["mrr"] = 1 / (ranks + 1)
    return pd.DataFrame(result)


def macro_average(ranks, users):
    """Average per runner first, then across runners, so heavy loggers do not dominate."""
    frame = per_target_metrics(ranks)
    frame["user"] = np.asarray(users)
    per_user = frame.groupby("user").mean()
    summary = per_user.mean().to_dict()
    summary.update(users=int(len(per_user)), targets=int(len(frame)))
    return summary


def phase_metrics(ranks, users, race, recovery):
    result = {}
    for name, rule in PHASES.items():
        mask = rule(np.asarray(race), np.asarray(recovery))
        if mask.any():
            result[name] = macro_average(np.asarray(ranks)[mask], np.asarray(users)[mask])
    return result


def exposure_metrics(scores, popularity, k=10):
    """Catalogue coverage of top-k lists and how popular the recommended types are on average."""
    top = np.argsort(-np.asarray(scores), axis=1, kind="stable")[:, :k]
    popularity_rank = np.argsort(np.argsort(-popularity, kind="stable"), kind="stable")
    return {f"coverage@{k}": float(len(np.unique(top)) / scores.shape[1]),
            f"mean_popularity_rank@{k}": float(popularity_rank[top].mean())}


class Baselines:
    """Rankings that need no learning. Fitted on training targets only."""

    def __init__(self, train_labels, train_race, num_items, seed=42):
        self.popularity = np.bincount(train_labels, minlength=num_items).astype(float)
        self.phase = np.zeros((RACE_BUCKETS, num_items))
        np.add.at(self.phase, (train_race, train_labels), 1)
        self.num_items = num_items
        self.seed = seed

    def scores(self, name, batch, example_ids):
        n = len(batch["labels"])
        tie = self.popularity / (self.popularity.max() + 1)
        if name == "random":
            # Seeded by example ID, so the ranking of a target does not depend on chunking.
            return np.stack([np.random.default_rng([self.seed, int(i)]).random(self.num_items) for i in example_ids])
        if name == "most_popular":
            return np.tile(self.popularity, (n, 1))
        if name == "phase_popular":
            return self.phase[batch["target_race"]] + tie[None, :]
        history = np.where(batch["mask"], batch["items"] - 1, -1)
        if name == "personal_popular":
            counts = np.zeros((n, self.num_items))
            rows, cols = np.nonzero(history >= 0)
            np.add.at(counts, (rows, history[rows, cols]), 1)
            return counts + tie[None, :]
        if name == "repeat_last":
            last = history[:, -1]
            result = np.tile(tie, (n, 1))
            result[np.arange(n), last] += 1
            return result
        raise ValueError(f"Unknown baseline {name}")


BASELINES = ("random", "most_popular", "phase_popular", "personal_popular", "repeat_last")
