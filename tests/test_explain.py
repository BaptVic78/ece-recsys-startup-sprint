import json
import os
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

os.environ["MARATHON_SKIP_APP"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

from marathon.api import create_app  # noqa: E402
from marathon.data import NUM_ITEMS  # noqa: E402
from marathon.model import ContextCrossAttentionRecommender  # noqa: E402
from marathon.sequences import RACE_BUCKETS, RECOVERY_BUCKETS, SESSION_FEATURES, WEEKDAYS, item_attributes  # noqa: E402
from marathon.explain import REMOVED, effect, report, run, sample_targets, top_slots  # noqa: E402
from marathon.sequences import MIN_HISTORY, SessionStore  # noqa: E402
from marathon.serving import EXPLAIN_TOP, Recommender, because_of  # noqa: E402

HISTORY = 32


def small_model(dropout=0.1):
    torch.manual_seed(0)
    return ContextCrossAttentionRecommender(NUM_ITEMS, item_attributes(), HISTORY, dim=16, heads=2, layers=1,
                                            dropout=dropout).eval()


def padded_batch(lengths, seed=0):
    """Left-padded histories of the given lengths, like SessionStore.batch builds them."""
    rng = np.random.default_rng(seed)
    size = len(lengths)
    mask = np.zeros((size, HISTORY), dtype=bool)
    for row, length in enumerate(lengths):
        mask[row, HISTORY - length:] = True
    batch = {
        "items": np.where(mask, rng.integers(1, NUM_ITEMS + 1, (size, HISTORY)), 0),
        "features": np.where(mask[..., None], rng.normal(size=(size, HISTORY, SESSION_FEATURES)), 0).astype(np.float32),
        "race": np.where(mask, rng.integers(0, RACE_BUCKETS, (size, HISTORY)), 0),
        "recovery": np.where(mask, rng.integers(0, RECOVERY_BUCKETS, (size, HISTORY)), 0),
        "weekday": np.where(mask, rng.integers(0, WEEKDAYS, (size, HISTORY)), 0),
        "mask": mask,
        "target_weekday": rng.integers(0, WEEKDAYS, size),
        "target_gap": rng.random(size).astype(np.float32),
        "target_race": rng.integers(0, RACE_BUCKETS, size),
        "target_recovery": rng.integers(0, RECOVERY_BUCKETS, size),
    }
    return {key: torch.as_tensor(value) for key, value in batch.items()}


def save_small_model(directory):
    model = small_model(dropout=0.0)
    torch.save(model.state_dict(), directory / "full.pt")
    config = {"model": "full", "use_race_context": True, "max_history": HISTORY, "num_items": NUM_ITEMS,
              "dim": 16, "heads": 2, "layers": 1, "dropout": 0.0, "best_epoch": 1}
    (directory / "full.json").write_text(json.dumps(config), encoding="utf-8")
    counts = np.ones((RACE_BUCKETS, NUM_ITEMS), dtype=int)
    (directory / "phase_popularity.json").write_text(json.dumps({"counts": counts.tolist()}), encoding="utf-8")


def history(days=30, start=date(2019, 8, 1)):
    """15 sessions, fewer than the 32-slot window, so the model input is padded."""
    return [{"date": start + timedelta(days=offset), "km": 8.0 + offset % 5, "minutes": (8.0 + offset % 5) * 5.5}
            for offset in range(0, days, 2)]


class AttentionWeightsTest(unittest.TestCase):
    def setUp(self):
        self.model = small_model()
        self.batch = padded_batch([5, 12, 32, 1])

    def test_weights_sum_to_one_over_valid_history_and_are_zero_on_padding(self):
        with torch.no_grad():
            _, weights = self.model(self.batch, return_attention=True)
        self.assertEqual(tuple(weights.shape), (4, NUM_ITEMS, HISTORY))
        mask = self.batch["mask"][:, None, :].expand_as(weights)
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(4, NUM_ITEMS))
        self.assertTrue((weights[~mask] == 0).all())
        self.assertTrue((weights >= 0).all())

    def test_scores_are_the_same_with_and_without_weights(self):
        with torch.no_grad():
            default = self.model(self.batch)
            scores, _ = self.model(self.batch, return_attention=True)
        self.assertIsInstance(default, torch.Tensor)  # the default call still returns scores only
        # The two calls can take different attention kernels, so allow float rounding only.
        torch.testing.assert_close(scores, default, rtol=0, atol=1e-5)
        self.assertTrue(torch.equal(default.argsort(dim=1, descending=True, stable=True),
                                    scores.argsort(dim=1, descending=True, stable=True)))


class BecauseOfTest(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({"date": pd.to_datetime(["2019-08-01", "2019-08-03", "2019-08-05", "2019-08-07"]),
                             "km": [10.0, 28.0, 8.0, 12.0], "minutes": [55.0, 160.0, 44.0, 66.0],
                             "pace": [5.5, 160 / 28, 5.5, 5.5]})

    def test_padding_is_never_returned_even_with_the_highest_weight(self):
        weights = np.array([0.9, 0.0, 0.05, 0.03, 0.02, 0.0])  # slot 0 is padding with a (forced) high weight
        valid = np.array([False, False, True, True, True, True])
        positions = np.array([-2, -1, 0, 1, 2, 3])
        result = because_of(weights, self.frame(), positions, valid)
        self.assertEqual([run["date"] for run in result], ["2019-08-01", "2019-08-03", "2019-08-05"])
        self.assertEqual([run["weight"] for run in result], [0.05, 0.03, 0.02])
        self.assertEqual(result[1], {"date": "2019-08-03", "km": 28.0, "minutes": 160.0, "pace": "5:43/km",
                                     "weight": 0.03})

    def test_short_history_returns_fewer_sessions(self):
        weights = np.array([0.0, 0.0, 0.4, 0.6])
        valid = np.array([False, False, True, True])
        result = because_of(weights, self.frame(), np.array([-2, -1, 0, 1]), valid)
        self.assertEqual([run["date"] for run in result], ["2019-08-03", "2019-08-01"])


class ServedExplanationTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        save_small_model(Path(self.directory.name))
        self.recommender = Recommender(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_every_recommendation_names_real_earlier_sessions(self):
        sessions = history()
        target = date(2019, 9, 1)
        day = self.recommender.recommend(sessions, target, date(2019, 9, 29), top_k=5, strategy="sample", seed=3)
        known = {str(s["date"]): s for s in sessions}
        for item in day["recommendations"] + [day["planned"]]:
            runs = item["because_of"]
            self.assertEqual(len(runs), EXPLAIN_TOP)
            self.assertEqual(len({run["date"] for run in runs}), EXPLAIN_TOP)
            for run in runs:
                self.assertIn(run["date"], known)  # never padding, never an invented session
                self.assertLess(run["date"], str(target))
                self.assertEqual(run["km"], known[run["date"]]["km"])
            weights = [run["weight"] for run in runs]
            self.assertEqual(weights, sorted(weights, reverse=True))
            self.assertLessEqual(sum(weights), 1.0 + 0.01)  # weights are rounded to 3 decimals

    def test_scores_do_not_change_when_explanations_are_added(self):
        scores, _ = self.recommender.rank(history(), date(2019, 9, 1), date(2019, 9, 29))
        day = self.recommender.recommend(history(), date(2019, 9, 1), date(2019, 9, 29), top_k=NUM_ITEMS)
        served = {item["item"]: item["score"] for item in day["recommendations"]}
        self.assertEqual([served[i + 1] for i in range(NUM_ITEMS)], [float(s) for s in scores])

    def test_cold_start_has_no_explanation(self):
        day = self.recommender.recommend(history(days=6), date(2019, 9, 1), strategy="sample")
        self.assertEqual(day["strategy"], "phase_popularity")
        self.assertTrue(all(item["because_of"] == [] for item in day["recommendations"] + [day["planned"]]))

    def test_recommend_endpoint_returns_because_of_for_each_item(self):
        client = TestClient(create_app(self.directory.name, ":memory:", Path(self.directory.name) / "no-demo"))
        sessions = [{**s, "date": str(s["date"])} for s in history()]
        client.post("/users", json={"user_id": "lea", "race_date": "2019-09-29", "sessions": sessions})
        body = client.post("/recommend", json={"user_id": "lea", "date": "2019-09-01", "top_k": 3}).json()
        self.assertEqual(len(body["recommendations"]), 3)
        for item in body["recommendations"] + [body["planned"]]:
            self.assertEqual(len(item["because_of"]), EXPLAIN_TOP)
            self.assertEqual(set(item["because_of"][0]), {"date", "km", "minutes", "pace", "weight"})


def synthetic_store():
    """Three runners with 20 sessions each; the last 6 of every runner are in the test split."""
    rows = []
    for athlete in range(3):
        for offset in range(20):
            day = pd.Timestamp("2019-08-01") + pd.Timedelta(days=2 * offset + athlete)
            km = 6.0 + (offset * (athlete + 2)) % 15
            rows.append(dict(athlete=athlete, date=day, km=km, minutes=km * 5.5, intensity=1.0,
                             item=1 + (offset * 7 + athlete) % NUM_ITEMS, is_race=False, days_to_race=np.nan,
                             split="test" if offset >= 14 else "train"))
    return SessionStore(pd.DataFrame(rows))


class DeletionTestTest(unittest.TestCase):
    def test_selected_slots_are_always_real_sessions(self):
        mask = np.array([[False, False, True, True, True, True], [True] * 6])
        keys = np.array([[9.0, 8.0, 0.1, 0.4, 0.3, 0.2], [0.0, 0.5, 0.1, 0.9, 0.2, 0.3]])
        chosen = top_slots(keys, mask)
        self.assertEqual(chosen.sum(axis=1).tolist(), [REMOVED, REMOVED])
        self.assertFalse((chosen & ~mask).any())
        self.assertEqual(np.flatnonzero(chosen[0]).tolist(), [3, 4, 5])
        self.assertEqual(np.flatnonzero(chosen[1]).tolist(), [1, 3, 5])

    def test_removing_nothing_changes_nothing(self):
        model, batch = small_model(), padded_batch([10, 20, 32])
        with torch.no_grad():
            logits = model(batch).numpy()
        items = logits.argmax(axis=1)
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        rows = np.arange(3)
        result = effect(model, batch, np.zeros((3, HISTORY), dtype=bool), items, logits[rows, items],
                        probabilities[rows, items])
        np.testing.assert_allclose(result["logit_drop"], 0, atol=1e-5)
        self.assertEqual(result["top1_kept"].tolist(), [1.0, 1.0, 1.0])

    def test_deletion_run_samples_one_target_per_runner_and_reports_every_method(self):
        store = synthetic_store()
        rng = np.random.default_rng(0)
        targets = sample_targets(store, HISTORY, 10, rng)
        athletes = store.sessions["athlete"].to_numpy()[targets]
        self.assertEqual(len(targets), 3)
        self.assertEqual(len(set(athletes)), 3)
        self.assertTrue((store.position[targets] >= MIN_HISTORY + REMOVED).all())
        model = small_model()
        per_target, share, overlap = run(store, model, HISTORY, targets, rng, batch_size=2)
        self.assertEqual(set(per_target), {"attention", "random", "recent"})
        self.assertTrue(all(len(values) == 3 for metrics in per_target.values() for values in metrics.values()))
        self.assertTrue(((share > 0) & (share <= 1 + 1e-6)).all())
        config = {"model": "full", "best_epoch": 1}
        result, text = report(per_target, share, overlap, targets, rng, config)
        self.assertEqual(result["targets"], 3)
        self.assertIn("logit_drop: attention - random", result["comparisons"])
        self.assertIn("| 3 most recent sessions |", text)


if __name__ == "__main__":
    unittest.main()
