import unittest

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from marathon.data import NUM_ITEMS
from marathon.evaluate import Baselines, macro_average, phase_metrics, target_ranks
from marathon.model import ContextCrossAttentionRecommender
from marathon.sequences import SessionStore, item_attributes, race_bucket, recovery_bucket


def synthetic_sessions():
    """Two runners. Runner 1 races on 10 January, then trains again."""
    rows = []
    for athlete, start in ((1, "2019-01-01"), (2, "2019-01-03")):
        for offset in range(0, 20, 2):
            date = pd.Timestamp(start) + pd.Timedelta(days=offset)
            race = athlete == 1 and date == pd.Timestamp("2019-01-11")
            rows.append(dict(athlete=athlete, date=date, km=42.5 if race else 10.0 + offset, minutes=200.0,
                             intensity=1.0, item=1 + offset % NUM_ITEMS, is_race=race,
                             days_to_race=(pd.Timestamp("2019-01-11") - date).days if athlete == 1 and date <= pd.Timestamp("2019-01-11") else np.nan,
                             split="train" if date < pd.Timestamp("2019-01-15") else "test"))
    return pd.DataFrame(rows)


class SequenceTest(unittest.TestCase):
    def setUp(self):
        self.store = SessionStore(synthetic_sessions())

    def test_history_is_strictly_earlier_and_from_the_same_runner(self):
        targets = self.store.targets("test", min_history=2)
        batch = self.store.batch(targets, max_history=8)
        dates = self.store.sessions["date"].to_numpy()
        athletes = self.store.sessions["athlete"].to_numpy()
        for row, target in enumerate(targets):
            valid = np.flatnonzero(batch["mask"][row])
            history = target - (8 - valid)
            self.assertTrue((dates[history] < dates[target]).all())
            self.assertTrue((athletes[history] == athletes[target]).all())
            self.assertEqual(batch["labels"][row], self.store.item[target] - 1)
        # Runner 2 starts later, so its first test target has left padding and no foreign sessions.
        self.assertFalse(batch["mask"][np.flatnonzero(athletes[targets] == 2)[0]].all())

    def test_race_day_is_never_a_target_and_recovery_starts_after_it(self):
        sessions = self.store.sessions
        race_index = int(np.flatnonzero(sessions["is_race"])[0])
        self.assertNotIn(race_index, self.store.targets("train", min_history=0))
        context = self.store.target_context(np.array([race_index + 1]))
        self.assertEqual(context["target_recovery"][0], recovery_bucket(2))

    def test_buckets(self):
        self.assertEqual(race_bucket([np.nan, 0, 6, 7, 146, 147]).tolist(), [0, 1, 1, 2, 21, 0])
        self.assertEqual(recovery_bucket([np.nan, 0, 62, 63]).tolist(), [0, 1, 9, 0])


class ModelTest(unittest.TestCase):
    def batch(self, race_value):
        store = SessionStore(synthetic_sessions())
        batch = store.batch(store.targets("test", min_history=2), max_history=8)
        batch["race"] = np.full_like(batch["race"], race_value)
        batch["target_race"] = np.full_like(batch["target_race"], race_value)
        return {key: torch.as_tensor(value) for key, value in batch.items()}

    def test_forward_backward_and_ablation(self):
        torch.manual_seed(0)
        for use_race in (True, False):
            model = ContextCrossAttentionRecommender(NUM_ITEMS, item_attributes(), 8, dim=16,
                                                     use_race_context=use_race).eval()
            near, far = model(self.batch(1)), model(self.batch(15))
            self.assertEqual(tuple(near.shape), (len(near), NUM_ITEMS))
            # The ablation must be blind to the race calendar; the full model must not be.
            self.assertEqual(torch.allclose(near, far), not use_race)
            model.train()
            batch = self.batch(3)
            loss = F.cross_entropy(model(batch), batch["labels"])
            loss.backward()
            self.assertTrue(torch.isfinite(loss))


class ReportTest(unittest.TestCase):
    def test_race_date_sweep_counts_each_runner_once_and_ablation_is_flat(self):
        from marathon.report import one_target_per_runner, race_date_sweep
        store = SessionStore(synthetic_sessions())
        targets = store.targets("test", min_history=2)
        chosen = one_target_per_runner(store, targets, np.random.default_rng(0), sample=10)
        athletes = store.sessions["athlete"].to_numpy()[chosen]
        self.assertEqual(len(chosen), 2)
        self.assertEqual(len(set(athletes)), 2)
        torch.manual_seed(0)
        models = {name: ContextCrossAttentionRecommender(NUM_ITEMS, item_attributes(), 32, dim=16,
                                                         use_race_context=use).eval()
                  for name, use in (("ours", True), ("ablation", False))}
        sweep = race_date_sweep(models, store, targets, np.random.default_rng(0), sample=10)
        self.assertEqual(sweep["runners"], 2)
        self.assertEqual(len(set(sweep["ablation"].values())), 1)
        self.assertGreater(len(set(sweep["ours"].values())), 1)


class EvaluationTest(unittest.TestCase):
    def test_ties_never_favour_the_label(self):
        scores = np.array([[1.0, 1.0, 1.0], [0.0, 2.0, 1.0]])
        self.assertEqual(target_ranks(scores, np.array([2, 2])).tolist(), [2, 1])

    def test_macro_average_weights_runners_equally(self):
        ranks = np.array([0, 0, 0, 25])
        result = macro_average(ranks, np.array(["a", "a", "a", "b"]))
        self.assertAlmostEqual(result["recall@20"], 0.5)
        self.assertAlmostEqual(result["ndcg@10"], 0.5)
        self.assertEqual(result["users"], 2)
        phases = phase_metrics(ranks, np.array(["a", "a", "a", "b"]), np.array([1, 1, 0, 0]), np.array([0, 0, 0, 2]))
        self.assertEqual(phases["taper (race in 0-3 weeks)"]["targets"], 2)

    def test_baselines(self):
        baselines = Baselines(np.array([2, 2, 1]), np.array([0, 0, 3]), 4)
        batch = {"labels": np.array([0]), "target_race": np.array([3]),
                 "items": np.array([[0, 4, 4, 1]]), "mask": np.array([[False, True, True, True]])}
        self.assertEqual(int(np.argmax(baselines.scores("most_popular", batch, [0]))), 2)
        self.assertEqual(int(np.argmax(baselines.scores("phase_popular", batch, [0]))), 1)
        self.assertEqual(int(np.argmax(baselines.scores("personal_popular", batch, [0]))), 3)
        self.assertEqual(int(np.argmax(baselines.scores("repeat_last", batch, [0]))), 0)
        first = baselines.scores("random", batch, [7])
        self.assertTrue(np.array_equal(first, baselines.scores("random", batch, [7])))


if __name__ == "__main__":
    unittest.main()
