import json
import os
import tempfile
import time
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

os.environ["MARATHON_SKIP_APP"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

from marathon.api import create_app  # noqa: E402
from marathon.data import NUM_ITEMS, annotate_sessions, plausible_pace  # noqa: E402
from marathon.model import ContextCrossAttentionRecommender  # noqa: E402
from marathon.sequences import RACE_BUCKETS, SessionStore, item_attributes  # noqa: E402
from marathon.serving import Recommender, prescription  # noqa: E402


def save_small_model(directory):
    torch.manual_seed(0)
    model = ContextCrossAttentionRecommender(NUM_ITEMS, item_attributes(), 32, dim=16, heads=2, layers=1, dropout=0.0)
    torch.save(model.state_dict(), directory / "full.pt")
    config = {"model": "full", "use_race_context": True, "max_history": 32, "num_items": NUM_ITEMS,
              "dim": 16, "heads": 2, "layers": 1, "dropout": 0.0, "best_epoch": 1}
    (directory / "full.json").write_text(json.dumps(config), encoding="utf-8")
    counts = np.ones((RACE_BUCKETS, NUM_ITEMS), dtype=int)
    counts[:, 7] = 50
    (directory / "phase_popularity.json").write_text(json.dumps({"counts": counts.tolist()}), encoding="utf-8")


def history(days=30, start=date(2019, 8, 1)):
    return [{"date": start + timedelta(days=offset), "km": 8.0 + offset % 5, "minutes": (8.0 + offset % 5) * 5.5}
            for offset in range(0, days, 2)]


class ServingTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        save_small_model(Path(self.directory.name))
        self.recommender = Recommender(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_served_encoding_matches_training_code(self):
        sessions = history()
        target = date(2019, 9, 1)
        frame = self.recommender._frame(sessions, target, date(2019, 9, 29))
        expected = annotate_sessions(pd.DataFrame([{"athlete": "runner", **s} for s in sessions]
                                                  + [{"athlete": "runner", "date": target, "km": 10.0, "minutes": 60.0}]))
        self.assertTrue((frame["item"].to_numpy() == expected["item"].to_numpy()).all())
        self.assertEqual(frame["days_to_race"].iloc[-1], 28)
        batch = SessionStore(frame).batch(np.array([len(frame) - 1]))
        self.assertEqual(int(batch["target_race"][0]), 5)

    def test_future_sessions_are_ignored_and_cold_start_uses_fallback(self):
        target = date(2019, 9, 1)
        with_future = history() + [{"date": date(2019, 9, 5), "km": 30.0, "minutes": 150.0}]
        first = self.recommender.recommend(history(), target, date(2019, 9, 29))
        second = self.recommender.recommend(with_future, target, date(2019, 9, 29))
        self.assertEqual(first["recommendations"], second["recommendations"])
        self.assertEqual(first["strategy"], "model")
        cold = self.recommender.recommend(history(days=6), target, date(2019, 9, 29))
        self.assertEqual(cold["strategy"], "phase_popularity")
        self.assertEqual(cold["recommendations"][0]["item"], 8)

    def test_implausible_sessions_never_reach_the_reference_or_the_model(self):
        target = date(2019, 9, 1)
        clean = self.recommender.recommend(history(), target, date(2019, 9, 29))
        noisy = self.recommender.recommend(history() + [{"date": date(2019, 8, 30), "km": 1.0, "minutes": 900.0}],
                                           target, date(2019, 9, 29))
        self.assertEqual(clean["recommendations"], noisy["recommendations"])
        self.assertLess(noisy["reference_pace"], 15)
        self.assertTrue(all(item["minutes"] < 400 for item in noisy["recommendations"]))

    def test_plan_rolls_forward_and_prescription_is_concrete(self):
        days = self.recommender.plan(history(), [date(2019, 9, 3), date(2019, 9, 1)], date(2019, 9, 29))
        self.assertEqual([day["date"] for day in days], ["2019-09-01", "2019-09-03"])
        self.assertEqual(days[1]["history_sessions"], days[0]["history_sessions"] + 1)
        item = prescription(NUM_ITEMS, 5.0)
        self.assertEqual(item["km"], 37.0)
        self.assertEqual(item["pace"], "5:33/km")


class ApiTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        save_small_model(root)
        demo = root / "demo"
        demo.mkdir()
        pd.DataFrame([{"runner": 1, "race": "BERLIN 2019", "race_date": "2019-09-29", "today": "2019-09-01",
                       "finish_minutes": 230.0}]).to_csv(demo / "runners.csv", index=False)
        pd.DataFrame([{"runner": 1, **s} for s in history(days=40)]).to_csv(demo / "sessions.csv", index=False)
        self.client = TestClient(create_app(root, ":memory:", demo))

    def tearDown(self):
        self.directory.cleanup()

    def test_recommend_plan_and_log_interaction_end_to_end(self):
        sessions = [{**s, "date": str(s["date"])} for s in history()]
        response = self.client.post("/users", json={"user_id": "lea", "race_date": "2019-09-29",
                                                    "goal_minutes": 225, "sessions": sessions})
        self.assertEqual(response.status_code, 200)
        started = time.perf_counter()
        result = self.client.post("/recommend", json={"user_id": "lea", "date": "2019-09-01", "top_k": 5}).json()
        self.assertLess(time.perf_counter() - started, 2.0)
        self.assertEqual(len(result["recommendations"]), 5)
        self.assertEqual(result["strategy"], "model")
        self.assertEqual(result["goal_pace"], "5:20/km")
        top = result["recommendations"][0]
        logged = self.client.post("/log_interaction", json={
            "user_id": "lea", "type": "completed", "date": "2019-09-01", "rec_id": result["rec_id"],
            "item": top["item"], "km": 12.0, "minutes": 66.0, "effort": "ok"})
        self.assertEqual(logged.status_code, 200)
        user = self.client.get("/users/lea").json()
        self.assertEqual(user["interactions"][0]["type"], "completed")
        self.assertEqual(user["sessions"][-1]["date"], "2019-09-01")
        self.assertEqual(user["next_day"], "2019-09-02")
        missed = self.client.post("/log_interaction", json={"user_id": "lea", "type": "missed", "date": "2019-09-02"})
        self.assertEqual(missed.status_code, 200)
        plan = self.client.post("/plan", json={"user_id": "lea", "days": 7, "run_weekdays": [1, 3, 5, 6]}).json()
        self.assertEqual(plan["start_date"], "2019-09-03")
        self.assertTrue(all(day["date"] >= "2019-09-03" for day in plan["days"]))
        self.assertTrue(all(len(day["recommendations"]) == 3 for day in plan["days"]))

    def test_every_recommendation_can_be_logged_back_at_both_pace_bounds(self):
        # Reviewer's case: one 3 km / 45 min session (15 min/km); and the fast bound, 2.5 min/km.
        for user_id, km, minutes in (("slow", 3.0, 45.0), ("fast", 10.0, 25.0)):
            with self.subTest(user=user_id):
                self.client.post("/users", json={"user_id": user_id, "race_date": "2019-09-29",
                                                 "sessions": [{"date": "2019-08-31", "km": km, "minutes": minutes}]})
                rec = self.client.post("/recommend", json={"user_id": user_id, "date": "2019-09-01",
                                                           "top_k": NUM_ITEMS}).json()
                for item in rec["recommendations"]:
                    self.assertTrue(plausible_pace(item["km"], item["minutes"]), item)
                    self.assertTrue(plausible_pace(item["km"], item["km"] * item["pace_min_per_km"]), item)
                top = rec["recommendations"][0]
                logged = self.client.post("/log_interaction", json={
                    "user_id": user_id, "type": "completed", "date": "2019-09-01", "rec_id": rec["rec_id"],
                    "item": top["item"], "km": top["km"], "minutes": top["minutes"]})
                self.assertEqual(logged.status_code, 200, logged.text)

    def test_invalid_interactions_are_rejected(self):
        self.client.post("/users", json={"user_id": "lea", "race_date": "2019-09-29"})
        rec = self.client.post("/recommend", json={"user_id": "lea", "date": "2019-09-01", "top_k": 3}).json()
        bad = [
            {"user_id": "lea", "type": "completed", "date": "2019-09-01"},
            {"user_id": "lea", "type": "missed", "date": "2019-09-01", "effort": "too_hard"},
            {"user_id": "lea", "type": "skipped", "date": "2019-09-01", "rec_id": "nope"},
            {"user_id": "lea", "type": "skipped", "date": "2019-09-01", "rec_id": rec["rec_id"],
             "item": next(i for i in range(1, NUM_ITEMS + 1) if i not in [r["item"] for r in rec["recommendations"]])},
            {"user_id": "nobody", "type": "missed", "date": "2019-09-01"},
            {"user_id": "lea", "type": "liked", "date": "2019-09-01"},
            {"user_id": "lea", "type": "completed", "date": "2019-09-01", "km": 1.0, "minutes": 900.0},
        ]
        codes = [self.client.post("/log_interaction", json=payload).status_code for payload in bad]
        self.assertEqual(codes, [422, 422, 422, 422, 404, 422, 422])
        implausible = {"user_id": "max", "sessions": [{"date": "2019-08-01", "km": 1.0, "minutes": 900.0}]}
        self.assertEqual(self.client.post("/users", json=implausible).status_code, 422)
        self.assertEqual(self.client.get("/users/max").status_code, 404)
        self.assertEqual(self.client.get("/users/lea").json()["interactions"], [])

    def test_demo_runner_loads_only_the_past(self):
        runners = self.client.get("/demo/runners").json()
        self.assertEqual(runners[0]["runner"], 1)
        loaded = self.client.post("/demo/1/load").json()
        self.assertEqual(loaded["user_id"], "demo-1")
        user = self.client.get("/users/demo-1").json()
        self.assertTrue(all(s["date"] < "2019-09-01" for s in user["sessions"]))
        self.assertEqual(self.client.get("/health").json()["status"], "ok")

    def test_loading_a_demo_again_restarts_it(self):
        loaded = self.client.post("/demo/1/load").json()
        rec = self.client.post("/recommend", json={"user_id": "demo-1", "date": loaded["today"], "top_k": 3}).json()
        self.client.post("/log_interaction", json={"user_id": "demo-1", "type": "completed", "date": loaded["today"],
                                                   "rec_id": rec["rec_id"], "km": 8.0, "minutes": 45.0})
        reloaded = self.client.post("/demo/1/load").json()
        user = self.client.get("/users/demo-1").json()
        self.assertEqual(reloaded["today"], loaded["today"])
        self.assertTrue(all(s["date"] < loaded["today"] for s in user["sessions"]))
        self.assertEqual(user["interactions"], [])
        self.assertLessEqual(user["next_day"], loaded["today"])


if __name__ == "__main__":
    unittest.main()
