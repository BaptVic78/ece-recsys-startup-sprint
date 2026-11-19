"""Service, retraining and simulation on a small synthetic bundle. Requires PyTorch."""

import json
import statistics
import tempfile
import unittest
from pathlib import Path

import torch

from test_inference import build_bundle
from strideplan.bundle import load_bundle, read_registry, resolve_bundle_dir
from strideplan.inference import Recommender, load_histories
from strideplan.reranker import NO_FEEDBACK, Policy
from strideplan.retrain import RetrainConfig, register_base, retrain, save_candidate
from strideplan.service import StridePlanService
from strideplan.simulate import build_runners, run_arm, typical_sessions
from strideplan.retrain import featurize_histories
from strideplan.store import FeedbackError, Store


DAY = 86400
NOW = 20 * DAY


class LoopTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        build_bundle(self.directory.name)
        self.bundle = load_bundle(self.directory.name)
        self.histories = load_histories(Path(self.directory.name) / "sessions.csv")
        self.store = Store(":memory:")
        self.service = StridePlanService(Recommender(self.bundle), self.store, self.histories,
                                         Policy(epsilon=0.0), seed=0)

    def tearDown(self):
        self.store.close()
        self.directory.cleanup()

    def test_recommendation_is_logged_and_feedback_changes_the_next_one(self):
        first = self.service.recommend("runner", 360, 3, now=NOW)
        served = self.store.recommendation(first["rec_id"])
        self.assertEqual([item["template_id"] for item in served["items"]],
                         [item["template_id"] for item in first["recommendations"]])
        self.assertEqual(served["model_version"], self.bundle.version)
        top = first["recommendations"][0]["template_id"]
        self.service.record_feedback(first["rec_id"], top, "rejected", now=NOW + 60)
        second = self.service.recommend("runner", 360, 3, now=NOW + 120)
        self.assertNotEqual(second["recommendations"][0]["template_id"], top)
        self.assertTrue(second["feedback_applied"])
        self.assertEqual(second["feedback_events_used"], 1)

    def test_completed_session_enters_history(self):
        before = self.service.recommend("runner", 360, 3, now=NOW)
        template = before["recommendations"][0]["template_id"]
        session = {"start_timestamp": NOW + 600, "duration_seconds": 2400, "avg_speed_mps": 3.0}
        result = self.service.record_feedback(before["rec_id"], template, "completed", "ok", session, now=NOW + 4000)
        self.assertTrue(result["session_logged"])
        self.assertEqual(len(self.service.history("runner")), len(self.histories["runner"]) + 1)
        self.assertEqual(self.store.sessions("runner")[0]["rec_id"], before["rec_id"])

    def test_logged_session_without_heart_rate_uses_runner_median(self):
        self.service.record_session("runner", {"start_timestamp": NOW, "duration_seconds": 2400, "avg_speed_mps": 3.0})
        logged = self.service.history("runner")[-1]
        known = [float(r["avg_heart_rate_bpm"]) for r in self.histories["runner"] if r["avg_heart_rate_bpm"]]
        self.assertEqual(logged["avg_heart_rate_bpm"], statistics.median(known))
        self.assertIsNone(self.store.sessions("runner")[0]["avg_heart_rate_bpm"])

    def test_sessions_need_a_user_and_a_completion(self):
        anonymous = self.service.recommend(None, 360, 2, now=NOW)
        template = anonymous["recommendations"][0]["template_id"]
        session = {"start_timestamp": NOW, "duration_seconds": 2400, "avg_speed_mps": 3.0}
        with self.assertRaises(FeedbackError):
            self.service.record_feedback(anonymous["rec_id"], template, "completed", session=session, now=NOW)
        named = self.service.recommend("runner", 360, 2, now=NOW)
        with self.assertRaises(FeedbackError):
            self.service.record_feedback(named["rec_id"], named["recommendations"][0]["template_id"], "rejected",
                                         session=session, now=NOW)

    def test_exploration_marks_one_slot(self):
        service = StridePlanService(Recommender(self.bundle), self.store, self.histories, Policy(epsilon=1.0), seed=0)
        result = service.recommend("runner", 360, 3, now=NOW)
        self.assertEqual([item["explored"] for item in result["recommendations"]], [False, False, True])
        self.assertTrue(self.store.recommendation(result["rec_id"])["items"][2]["explored"])

    def give_feedback(self, rounds=12):
        for index in range(rounds):
            now = NOW + index * DAY
            result = self.service.recommend("runner", 360, 3, now=now)
            items = [item["template_id"] for item in result["recommendations"]]
            self.service.record_feedback(result["rec_id"], items[0], "rejected", now=now + 60)
            effort = ("too_hard", "ok", "too_easy")[index % 3]
            session = {"start_timestamp": now + 600, "duration_seconds": 1800, "avg_speed_mps": 2.5}
            self.service.record_feedback(result["rec_id"], items[1], "completed", effort, session, now=now + 3600)

    def test_retraining_learns_from_feedback_and_registry_promotes_or_keeps(self):
        self.give_feedback()
        config = RetrainConfig(epochs=2, min_feedback=10, min_effort_labels=5, learning_rate=1e-2)
        candidate, report = retrain(self.bundle, self.histories, self.store, config)
        self.assertTrue(report["effort_head_trained"])
        self.assertEqual(report["feedback"]["train_events"] + report["feedback"]["holdout_events"], 24)
        self.assertGreater(report["feedback"]["holdout_from"], NOW)
        changed = any(not torch.equal(a, b) for a, b in zip(candidate.state_dict().values(),
                                                              self.bundle.model.state_dict().values()))
        self.assertTrue(changed)
        with tempfile.TemporaryDirectory() as models:
            register_base(models, self.directory.name, self.bundle)
            self.assertEqual(read_registry(models)["current"], self.bundle.version)
            version = save_candidate(models, candidate, self.bundle, report)
            saved = load_bundle(Path(models) / version)
            self.assertEqual(saved.lineage, (self.bundle.version,))
            self.assertEqual(saved.catalogue, self.bundle.catalogue)
            self.assertEqual(saved.effort_head_trained, report["serve_effort"])
            payload = json.loads((Path(models) / version / "bundle.json").read_text(encoding="utf-8"))
            self.assertIn("promoted", payload["retrain"])
            self.assertEqual(resolve_bundle_dir(models), Path(models) / self.bundle.version)

    def test_retraining_needs_enough_feedback(self):
        with self.assertRaises(ValueError):
            retrain(self.bundle, self.histories, self.store, RetrainConfig())

    def test_simulated_arms_share_runners_and_report_metrics(self):
        rows = featurize_histories(self.histories, self.bundle.catalogue, self.bundle.scaler, self.bundle.max_speed)
        runners = build_runners(rows, self.bundle.catalogue, 5, -0.5, seed=1)
        self.assertEqual(list(runners), ["runner"])
        config = RetrainConfig(epochs=1, min_feedback=2, min_effort_labels=1)
        for arm in ("no_loop", "short_loop", "full_loop"):
            result = run_arm(arm, self.bundle, self.histories, runners, 4, 1, 3, config,
                             typical_sessions(rows))
            self.assertEqual(len(result["rounds"]), 4)
            for metric in ("acceptance", "mrr", "list_quality", "comfortable_share", "too_hard_share"):
                self.assertTrue(0 <= result["rounds"][-1][metric] <= 1)
            self.assertEqual(result["retrain"] is not None, arm == "full_loop")

    def test_no_feedback_policy_ignores_feedback(self):
        service = StridePlanService(Recommender(self.bundle), self.store, self.histories, NO_FEEDBACK)
        first = service.recommend("runner", 360, 3, now=NOW)
        service.record_feedback(first["rec_id"], first["recommendations"][0]["template_id"], "rejected", now=NOW + 1)
        second = service.recommend("runner", 360, 3, now=NOW + 2)
        self.assertEqual([i["template_id"] for i in first["recommendations"]],
                         [i["template_id"] for i in second["recommendations"]])


if __name__ == "__main__":
    unittest.main()
