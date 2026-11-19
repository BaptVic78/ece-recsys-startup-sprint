import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from strideplan.api import create_app


class MockApiTest(unittest.TestCase):
    """Runs with FastAPI only: no bundle means no torch import."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        missing = Path(self.directory.name) / "missing"
        self.client = TestClient(create_app(bundle_dir=missing, sessions_path=missing / "sessions.csv"))

    def tearDown(self):
        self.directory.cleanup()

    def test_mock_recommendations_respect_budget_and_top_k(self):
        response = self.client.post("/recommend", json={"user_id": "demo", "max_duration_minutes": 40, "top_k": 1})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["mode"], "mock")
        self.assertFalse(result["personalized"])
        self.assertEqual(len(result["recommendations"]), 1)
        self.assertTrue(all(item["duration_max_minutes"] <= 40 for item in result["recommendations"]))

    def test_empty_candidates_and_invalid_requests(self):
        response = self.client.post("/recommend", json={"max_duration_minutes": 10})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["recommendations"], [])
        bad_session = {"start_timestamp": 0, "duration_seconds": -5}
        for payload in ({}, {"max_duration_minutes": -1}, {"max_duration_minutes": 40, "sport": "bike"},
                        {"max_duration_minutes": 40, "top_k": True}, {"max_duration_minutes": 40, "prompt": "hello"},
                        {"max_duration_minutes": 40, "history": [bad_session]}):
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post("/recommend", json=payload).status_code, 422)

    def test_health_and_openapi(self):
        self.assertEqual(self.client.get("/health").json(),
                         {"status": "ok", "mode": "mock", "model_version": None, "known_users": 0})
        paths = self.client.get("/openapi.json").json()["paths"]
        self.assertIn("/recommend", paths)
        self.assertIn("/catalogue", paths)


class ModelApiTest(unittest.TestCase):
    """Requires PyTorch: serves a small untrained bundle built from synthetic sessions."""

    def setUp(self):
        from test_inference import build_bundle
        self.directory = tempfile.TemporaryDirectory()
        build_bundle(self.directory.name)
        self.client = TestClient(create_app(bundle_dir=self.directory.name,
                                            sessions_path=Path(self.directory.name) / "sessions.csv"))

    def tearDown(self):
        self.directory.cleanup()

    def test_known_user_gets_personalized_model_ranking(self):
        health = self.client.get("/health").json()
        self.assertEqual(health["mode"], "model")
        self.assertEqual(health["known_users"], 1)
        result = self.client.post("/recommend", json={"user_id": "runner", "max_duration_minutes": 360,
                                                      "top_k": 3}).json()
        self.assertEqual((result["mode"], result["strategy"], result["personalized"]), ("model", "gru", True))
        self.assertEqual(result["model_version"], health["model_version"])
        self.assertEqual(result["history_sessions_used"], 4)
        self.assertEqual(len(result["recommendations"]), 3)
        scores = [item["score"] for item in result["recommendations"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_unknown_user_falls_back_to_popularity(self):
        result = self.client.post("/recommend", json={"user_id": "new", "max_duration_minutes": 360}).json()
        self.assertEqual((result["strategy"], result["personalized"]), ("global_popularity", False))
        self.assertTrue(all(item["score"] is None for item in result["recommendations"]))

    def test_inline_history_replaces_stored_history(self):
        history = [{"start_timestamp": day * 86400, "duration_seconds": 1800, "avg_speed_mps": 3.0}
                   for day in range(3)]
        result = self.client.post("/recommend", json={"user_id": "runner", "max_duration_minutes": 360,
                                                      "history": history}).json()
        self.assertEqual((result["strategy"], result["history_sessions_used"]), ("gru", 3))

    def test_catalogue_lists_every_category(self):
        catalogue = self.client.get("/catalogue").json()
        self.assertEqual([item["template_id"] for item in catalogue],
                         [str(index) for index in range(1, len(catalogue) + 1)])


if __name__ == "__main__":
    unittest.main()
