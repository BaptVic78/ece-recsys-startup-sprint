import unittest

from fastapi.testclient import TestClient

from strideplan.api import app


class ApiTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_mock_recommendations_respect_budget_and_top_k(self):
        response = self.client.post("/recommend", json={"user_id": "demo", "max_duration_minutes": 40, "top_k": 1})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["mode"], "mock")
        self.assertFalse(result["personalized"])
        self.assertEqual(len(result["recommendations"]), 1)
        self.assertTrue(all(item["duration_minutes"] <= 40 for item in result["recommendations"]))

    def test_empty_candidates_and_invalid_requests(self):
        response = self.client.post("/recommend", json={"max_duration_minutes": 10})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["recommendations"], [])
        for payload in ({}, {"max_duration_minutes": -1}, {"max_duration_minutes": 40, "sport": "bike"},
                        {"max_duration_minutes": 40, "top_k": True}, {"max_duration_minutes": 40, "prompt": "hello"}):
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post("/recommend", json=payload).status_code, 422)

    def test_health_and_openapi(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok", "mode": "mock"})
        self.assertIn("/recommend", self.client.get("/openapi.json").json()["paths"])


if __name__ == "__main__":
    unittest.main()
