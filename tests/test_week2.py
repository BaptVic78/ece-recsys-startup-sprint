import math
import unittest

import torch
from fastapi.testclient import TestClient

from strideplan.api import app
from strideplan.evaluation import evaluate, most_popular, ranking_metrics
from strideplan.model import WorkoutRanker


class MetricsTest(unittest.TestCase):
    def test_known_answers(self):
        self.assertEqual(ranking_metrics(["a", "b"], {"a"}, 1), {"recall": 1, "ndcg": 1})
        self.assertEqual(ranking_metrics(["a", "b"], {"b"}, 1), {"recall": 0, "ndcg": 0})
        result = ranking_metrics(["a", "b"], {"b"}, 10)
        self.assertEqual(result["recall"], 1)
        self.assertAlmostEqual(result["ndcg"], 1 / math.log2(3))
        result = ranking_metrics(["a", "b", "c"], {"a", "c"}, 2)
        self.assertEqual(result["recall"], 0.5)
        self.assertAlmostEqual(result["ndcg"], 1 / (1 + 1 / math.log2(3)))
        self.assertEqual(ranking_metrics([], {"a"}, 10), {"recall": 0, "ndcg": 0})

    def test_macro_average_and_input_contract(self):
        result = evaluate({"u": ["a"], "v": ["b"]}, {"u": {"a"}, "v": {"a"}}, (1,))
        self.assertEqual(result, {"evaluated_users": 2, "recall@1": 0.5, "ndcg@1": 0.5})
        for ranking, relevant, k in [([1, 1], {1}, 2), ([1], set(), 2), ([1], {1}, 0), ([1], {1}, True)]:
            with self.assertRaises(ValueError):
                ranking_metrics(ranking, relevant, k)
        with self.assertRaises(ValueError):
            evaluate({"u": [1]}, {"v": {1}})
        with self.assertRaises(ValueError):
            evaluate({}, {})

    def test_popularity_ranks_candidates_from_training_counts(self):
        self.assertEqual(most_popular([[2, 2, 3], [1, 2]], [4, 3, 2, 1]), [2, 1, 3, 4])


class ModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = WorkoutRanker(3)

    def test_forward_backward_and_history_dependence(self):
        items = torch.tensor([[1, 2], [3, 2]])
        features = torch.zeros(2, 2, 5)
        ranking, effort = self.model(items, features, torch.tensor([2, 2]))
        self.assertEqual(ranking.shape, (2, 3))
        self.assertEqual(effort.shape, (2, 3))
        self.assertFalse(torch.allclose(ranking[0], ranking[1]))
        loss = torch.nn.functional.cross_entropy(ranking, torch.tensor([1, 2]))
        loss += torch.nn.functional.binary_cross_entropy_with_logits(effort, torch.zeros_like(effort))
        loss.backward()
        for parameter in self.model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_padding_and_empty_history(self):
        features = torch.zeros(1, 1, 5)
        short = self.model(torch.tensor([[1]]), features, torch.tensor([1]))
        padded = self.model(torch.tensor([[1, 0, 0]]), torch.cat((features, torch.ones(1, 2, 5)), dim=1), torch.tensor([1]))
        for first, second in zip(short, padded):
            torch.testing.assert_close(first, second)
        ranking, effort = self.model(torch.zeros(1, 1, dtype=torch.long), features, torch.tensor([0]))
        torch.testing.assert_close(ranking, self.model.ranking_bias[None, :])
        self.assertTrue(torch.isfinite(effort).all())

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            self.model(torch.tensor([[4]]), torch.zeros(1, 1, 5), torch.tensor([1]))
        with self.assertRaises(ValueError):
            self.model(torch.tensor([[1]]), torch.zeros(1, 1, 5), torch.tensor([2]))
        with self.assertRaises(ValueError):
            self.model(torch.tensor([[1]]), torch.full((1, 1, 5), float("nan")), torch.tensor([1]))


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
