import math
import unittest

from strideplan.evaluation import evaluate, most_popular, ranking_metrics


class MetricsTest(unittest.TestCase):
    def test_default_cutoffs_fit_small_catalogue(self):
        result = evaluate({1: [1, 2, 3]}, {1: {2}})
        self.assertEqual(set(result), {"evaluated_users", "recall@1", "ndcg@1", "recall@3", "ndcg@3"})
        self.assertEqual(result["recall@1"], 0)
        self.assertEqual(result["recall@3"], 1)

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


if __name__ == "__main__":
    unittest.main()
