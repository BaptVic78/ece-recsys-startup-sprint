"""Store and short-loop re-ranking. Python standard library only."""

import unittest

from strideplan.reranker import NO_FEEDBACK, Policy, effort_bias, intensity, rerank
from strideplan.store import FeedbackError, Store


CATALOGUE = {"duration_edges_seconds": [2400.0], "speed_edges_mps": [2.5], "num_templates": 4,
             "duration_range_seconds": [600.0, 6000.0], "speed_range_mps": [2.0, 4.0]}
DAY = 86400


def ranking(*scores):
    return [{"template_id": template, "score": score, "predicted_effort": None}
            for template, score in enumerate(scores, start=1)]


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")
        self.rec_id = self.store.log_recommendation(
            "runner", 100.0, "gru-x", "gru", 45,
            [{"position": 0, "template_id": 3, "score": 0.5, "explored": False, "model_position": 0},
             {"position": 1, "template_id": 1, "score": 0.2, "explored": True, "model_position": 2}])

    def tearDown(self):
        self.store.close()

    def test_feedback_is_joined_with_the_served_list(self):
        self.store.add_feedback(self.rec_id, 1, "completed", "too_hard", 200.0)
        [event] = self.store.feedback("runner")
        self.assertEqual((event["template_id"], event["position"], event["effort"]), (1, 1, "too_hard"))
        self.assertEqual((event["served_at"], event["model_version"]), (100.0, "gru-x"))
        self.assertEqual(self.store.counts(), {"recommendations": 1, "feedback": 1, "sessions": 0})

    def test_invalid_feedback_is_refused(self):
        cases = [("missing", 3, "accepted", None, 200.0), (self.rec_id, 2, "accepted", None, 200.0),
                 (self.rec_id, 3, "liked", None, 200.0), (self.rec_id, 3, "rejected", "ok", 200.0),
                 (self.rec_id, 3, "completed", "hard", 200.0), (self.rec_id, 3, "accepted", None, 50.0)]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(FeedbackError):
                self.store.add_feedback(*case)

    def test_sessions_are_deduplicated_and_ordered(self):
        session = {"workout_id": "w1", "start_timestamp": 500.0, "duration_seconds": 1800, "avg_speed_mps": 3.0}
        self.assertTrue(self.store.add_session("runner", session, 600.0))
        self.assertFalse(self.store.add_session("runner", session, 700.0))
        self.store.add_session("runner", {**session, "workout_id": "w0", "start_timestamp": 400.0}, 700.0)
        self.assertEqual([row["workout_id"] for row in self.store.sessions("runner")], ["w0", "w1"])
        self.assertEqual(self.store.sessions("other"), [])


class RerankerTest(unittest.TestCase):
    def test_intensity_orders_light_to_hard(self):
        self.assertEqual([intensity(t, CATALOGUE) for t in range(1, 5)], [-1.0, 0.0, 0.0, 1.0])

    def test_recent_rejection_lowers_a_category_and_fades(self):
        feedback = [{"action": "rejected", "template_id": 1, "effort": None, "created_at": 0.0}]
        fresh = rerank(ranking(0.4, 0.3, 0.2, 0.1), feedback, CATALOGUE, now=DAY)
        self.assertNotEqual(fresh[0]["template_id"], 1)
        self.assertIn("lowered: you skipped this recently", next(i for i in fresh if i["template_id"] == 1)["reasons"])
        old = rerank(ranking(0.4, 0.3, 0.2, 0.1), feedback, CATALOGUE, now=200 * DAY)
        self.assertEqual(old[0]["template_id"], 1)

    def test_effort_buttons_shift_intensity(self):
        hard = [{"action": "completed", "template_id": 4, "effort": "too_hard", "created_at": 0.0}]
        easy = [{**hard[0], "effort": "too_easy"}]
        base = ranking(0.25, 0.25, 0.25, 0.25)
        self.assertEqual(rerank(base, hard, CATALOGUE, now=DAY)[0]["template_id"], 1)
        self.assertEqual(rerank(base, easy, CATALOGUE, now=DAY)[0]["template_id"], 4)
        self.assertLess(effort_bias(hard, DAY, Policy()), 0)
        ok = [{**hard[0], "effort": "ok"}]
        self.assertEqual([i["template_id"] for i in rerank(base, ok, CATALOGUE, now=DAY)], [1, 2, 3, 4])

    def test_learned_effort_prefers_comfortable_categories(self):
        base = [{"template_id": 1, "score": 0.5, "predicted_effort": 0.95},
                {"template_id": 2, "score": 0.4, "predicted_effort": 0.5}]
        self.assertEqual(rerank(base, [], CATALOGUE, now=0)[0]["template_id"], 2)

    def test_no_feedback_policy_keeps_model_order(self):
        feedback = [{"action": "rejected", "template_id": 1, "effort": None, "created_at": 0.0},
                    {"action": "completed", "template_id": 4, "effort": "too_hard", "created_at": 0.0}]
        result = rerank(ranking(0.4, 0.3, 0.2, 0.1), feedback, CATALOGUE, now=DAY, policy=NO_FEEDBACK)
        self.assertEqual([i["template_id"] for i in result], [1, 2, 3, 4])
        self.assertTrue(all(not item["reasons"] for item in result))

    def test_popularity_ranking_without_scores_can_be_adjusted(self):
        popular = [{"template_id": t, "score": None, "predicted_effort": None} for t in (4, 3, 2, 1)]
        feedback = [{"action": "rejected", "template_id": 4, "effort": None, "created_at": 0.0}]
        self.assertNotEqual(rerank(popular, feedback, CATALOGUE, now=DAY)[0]["template_id"], 4)


if __name__ == "__main__":
    unittest.main()
