import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from strideplan.bundle import load_bundle, write_bundle
from strideplan.experiment import prepare_rows
from strideplan.features import FEATURE_NAMES, describe_template
from strideplan.inference import Recommender, encode_history, load_histories
from strideplan.model import WorkoutRanker


def write_sessions(directory):
    rows = [dict(user_id="runner", workout_id=str(index), start_timestamp=index * 86400,
                 duration_seconds=1200 + index * 300, avg_speed_mps=2 + index / 5,
                 avg_heart_rate_bpm="" if index == 3 else 130 + index, split="train") for index in range(1, 13)]
    rows.append(dict(user_id="runner", workout_id="bad-gps", start_timestamp=13 * 86400, duration_seconds=1800,
                     avg_speed_mps="", avg_heart_rate_bpm=140, split="validation"))
    path = Path(directory) / "sessions.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def build_bundle(directory, max_history=4):
    histories, catalogue, scaler, _ = prepare_rows(write_sessions(directory))
    torch.manual_seed(0)
    model = WorkoutRanker(catalogue["num_templates"], feature_size=len(FEATURE_NAMES), hidden_size=8)
    torch.save(model.state_dict(), Path(directory) / "ranker.pt")
    popularity = list(range(catalogue["num_templates"], 0, -1))
    write_bundle(directory, model, catalogue, scaler, popularity, max_history, 10.0)
    return histories, model


class InferenceTest(unittest.TestCase):
    def test_bundle_round_trip_restores_identical_model(self):
        with tempfile.TemporaryDirectory() as directory:
            _, model = build_bundle(directory)
            bundle = load_bundle(directory)
            self.assertTrue(bundle.version.startswith("gru-"))
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(bundle.model.state_dict()[name], tensor)
            self.assertFalse(bundle.model.training)

    def test_tampered_weights_or_features_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            build_bundle(directory)
            path = Path(directory) / "bundle.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["preprocessing"]["features"] = ["other"] * len(FEATURE_NAMES)
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_bundle(directory)
        with tempfile.TemporaryDirectory() as directory:
            build_bundle(directory)
            torch.save({}, Path(directory) / "ranker.pt")
            with self.assertRaises(ValueError):
                load_bundle(directory)

    def test_serving_features_match_training_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            histories, _ = build_bundle(directory)
            bundle = load_bundle(directory)
            records = load_histories(Path(directory) / "sessions.csv")["runner"]
            templates, features = encode_history(list(reversed(records)), bundle.catalogue, bundle.scaler,
                                                 bundle.max_speed, bundle.max_history)
            expected = histories["runner"][-4:]
            self.assertEqual(templates, [row["template"] for row in expected])
            np.testing.assert_allclose(np.stack(features), np.stack([row["features"] for row in expected]))

    def test_cold_start_uses_global_popularity(self):
        with tempfile.TemporaryDirectory() as directory:
            build_bundle(directory)
            recommender = Recommender(load_bundle(directory))
            for records in ([], [dict(start_timestamp=1, duration_seconds=600, avg_speed_mps=None)]):
                result = recommender.rank(records)
                self.assertEqual(result["strategy"], "global_popularity")
                self.assertEqual([template for template, _ in result["ranking"]], recommender.bundle.popularity)

    def test_history_ranking_covers_catalogue_with_probabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            build_bundle(directory)
            recommender = Recommender(load_bundle(directory))
            records = load_histories(Path(directory) / "sessions.csv")["runner"]
            result = recommender.rank(records)
            self.assertEqual(result["strategy"], "gru")
            self.assertEqual(result["history_sessions_used"], 4)
            ranking = result["ranking"]
            self.assertEqual(sorted(template for template, _ in ranking),
                             list(range(1, recommender.catalogue["num_templates"] + 1)))
            scores = [score for _, score in ranking]
            self.assertAlmostEqual(sum(scores), 1.0, places=5)
            self.assertEqual(scores, sorted(scores, reverse=True))

    def test_budget_keeps_reachable_categories_and_caps_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            build_bundle(directory)
            recommender = Recommender(load_bundle(directory))
            records = load_histories(Path(directory) / "sessions.csv")["runner"]
            budget = recommender.templates[recommender.catalogue["num_templates"]]["duration_min_minutes"] - 1
            result = recommender.recommend(records, budget, 50)
            self.assertTrue(result["recommendations"])
            for item in result["recommendations"]:
                self.assertLessEqual(item["duration_min_minutes"], budget)
                self.assertLessEqual(item["duration_max_minutes"], budget)
            self.assertEqual(len(recommender.recommend(records, 360, 2)["recommendations"]), 2)
            self.assertEqual(recommender.recommend(records, 0.1, 5)["recommendations"], [])

    def test_template_description_uses_band_boundaries(self):
        catalogue = {"duration_edges_seconds": [2400.0], "speed_edges_mps": [2.5], "num_templates": 4,
                     "duration_range_seconds": [600.0, 6000.0], "speed_range_mps": [2.0, 4.0]}
        self.assertEqual(describe_template(4, catalogue),
                         {"template_id": 4, "duration_min_minutes": 40.0, "duration_max_minutes": 100.0,
                          "speed_min_kmh": 9.0, "speed_max_kmh": 14.4, "title": "Run 40-100 min at 9.0-14.4 km/h"})
        with self.assertRaises(ValueError):
            describe_template(5, catalogue)


if __name__ == "__main__":
    unittest.main()
