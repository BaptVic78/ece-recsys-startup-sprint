import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from strideplan.experiment import examples, fit_catalogue, prepare_rows, template_id, tensors


class ExperimentTest(unittest.TestCase):
    def write_rows(self, directory, future_duration=1000, future_speed=3):
        rows = [dict(user_id="runner", workout_id=str(index), start_timestamp=index * 1000,
                     duration_seconds=600 + index * 30, avg_speed_mps=2 + index / 10,
                     avg_heart_rate_bpm="" if index == 2 else 130 + index,
                     split="train") for index in range(1, 8)]
        rows.extend([dict(user_id="runner", workout_id="validation", start_timestamp=8000,
                          duration_seconds=future_duration, avg_speed_mps=future_speed,
                          avg_heart_rate_bpm=299, split="validation"),
                     dict(user_id="runner", workout_id="test", start_timestamp=9000,
                          duration_seconds=future_duration, avg_speed_mps=future_speed,
                          avg_heart_rate_bpm=299, split="test")])
        path = Path(directory) / "sessions.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_fit_ignores_heldout_values_and_imputes_training_median(self):
        with tempfile.TemporaryDirectory() as directory:
            histories, catalogue, scaler, _ = prepare_rows(self.write_rows(directory))
            _, changed_catalogue, changed_scaler, _ = prepare_rows(self.write_rows(directory, 20000, 9))
            self.assertEqual(catalogue, changed_catalogue)
            self.assertEqual(scaler, changed_scaler)
            self.assertEqual(scaler["medians"][3], 134.5)
            self.assertTrue(all(np.isfinite(row["features"]).all() for row in histories["runner"]))
            cases = examples(histories, "test")
            dataset = tensors(cases, 10)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.tensors[2].tolist(), [8])
            self.assertEqual(cases[0]["context"][-1]["split"], "validation")

    def test_invalid_speed_target_is_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            histories, _, _, rejected = prepare_rows(self.write_rows(directory, future_speed=390))
            self.assertEqual(rejected, {"missing_or_invalid_speed": 2})
            self.assertEqual(examples(histories, "test"), [])

    def test_context_is_strictly_past_and_never_contains_test_events(self):
        rows = [dict(user="runner", time=time, split=split, template=1)
                for time, split in [(1, "train"), (2, "train"), (2, "train"),
                                    (3, "validation"), (4, "test"), (5, "test")]]
        training = examples({"runner": rows}, "train")
        self.assertTrue(all(all(row["time"] < case["target_time"] for row in case["context"]) for case in training))
        for case in examples({"runner": rows}, "test", minimum_training_dates=1):
            self.assertTrue(all(row["time"] < case["target_time"] for row in case["context"]))
            self.assertNotIn("test", [row["split"] for row in case["context"]])
        self.assertEqual(examples({"runner": rows}, "test"), [])

    def test_constant_training_values_create_one_template(self):
        catalogue = fit_catalogue([dict(duration=600, speed=3)] * 10)
        self.assertEqual(catalogue["num_templates"], 1)
        self.assertEqual(template_id(dict(duration=600, speed=3), catalogue), 1)


if __name__ == "__main__":
    unittest.main()
