import contextlib
import csv
import gzip
import io
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from strideplan_pipeline.pipeline import normalize, prepare
from strideplan_pipeline.validate import validate
from strideplan_pipeline.download import main as download


class PipelineTest(unittest.TestCase):
    def test_download_publishes_only_complete_files(self):
        for size in (4, 10):
            with self.subTest(expected_size=size), tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / "nested" / "raw.json"
                response = io.BytesIO(b"data")
                response.headers = {"Content-Length": str(size)}
                with patch("sys.argv", ["download", "--output", str(target)]), \
                     patch("urllib.request.urlopen", return_value=response), \
                     contextlib.redirect_stdout(io.StringIO()):
                    if size == 4:
                        download()
                        self.assertEqual(target.read_bytes(), b"data")
                        self.assertFalse(target.with_suffix(".json.part").exists())
                    else:
                        with self.assertRaisesRegex(OSError, "Incomplete download"):
                            download()
                        self.assertFalse(target.exists())

    def test_gzipped_json_and_train_only_user(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.json.gz"
            with gzip.open(source, "wt") as stream:
                stream.write(json.dumps(dict(userId=1, id=2, sport="run", timestamp=[100, 180])))
            with contextlib.redirect_stdout(io.StringIO()):
                stats = prepare(source, Path(directory) / "output", min_duration=1)
            self.assertEqual(stats["split_sessions"], {"train": 1})
            self.assertEqual(stats["evaluation_users"], 0)

    def test_filter_duration_duplicates_and_temporal_splits(self):
        def session(identifier, start, user="a", sport="run"):
            return dict(userId=user, id=identifier, sport=sport, timestamp=[start, start + 60])

        records = [session("3", 300), session("1", 100), session("2", 200),
                   session("4", 300), session("1", 100), session("5", 400, sport="bike"),
                   session("6", 100, user="short"),
                   dict(userId="a", id="bad", sport="run", timestamp=[200, 100])]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.json"
            source.write_text("\n".join(repr(record) for record in records) + "\ninvalid\n")
            output = Path(directory) / "output"
            with contextlib.redirect_stdout(io.StringIO()):
                stats = prepare(source, output, min_duration=1, min_train_sessions=1)
            with (output / "sessions.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(stats["retained_sessions"], 5)
            self.assertEqual(stats["records"]["duplicate_workout"], 1)
            self.assertEqual(stats["records"]["unordered_timestamps"], 1)
            self.assertEqual(stats["records"]["malformed_record"], 1)
            self.assertEqual(stats["evaluation_users"], 1)
            self.assertEqual([row["split"] for row in rows if row["user_id"] == "a"],
                             ["train", "validation", "test", "test"])
            self.assertTrue(all(float(row["duration_seconds"]) == 60 for row in rows))
            self.assertEqual(json.loads((output / "stats.json").read_text()), stats)
            validate(output)

    def test_features_have_explicit_units_and_missing_values(self):
        record = dict(userId=1, id=2, sport="run", timestamp=[100, 400, 700],
                      latitude=[0, 0, 0], longitude=[0, 0.01, 0.02],
                      heart_rate=[100, 120, 140], altitude=[10, 25, 20], gender="female")
        row = normalize(record)
        self.assertAlmostEqual(row["distance_meters"], 2223.9, delta=0.1)
        self.assertAlmostEqual(row["avg_speed_mps"], row["distance_meters"] / 600)
        self.assertEqual(row["avg_heart_rate_bpm"], 120)
        self.assertEqual(row["elevation_gain_meters"], 15)
        self.assertEqual(row["gender"], "female")
        record["latitude"] = [0, 0]
        record["heart_rate"] = [0, 120, 140]
        record["altitude"] = [10, float("nan"), 20]
        row = normalize(record)
        for field in ("distance_meters", "avg_speed_mps", "avg_heart_rate_bpm", "elevation_gain_meters"):
            self.assertIsNone(row[field])

    def test_default_duration_limits_and_huge_timestamps(self):
        for duration in (1, 299, 21601, 3 * 86400):
            with self.subTest(duration=duration), self.assertRaisesRegex(ValueError, "duration_out_of_range"):
                normalize(dict(userId=1, id=2, sport="run", timestamp=[100, 100 + duration]))
        for duration in (300, 21600):
            self.assertEqual(normalize(dict(userId=1, id=2, sport="run", timestamp=[100, 100 + duration]))["duration_seconds"], duration)
        with self.assertRaisesRegex(ValueError, "invalid_timestamps"):
            normalize(dict(userId=1, id=2, sport="run", timestamp=[100, 10**400]))

    def test_default_history_requirement_and_validator_detects_corruption(self):
        records = [dict(userId=user, id=f"{user}-{index}", sport="run",
                        timestamp=[1000 * (index + 1), 1000 * (index + 1) + 600])
                   for user, size in (("short", 6), ("long", 7)) for index in range(size)]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.json"
            # Include a corrupt record and change syntax mid-stream.
            lines = [repr(dict(userId=1, id="bad", sport="run", timestamp=[100, 10**400]))]
            lines.extend(json.dumps(record) if index % 2 else repr(record) for index, record in enumerate(records))
            source.write_text("\n".join(lines))
            output = Path(directory) / "output"
            with contextlib.redirect_stdout(io.StringIO()):
                stats = prepare(source, output)
            self.assertEqual(stats["evaluation_users"], 1)
            self.assertEqual(stats["records"]["invalid_timestamps"], 1)
            self.assertEqual(stats["split_sessions"], {"train": 11, "validation": 1, "test": 1})
            validate(output)
            with (output / "train.csv").open("a") as stream:
                stream.write((output / "test.csv").read_text().splitlines()[1] + "\n")
            with self.assertRaises(ValueError):
                validate(output)


if __name__ == "__main__":
    unittest.main()
