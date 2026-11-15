import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from strideplan_pipeline.pipeline import prepare


class PipelineTest(unittest.TestCase):
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
                stats = prepare(source, output)
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


if __name__ == "__main__":
    unittest.main()
