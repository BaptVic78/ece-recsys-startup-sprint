import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from marathon.data import NUM_ITEMS, clean_days, describe_item, item_id, load_grid, plausible_pace, prepare
from marathon.validate import validate


def synthetic_grid(path, skip_date=None):
    """Four runners over 2019. Runner 0 runs Boston and Berlin, runner 3 never has a race day."""
    dates = pd.date_range("2019-01-01", "2019-12-31")
    if skip_date is not None:
        dates = dates[dates != pd.Timestamp(skip_date)]
    rows = []
    for athlete in range(4):
        major = {0: "BOSTON 2019,BERLIN 2019", 3: "BERLIN 2019"}.get(athlete, "BERLIN 2019,LONDON 2017")
        for day, date in enumerate(dates):
            km = 10.0 if day % 2 == 0 else 0.0
            minutes = km * 5.0
            if date == pd.Timestamp("2019-01-11"):
                minutes = km * 4.0  # faster than the 5 min/km reference built from earlier days
            if date == pd.Timestamp("2019-09-29") and athlete != 3:
                km, minutes = 42.5, 212.5
            if date == pd.Timestamp("2019-04-15") and athlete == 0:
                km, minutes = 42.4, 212.0
            rows.append(dict(datetime=date, athlete=athlete, distance=km, duration=minutes, gender="F",
                             age_group="18 - 34", country="France", major=major))
    pd.DataFrame(rows).to_parquet(path, index=False)


class MarathonDataTest(unittest.TestCase):
    def test_prepare_builds_temporal_split_and_causal_items(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "grid.parquet"
            synthetic_grid(source)
            output = Path(directory) / "out"
            stats = prepare(source, output)
            self.assertEqual(stats["races"]["BERLIN 2019"]["kept"], 3)
            self.assertEqual(stats["users"], 3)
            sessions = pd.read_parquet(output / "sessions.parquet")
            runner = sessions[sessions.athlete == 1].set_index("date")
            # The first sessions have fewer than three earlier runs, so no reference pace.
            self.assertTrue(runner.loc["2019-01-01", "reference_missing"])
            # On 11 January the reference is the median of strictly earlier days (5 min/km).
            self.assertAlmostEqual(runner.loc["2019-01-11", "reference_pace"], 5.0)
            self.assertAlmostEqual(runner.loc["2019-01-11", "intensity"], 0.8)
            self.assertEqual(runner.loc["2019-01-11", "item"], item_id(10.0, 0.8))
            self.assertEqual(runner.loc["2019-09-28", "days_to_race"], 1)
            self.assertTrue(runner.loc["2019-09-29", "is_race"])
            self.assertTrue(np.isnan(runner.loc["2019-10-02", "days_to_race"]))
            boston_runner = sessions[sessions.athlete == 0].set_index("date")
            self.assertEqual(boston_runner.loc["2019-04-13", "race_id"], "0-BOSTON_2019")
            self.assertEqual(boston_runner.loc["2019-04-17", "race_id"], "0-BERLIN_2019")
            self.assertEqual(runner.loc["2019-07-30", "split"], "train")
            self.assertEqual(runner.loc["2019-08-01", "split"], "validation")
            self.assertEqual(runner.loc["2019-09-02", "split"], "test")
            validate(output)

    def test_a_date_missing_for_every_athlete_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "grid.parquet"
            synthetic_grid(source, skip_date="2019-09-28")
            with self.assertRaisesRegex(ValueError, "consecutive days"):
                load_grid(source)

    def test_validation_recomputes_derived_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "grid.parquet"
            synthetic_grid(source)
            output = Path(directory) / "out"
            prepare(source, output)
            path = output / "sessions.parquet"
            sessions = pd.read_parquet(path)
            # Make one mid-season session slower without touching the derived columns:
            # its later sessions' pace reference no longer matches the export.
            row = sessions.index[(sessions.athlete == 1) & (sessions.date == pd.Timestamp("2019-02-02"))][0]
            sessions.loc[row, "minutes"] = 70.0
            sessions.to_parquet(path, index=False)
            with self.assertRaisesRegex(ValueError, "does not match its recomputation"):
                validate(output)

    def test_shared_pace_rule(self):
        self.assertEqual(plausible_pace([10, 1, 10, 0, 10], [50, 900, 20, 5, np.nan]).tolist(),
                         [True, False, False, False, False])

    def test_item_ids_cover_the_catalogue(self):
        ids = item_id(np.array([1.0, 4.99, 5.0, 50.0, 50.0]), np.array([0.5, 1.0, 0.90, 0.5, 2.0]))
        self.assertEqual(ids.tolist(), [1, 4, 8, NUM_ITEMS - 5, NUM_ITEMS])
        self.assertEqual(describe_item(1)["km_min"], 1.0)
        self.assertEqual(describe_item(NUM_ITEMS), {"item": NUM_ITEMS, "km_min": 35.0, "km_max": None,
                                                     "intensity": "easier", "pace_ratio_min": 1.08,
                                                     "pace_ratio_max": None})

    def test_implausible_days_are_zeroed(self):
        distance = np.array([[10.0, 10.0, 0.0, -1.0, np.nan]])
        duration = np.array([[50.0, 1.0, 30.0, 5.0, 5.0]])
        clean_distance, clean_duration, zeroed = clean_days(distance, duration)
        self.assertEqual(zeroed, 4)
        self.assertEqual(clean_distance.tolist(), [[10.0, 0.0, 0.0, 0.0, 0.0]])
        self.assertEqual(clean_duration.tolist(), [[50.0, 0.0, 0.0, 0.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
