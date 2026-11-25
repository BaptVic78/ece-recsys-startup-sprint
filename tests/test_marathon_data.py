import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from marathon.data import assign_splits, clean_days, prepare, weekly
from marathon.validate import validate


def synthetic_grid(path, athletes=6):
    """One row per athlete per day for 2019. Athlete 0 runs Berlin and Boston, 5 has no race day."""
    dates = pd.date_range("2019-01-01", "2019-12-31")
    rows = []
    for athlete in range(athletes):
        major = {0: "BOSTON 2019,BERLIN 2019", 5: "BERLIN 2019"}.get(athlete, "BERLIN 2019,LONDON 2017")
        for day, date in enumerate(dates):
            km = 10.0 if day % 2 == 0 else 0.0
            if date == pd.Timestamp("2019-09-29") and athlete != 5:
                km = 42.5
            if date == pd.Timestamp("2019-04-15") and athlete == 0:
                km = 42.4
            if date == pd.Timestamp("2019-09-28"):
                km = 99.0  # the day before Berlin: must appear only as history, never as a label
            minutes = km * 5.0
            rows.append(dict(datetime=date, athlete=athlete, distance=km, duration=minutes, gender="F",
                             age_group="18 - 34", country="France", major=major))
    pd.DataFrame(rows).to_parquet(path, index=False)


class MarathonDataTest(unittest.TestCase):
    def test_prepare_extracts_labelled_races_without_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "grid.parquet"
            synthetic_grid(source)
            output = Path(directory) / "out"
            stats = prepare(source, output)
            races = pd.read_csv(output / "races.csv")
            self.assertEqual(stats["races"]["BERLIN 2019"], {"registered": 6, "no_marathon_day": 1,
                                                            "implausible_time": 0, "kept": 5})
            self.assertEqual(len(races), 6)
            self.assertEqual(set(races[races.athlete == 0]["race"]), {"BOSTON 2019", "BERLIN 2019"})
            self.assertTrue(races["finish_minutes"].between(212, 213).all())
            days = pd.read_parquet(output / "days.parquet")
            berlin = days[days.race_id == "1-BERLIN_2019"]
            self.assertEqual(berlin["days_before"].min(), 1)
            self.assertEqual(berlin.loc[berlin.days_before == 1, "km"].item(), 99.0)
            self.assertEqual(len(berlin), 140)
            boston = days[days.race_id == "0-BOSTON_2019"]
            self.assertEqual(len(boston), 104)
            validate(output)

    def test_splits_are_by_athlete_and_deterministic(self):
        splits = assign_splits(list(range(100)) * 2, seed=3)
        self.assertEqual(len(splits), 100)
        self.assertEqual(splits, assign_splits(range(100), seed=3))
        self.assertEqual(sorted(pd.Series(splits).value_counts().tolist()), [15, 15, 70])

    def test_implausible_days_are_zeroed(self):
        distance = np.array([[10.0, 10.0, 0.0, -1.0, np.nan]])
        duration = np.array([[50.0, 1.0, 30.0, 5.0, 5.0]])
        clean_distance, clean_duration, zeroed = clean_days(distance, duration)
        self.assertEqual(zeroed, 4)
        self.assertEqual(clean_distance.tolist(), [[10.0, 0.0, 0.0, 0.0, 0.0]])
        self.assertEqual(clean_duration.tolist(), [[50.0, 0.0, 0.0, 0.0, 0.0]])

    def test_weekly_keeps_only_complete_weeks(self):
        days = pd.DataFrame({"race_id": "r", "days_before": range(1, 11), "km": [5.0] * 10, "minutes": [25.0] * 10})
        weeks = weekly(days)
        self.assertEqual(weeks["weeks_before"].tolist(), [1])
        self.assertEqual(weeks.loc[0, "km"], 35.0)
        self.assertEqual(weeks.loc[0, "runs"], 7)
        self.assertEqual(weeks.loc[0, "pace_min_per_km"], 5.0)


if __name__ == "__main__":
    unittest.main()
