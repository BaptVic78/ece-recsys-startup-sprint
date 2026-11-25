"""Extract labelled 2019 marathons and the training that preceded them from BMClab daily data.

A case is one runner's World Marathon Major in 2019. Its label is the race-day duration.
Its history is the daily distance and duration in the days strictly before race day.
Splits are by athlete, so a runner's races never appear in two splits.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Race dates of the six 2019 World Marathon Majors.
RACES = {
    "TOKYO 2019": "2019-03-03",
    "BOSTON 2019": "2019-04-15",
    "LONDON 2019": "2019-04-28",
    "BERLIN 2019": "2019-09-29",
    "CHICAGO 2019": "2019-10-13",
    "NEW YORK 2019": "2019-11-03",
}
MAX_HISTORY_DAYS = 140
# Days whose average pace is outside this range (min/km) are treated as recording errors.
PACE_RANGE = (2.5, 15.0)


def load_grid(path):
    """Return metadata and dense athlete x day arrays. The file has one row per athlete per day."""
    frame = pd.read_parquet(path, columns=["datetime", "athlete", "distance", "duration",
                                           "gender", "age_group", "country", "major"])
    frame = frame.sort_values(["athlete", "datetime"], kind="stable")
    dates = np.sort(frame["datetime"].unique())
    athletes = frame["athlete"].unique()
    if len(frame) != len(dates) * len(athletes):
        raise ValueError("Expected exactly one row per athlete per day")
    if not (frame["datetime"].to_numpy().reshape(len(athletes), -1) == dates).all():
        raise ValueError("Every athlete must cover the same sorted calendar")
    distance = frame["distance"].to_numpy(np.float64).reshape(len(athletes), -1)
    duration = frame["duration"].to_numpy(np.float64).reshape(len(athletes), -1)
    meta = frame.iloc[::len(dates)][["athlete", "gender", "age_group", "country", "major"]].reset_index(drop=True)
    return meta, pd.DatetimeIndex(dates), distance, duration


def clean_days(distance, duration):
    """Zero out non-finite, negative or implausible-pace days. Return cleaned copies and a count."""
    distance, duration = distance.copy(), duration.copy()
    bad = ~np.isfinite(distance) | ~np.isfinite(duration) | (distance < 0) | (duration < 0)
    moving = (distance > 0) & ~bad
    with np.errstate(divide="ignore", invalid="ignore"):
        pace = np.where(moving, duration / np.where(moving, distance, 1), np.nan)
    bad |= moving & ~((pace >= PACE_RANGE[0]) & (pace <= PACE_RANGE[1]))
    bad |= (distance == 0) & (duration > 0)
    distance[bad] = 0
    duration[bad] = 0
    return distance, duration, int(bad.sum())


def find_races(meta, dates, distance, duration, km_range=(41.0, 47.0), minutes_range=(120.0, 420.0)):
    """One row per runner and 2019 major with a marathon-length day on the race date."""
    rows = []
    counts = {}
    majors = meta["major"].fillna("").to_numpy()
    for race, date in RACES.items():
        day = dates.get_loc(pd.Timestamp(date))
        registered = np.array([race in [item.strip() for item in value.split(",")] for value in majors])
        km, minutes = distance[:, day], duration[:, day]
        length_ok = registered & (km >= km_range[0]) & (km <= km_range[1])
        found = length_ok & (minutes >= minutes_range[0]) & (minutes <= minutes_range[1])
        counts[race] = {"registered": int(registered.sum()), "no_marathon_day": int((registered & ~length_ok).sum()),
                        "implausible_time": int((length_ok & ~found).sum()), "kept": int(found.sum())}
        for index in np.flatnonzero(found):
            rows.append({"race_id": f"{meta.athlete[index]}-{race.replace(' ', '_')}",
                         "athlete": int(meta.athlete[index]), "race": race, "race_date": date,
                         "race_day_index": day, "row": index,
                         "finish_minutes": float(minutes[index]), "race_km": float(km[index]),
                         "gender": meta.gender[index], "age_group": meta.age_group[index],
                         "country": meta.country[index]})
    return pd.DataFrame(rows), counts


def history_windows(races, distance, duration, max_days=MAX_HISTORY_DAYS):
    """Daily history strictly before race day. days_before=1 is the day before the race."""
    parts = []
    for race in races.itertuples():
        available = min(max_days, race.race_day_index)
        days_before = np.arange(1, available + 1)
        columns = race.race_day_index - days_before
        parts.append(pd.DataFrame({"race_id": race.race_id, "days_before": days_before,
                                   "km": distance[race.row, columns], "minutes": duration[race.row, columns]}))
    return pd.concat(parts, ignore_index=True)


def weekly(days):
    """Aggregate daily history into weeks. weeks_before=1 covers days_before 1..7."""
    frame = days.assign(weeks_before=(days["days_before"] - 1) // 7 + 1, run=days["km"] >= 1)
    complete = frame.groupby(["race_id", "weeks_before"])["days_before"].transform("size") == 7
    frame = frame[complete]
    grouped = frame.groupby(["race_id", "weeks_before"])
    result = grouped.agg(km=("km", "sum"), minutes=("minutes", "sum"), runs=("run", "sum"),
                         longest_km=("km", "max")).reset_index()
    result["pace_min_per_km"] = np.where(result["km"] > 0, result["minutes"] / result["km"].where(result["km"] > 0), np.nan)
    return result


def assign_splits(athletes, seed=42, validation=0.15, test=0.15):
    unique = np.array(sorted(set(athletes)))
    order = np.random.default_rng(seed).permutation(len(unique))
    n_test, n_validation = round(len(unique) * test), round(len(unique) * validation)
    split = np.full(len(unique), "train", dtype=object)
    split[order[:n_test]] = "test"
    split[order[n_test:n_test + n_validation]] = "validation"
    return dict(zip(unique.tolist(), split.tolist()))


def prepare(source, output, seed=42):
    meta, dates, distance, duration = load_grid(source)
    distance, duration, zeroed = clean_days(distance, duration)
    races, counts = find_races(meta, dates, distance, duration)
    if races.empty:
        raise ValueError("No labelled races found")
    days = history_windows(races, distance, duration)
    weeks = weekly(days)
    splits = assign_splits(races["athlete"], seed)
    races["split"] = races["athlete"].map(splits)
    races["history_days"] = np.minimum(MAX_HISTORY_DAYS, races["race_day_index"])
    races = races.drop(columns=["row", "race_day_index"])
    output.mkdir(parents=True, exist_ok=True)
    races.to_csv(output / "races.csv", index=False)
    days.to_parquet(output / "days.parquet", index=False)
    weeks.to_csv(output / "weeks.csv", index=False)
    stats = {"source": str(source), "athletes_in_source": int(len(meta)), "days_in_source": int(len(dates)),
             "implausible_days_zeroed": zeroed, "races": counts, "labelled_races": int(len(races)),
             "labelled_athletes": int(races["athlete"].nunique()),
             "split_races": races["split"].value_counts().to_dict(),
             "split_athletes": races.drop_duplicates("athlete")["split"].value_counts().to_dict(),
             "finish_minutes_quantiles": races["finish_minutes"].quantile([0, .05, .25, .5, .75, .95, 1]).round(1)
                                                                .rename(lambda q: f"p{round(q * 100)}").to_dict(),
             "history_days": {"max": MAX_HISTORY_DAYS, "races_with_16_weeks": int((races["history_days"] >= 112).sum())},
             "label_definition": "race-day total duration in minutes for a 41-47 km day on the race date",
             "split_policy": f"by athlete, seed {seed}, 70/15/15"}
    (output / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/bmclab/run_ww_2019_d.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/marathon"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.input, args.output, args.seed)


if __name__ == "__main__":
    main()
