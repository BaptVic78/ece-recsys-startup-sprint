"""Build the session catalogue, the race calendar and the temporal split from BMClab daily data.

Users are runners who completed a 2019 World Marathon Major. Interactions are their running days
(one session per day, as the source stores daily totals). Each session is mapped to one of 66
session types: a distance band crossed with an intensity band, the intensity being the session's
pace relative to the runner's own median pace over the 28 days strictly before it.

The split is a global temporal cutoff: train before 1 August 2019, validation in August,
test from 1 September. Band edges are fixed in advance, so nothing is fitted on later data.
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
SPLIT_STARTS = {"validation": "2019-08-01", "test": "2019-09-01"}
MIN_SESSION_KM = 1.0
# Days whose average pace is outside this range (min/km) are treated as recording errors.
PACE_RANGE = (2.5, 15.0)
# Session types. Distance bands in km: [1,5), [5,8), ..., [35, inf). Intensity bands are the
# ratio of session pace to the runner's trailing median pace: below 0.90 is much faster than usual.
DISTANCE_EDGES = [5, 8, 10, 12, 15, 18, 21, 25, 30, 35]
INTENSITY_EDGES = [0.90, 0.95, 0.99, 1.03, 1.08]
INTENSITY_LABELS = ["much faster", "faster", "slightly faster", "usual", "slightly easier", "easier"]
REFERENCE_DAYS = 28
REFERENCE_MIN_RUNS = 3
NUM_ITEMS = (len(DISTANCE_EDGES) + 1) * (len(INTENSITY_EDGES) + 1)


def item_id(km, intensity):
    """Session type ID in 1..NUM_ITEMS. Values equal to an edge go to the higher band."""
    distance_band = np.searchsorted(DISTANCE_EDGES, km, side="right")
    intensity_band = np.searchsorted(INTENSITY_EDGES, intensity, side="right")
    return 1 + distance_band * (len(INTENSITY_EDGES) + 1) + intensity_band


def describe_item(item):
    """Human-readable distance range and intensity of a session type."""
    distance_band, intensity_band = divmod(int(item) - 1, len(INTENSITY_EDGES) + 1)
    edges = [MIN_SESSION_KM, *DISTANCE_EDGES]
    low = edges[distance_band]
    high = DISTANCE_EDGES[distance_band] if distance_band < len(DISTANCE_EDGES) else None
    pace_edges = [None, *INTENSITY_EDGES, None]
    return {"item": int(item), "km_min": float(low), "km_max": None if high is None else float(high),
            "intensity": INTENSITY_LABELS[intensity_band],
            "pace_ratio_min": pace_edges[intensity_band], "pace_ratio_max": pace_edges[intensity_band + 1]}


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
                         "athlete": int(meta.athlete[index]), "race": race,
                         "race_date": pd.Timestamp(date), "finish_minutes": float(minutes[index]),
                         "race_km": float(km[index])})
    return pd.DataFrame(rows), counts


def build_sessions(meta, dates, distance, duration, athletes):
    """Running days of the given athletes, with a causal pace reference and a session type."""
    rows = np.flatnonzero(meta["athlete"].isin(athletes).to_numpy())
    runner, day = np.nonzero(distance[rows] >= MIN_SESSION_KM)
    sessions = pd.DataFrame({"athlete": meta["athlete"].to_numpy()[rows][runner], "date": dates[day],
                             "km": distance[rows][runner, day], "minutes": duration[rows][runner, day]})
    return annotate_sessions(sessions)


def annotate_sessions(sessions):
    """Add pace, the causal 28-day pace reference, intensity and session type.

    Used both to build the training data and to process sessions logged through the API,
    so served histories are encoded exactly as in training. Expects at most one session per
    runner and day, with km of at least MIN_SESSION_KM.
    """
    sessions = sessions.sort_values(["athlete", "date"], kind="stable").reset_index(drop=True)
    sessions["date"] = pd.to_datetime(sessions["date"])
    sessions["pace"] = sessions["minutes"] / sessions["km"]
    # closed="left" excludes the session itself: the reference uses strictly earlier days only.
    reference = (sessions.set_index("date").groupby("athlete")["pace"]
                 .rolling(f"{REFERENCE_DAYS}D", closed="left", min_periods=REFERENCE_MIN_RUNS).median())
    sessions["reference_pace"] = reference.to_numpy()
    sessions["reference_missing"] = sessions["reference_pace"].isna()
    intensity = (sessions["pace"] / sessions["reference_pace"]).fillna(1.0)
    sessions["intensity"] = intensity
    sessions["item"] = item_id(sessions["km"].to_numpy(), intensity.to_numpy())
    return sessions


def add_race_calendar(sessions, races):
    """Days until each session's next race. The race date is declared in advance; its result is not used."""
    calendar = races[["athlete", "race_id", "race_date"]].sort_values("race_date")
    calendar["race_date"] = calendar["race_date"].astype("datetime64[ns]")
    sessions = sessions.assign(date=sessions["date"].astype("datetime64[ns]"))
    merged = pd.merge_asof(sessions.sort_values("date"), calendar, left_on="date", right_on="race_date",
                           by="athlete", direction="forward", allow_exact_matches=True)
    merged = merged.sort_values(["athlete", "date"], kind="stable").reset_index(drop=True)
    merged["days_to_race"] = (merged["race_date"] - merged["date"]).dt.days
    merged["is_race"] = merged["days_to_race"] == 0
    return merged.drop(columns=["race_date"])


def assign_split(dates):
    split = np.full(len(dates), "train", dtype=object)
    split[dates >= pd.Timestamp(SPLIT_STARTS["validation"])] = "validation"
    split[dates >= pd.Timestamp(SPLIT_STARTS["test"])] = "test"
    return split


def prepare(source, output):
    meta, dates, distance, duration = load_grid(source)
    distance, duration, zeroed = clean_days(distance, duration)
    races, counts = find_races(meta, dates, distance, duration)
    if races.empty:
        raise ValueError("No labelled races found")
    sessions = build_sessions(meta, dates, distance, duration, races["athlete"].unique())
    sessions = add_race_calendar(sessions, races)
    sessions["split"] = assign_split(sessions["date"])
    output.mkdir(parents=True, exist_ok=True)
    races.to_csv(output / "races.csv", index=False)
    sessions.to_parquet(output / "sessions.parquet", index=False)
    targets = sessions[~sessions["is_race"]]
    stats = {"source": str(source), "athletes_in_source": int(len(meta)), "implausible_days_zeroed": zeroed,
             "races": counts, "labelled_races": int(len(races)), "users": int(sessions["athlete"].nunique()),
             "sessions": int(len(sessions)), "race_day_sessions": int(sessions["is_race"].sum()),
             "split_sessions": targets["split"].value_counts().to_dict(),
             "split_users": targets.groupby("split")["athlete"].nunique().to_dict(),
             "sessions_without_pace_reference": int(sessions["reference_missing"].sum()),
             "items": NUM_ITEMS, "items_seen_in_train": int(targets.loc[targets.split == "train", "item"].nunique()),
             "split_policy": f"global temporal cutoff: validation from {SPLIT_STARTS['validation']}, "
                             f"test from {SPLIT_STARTS['test']}",
             "item_definition": "distance band x pace relative to the runner's median pace over the 28 days strictly before"}
    (output / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/bmclab/run_ww_2019_d.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/marathon"))
    args = parser.parse_args()
    prepare(args.input, args.output)


if __name__ == "__main__":
    main()
