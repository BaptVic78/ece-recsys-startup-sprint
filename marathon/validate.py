"""Check the exported sessions for leakage and consistency without rereading the raw file."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from marathon.data import (MIN_SESSION_KM, NUM_ITEMS, RACES, SPLIT_STARTS, add_race_calendar, annotate_sessions,
                           assign_split, plausible_pace)


def validate(directory):
    races = pd.read_csv(directory / "races.csv", parse_dates=["race_date"])
    sessions = pd.read_parquet(directory / "sessions.parquet")
    stats = json.loads((directory / "stats.json").read_text(encoding="utf-8"))
    errors = []
    if races["race_id"].duplicated().any() or not set(races["race"]) <= set(RACES):
        errors.append("race table is inconsistent")
    if not races["finish_minutes"].between(120, 420).all():
        errors.append("race result outside the documented bounds")
    if sessions.duplicated(["athlete", "date"]).any():
        errors.append("more than one session per runner and day")
    if not sessions.groupby("athlete")["date"].is_monotonic_increasing.all():
        errors.append("sessions are not in chronological order")
    if not sessions["item"].between(1, NUM_ITEMS).all():
        errors.append("session type outside the catalogue")
    # Temporal split: the label must follow the date alone, with train strictly before validation and test.
    if (sessions["split"].to_numpy() != assign_split(sessions["date"])).any():
        errors.append("split label does not match the global date cutoffs")
    if sessions.loc[sessions.split == "train", "date"].max() >= pd.Timestamp(SPLIT_STARTS["validation"]):
        errors.append("a training session falls after the validation cutoff")
    if (sessions["days_to_race"].dropna() < 0).any():
        errors.append("a session points to a race in the past")
    race_days = sessions[sessions["is_race"]].merge(races, on="race_id")
    if len(race_days) != len(races) or not (race_days["date"] == race_days["race_date"]).all():
        errors.append("race-day sessions do not match the race table")
    if len(sessions) != stats["sessions"] or len(races) != stats["labelled_races"]:
        errors.append("stats.json counts do not match the exports")
    if not plausible_pace(sessions["km"], sessions["minutes"]).all() or (sessions["km"] < MIN_SESSION_KM).any():
        errors.append("a session breaks the distance or pace quality rule")
    # Recompute every derived column from the raw km and minutes, so a missing, extra or edited
    # session shows up as a mismatch in the causal pace reference, session type or race calendar.
    expected = add_race_calendar(annotate_sessions(sessions[["athlete", "date", "km", "minutes"]].copy()), races)
    actual = sessions.sort_values(["athlete", "date"], kind="stable").reset_index(drop=True)
    for column in ("reference_pace", "intensity", "days_to_race"):
        if not np.allclose(actual[column], expected[column], equal_nan=True):
            errors.append(f"{column} does not match its recomputation from km, minutes and dates")
    for column in ("item", "is_race", "reference_missing"):
        if not (actual[column].to_numpy() == expected[column].to_numpy()).all():
            errors.append(f"{column} does not match its recomputation from km, minutes and dates")
    if not actual["race_id"].fillna("").equals(expected["race_id"].fillna("")):
        errors.append("race_id does not match the race calendar")
    if errors:
        raise ValueError("; ".join(errors))
    summary = {"verified_sessions": len(sessions), "verified_users": int(sessions["athlete"].nunique()),
               "verified_races": len(races)}
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/marathon"))
    validate(parser.parse_args().data)


if __name__ == "__main__":
    main()
