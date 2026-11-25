"""Check the exported marathon cases for leakage and consistency without rereading the raw file."""

import argparse
import json
from pathlib import Path

import pandas as pd

from marathon.data import MAX_HISTORY_DAYS, RACES


def validate(directory):
    races = pd.read_csv(directory / "races.csv")
    days = pd.read_parquet(directory / "days.parquet")
    weeks = pd.read_csv(directory / "weeks.csv")
    stats = json.loads((directory / "stats.json").read_text(encoding="utf-8"))
    errors = []
    if races["race_id"].duplicated().any():
        errors.append("duplicate race_id")
    if (races.groupby("athlete")["split"].nunique() > 1).any():
        errors.append("an athlete appears in more than one split")
    if not set(races["split"]) <= {"train", "validation", "test"}:
        errors.append("unknown split label")
    if not set(races["race"]) <= set(RACES):
        errors.append("unknown race")
    if not races["finish_minutes"].between(120, 420).all() or not races["race_km"].between(41, 47).all():
        errors.append("label outside the documented bounds")
    if len(races) != stats["labelled_races"]:
        errors.append("stats.json race count does not match races.csv")
    # The race day itself and anything after it must never be in the history.
    if days["days_before"].min() < 1 or days["days_before"].max() > MAX_HISTORY_DAYS:
        errors.append("history day outside 1..MAX_HISTORY_DAYS")
    if days.duplicated(["race_id", "days_before"]).any():
        errors.append("duplicate history day")
    if set(days["race_id"]) != set(races["race_id"]):
        errors.append("history and races do not cover the same cases")
    lengths = days.groupby("race_id").size()
    if not (lengths == races.set_index("race_id").loc[lengths.index, "history_days"]).all():
        errors.append("history length does not match history_days")
    if (days[["km", "minutes"]] < 0).any().any():
        errors.append("negative history values")
    if weeks["weeks_before"].min() < 1 or not set(weeks["race_id"]) <= set(races["race_id"]):
        errors.append("weekly table is inconsistent")
    if errors:
        raise ValueError("; ".join(errors))
    summary = {"verified_races": len(races), "verified_athletes": int(races["athlete"].nunique()),
               "history_rows": len(days), "week_rows": len(weeks)}
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/marathon"))
    validate(parser.parse_args().data)


if __name__ == "__main__":
    main()
