"""Marathon finish-time forecast learned from real 2019 marathon preparations.

Features come from the runner's last 12 weeks of running (the same features at training and
serving time): weekly volume, runs per week, longest run, median pace, and Riegel's projection
of the fastest run of at least 10 km. A linear model maps them to the finish time.

Temporal protocol, as for the recommender: the model is fitted on the spring 2019 majors
(Boston, London) and evaluated on the autumn majors (Berlin, Chicago, New York), with each
runner's features taken 3 weeks before their race. The reported error and interval come from
that held-out autumn set.

    python -m marathon.forecast      # fit, evaluate and write models/marathon/forecast.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_DAYS = 84
HORIZON_DAYS = 21
MARATHON_KM = 42.195
FEATURES = ["riegel_minutes", "median_pace_x_marathon", "km_per_week", "runs_per_week", "longest_km"]
SPRING = ("BOSTON 2019", "LONDON 2019")
AUTUMN = ("BERLIN 2019", "CHICAGO 2019", "NEW YORK 2019")


def features(dates, km, minutes, as_of):
    """Features from runs strictly before `as_of` within WINDOW_DAYS. None if there is too little data."""
    dates = pd.to_datetime(pd.Series(dates)).dt.normalize().to_numpy()
    km = np.asarray(km, dtype=float)
    minutes = np.asarray(minutes, dtype=float)
    as_of = np.datetime64(pd.Timestamp(as_of).normalize())
    keep = (dates < as_of) & (dates >= as_of - np.timedelta64(WINDOW_DAYS, "D")) & (km >= 1)
    km, minutes = km[keep], minutes[keep]
    long = km >= 10
    if keep.sum() < 6 or not long.any():
        return None
    weeks = WINDOW_DAYS / 7
    return {"riegel_minutes": float((minutes[long] * (MARATHON_KM / km[long]) ** 1.06).min()),
            "median_pace_x_marathon": float(np.median(minutes / km) * MARATHON_KM),
            "km_per_week": float(km.sum() / weeks), "runs_per_week": float(len(km) / weeks),
            "longest_km": float(km.max())}


def design(rows):
    return np.column_stack([np.ones(len(rows))] + [[row[name] for row in rows] for name in FEATURES])


def predict(model, feature_row):
    """Predicted finish time and the held-out 80% interval, in minutes."""
    value = float(np.dot(model["coefficients"], [1.0] + [feature_row[name] for name in FEATURES]))
    low, high = model["residual_quantiles"]["p10"], model["residual_quantiles"]["p90"]
    return {"minutes": value, "low": value + low, "high": value + high}


def build_cases(sessions, races):
    groups = {athlete: frame for athlete, frame in sessions.groupby("athlete")}
    cases = []
    for race in races.itertuples():
        runner = groups.get(race.athlete)
        if runner is None:
            continue
        as_of = race.race_date - pd.Timedelta(days=HORIZON_DAYS)
        row = features(runner["date"], runner["km"], runner["minutes"], as_of)
        if row is not None:
            cases.append({"race": race.race, "finish": race.finish_minutes, **row})
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/processed/marathon"))
    parser.add_argument("--output", type=Path, default=Path("models/marathon/forecast.json"))
    args = parser.parse_args()
    sessions = pd.read_parquet(args.data / "sessions.parquet", columns=["athlete", "date", "km", "minutes", "is_race"])
    sessions = sessions[~sessions["is_race"]]
    races = pd.read_csv(args.data / "races.csv", parse_dates=["race_date"])
    cases = build_cases(sessions, races)
    train = [case for case in cases if case["race"] in SPRING]
    test = [case for case in cases if case["race"] in AUTUMN]
    y_train = np.array([case["finish"] for case in train])
    y_test = np.array([case["finish"] for case in test])
    coefficients = np.linalg.lstsq(design(train), y_train, rcond=None)[0]
    model = {"coefficients": coefficients.tolist()}
    residuals_train = y_train - design(train) @ coefficients
    model["residual_quantiles"] = {"p10": float(np.percentile(residuals_train, 10)),
                                   "p90": float(np.percentile(residuals_train, 90))}
    predicted = design(test) @ coefficients
    riegel = np.array([case["riegel_minutes"] for case in test])
    inside = ((y_test >= predicted + model["residual_quantiles"]["p10"])
              & (y_test <= predicted + model["residual_quantiles"]["p90"])).mean()
    model.update({
        "features": FEATURES, "window_days": WINDOW_DAYS, "horizon_days": HORIZON_DAYS,
        "train": {"races": list(SPRING), "cases": len(train)},
        "test": {"races": list(AUTUMN), "cases": len(test),
                 "mae_minutes": float(np.abs(y_test - predicted).mean()),
                 "mae_riegel_minutes": float(np.abs(y_test - riegel).mean()),
                 "mae_mean_minutes": float(np.abs(y_test - y_train.mean()).mean()),
                 "interval_coverage": float(inside)},
    })
    args.output.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: model[key] for key in ("train", "test", "residual_quantiles")}, indent=2))


if __name__ == "__main__":
    main()
