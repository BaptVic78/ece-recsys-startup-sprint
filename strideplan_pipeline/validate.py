"""Check prepared CSVs without reading the raw Endomondo download again."""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from strideplan_pipeline.pipeline import FIELDS, FEATURES, percentiles


def read_csv(path, fields):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != fields:
            raise ValueError(f"Unexpected columns in {path}")
        return list(reader)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(output):
    stats = json.loads((output / "stats.json").read_text(encoding="utf-8"))
    rows = read_csv(output / "sessions.csv", FIELDS)
    require(bool(rows), "No sessions to validate")
    by_id = {row["workout_id"]: row for row in rows}
    require(len(by_id) == len(rows), "Duplicate workout IDs in sessions.csv")
    groups = defaultdict(lambda: defaultdict(list))
    bounds = stats.get("duration_bounds_seconds", {"min": 0, "max": math.inf})
    for row in rows:
        duration = float(row["duration_seconds"])
        timestamp = float(row["start_timestamp"])
        require(math.isfinite(duration) and duration > 0 and bounds["min"] <= duration <= bounds["max"], "Invalid duration")
        require(math.isfinite(timestamp) and timestamp > 0, "Invalid start timestamp")
        require(row["sport"] == "run" and row["user_id"] and row["workout_id"], "Invalid session identity or sport")
        require(row["split"] in ("train", "validation", "test"), "Unknown split")
        for field in FEATURES[:-1]:
            if row[field]:
                value = float(row[field])
                require(math.isfinite(value) and value >= 0, f"Invalid {field}")
        groups[row["user_id"]][row["split"]].append(timestamp)

    exported_ids = []
    for split in ("train", "validation", "test"):
        exported = read_csv(output / f"{split}.csv", FIELDS)
        for row in exported:
            require(row["split"] == split, f"Wrong split label in {split}.csv")
            require(by_id.get(row["workout_id"]) == row, f"Split row differs from sessions.csv in {split}")
            exported_ids.append(row["workout_id"])
    require(Counter(exported_ids) == Counter(by_id.keys()), "Splits overlap or do not cover all sessions")

    eligible_users = set()
    minimum = stats.get("min_train_sessions", 1)
    for user_id, splits in groups.items():
        train = splits["train"]
        require(bool(train), f"Missing training history for {user_id}")
        all_times = {value for values in splits.values() for value in values}
        if len(all_times) >= minimum + 2:
            eligible_users.add(user_id)
            validation = splits["validation"]
            test = splits["test"]
            require(bool(validation) and bool(test), f"Missing holdout for {user_id}")
            require(len(set(train)) >= minimum, f"Insufficient training history for {user_id}")
            require(len(set(validation)) == len(set(test)) == 1, f"Multiple holdout start times for {user_id}")
            require(max(train) < min(validation) and max(validation) < min(test), f"Temporal leakage for {user_id}")
        else:
            require(not splits["validation"] and not splits["test"], f"Short history in evaluation for {user_id}")

    users = read_csv(output / "users.csv", ["user_id", "sessions", "evaluation_eligible"])
    require(len(users) == len(groups) and {row["user_id"] for row in users} == set(groups), "User summary mismatch")
    for row in users:
        user_id = row["user_id"]
        require(int(row["sessions"]) == sum(map(len, groups[user_id].values())), "User session count mismatch")
        require(row["evaluation_eligible"] == str(user_id in eligible_users), "User eligibility mismatch")
    require(stats["retained_sessions"] == len(rows) and stats["retained_users"] == len(groups), "Stats totals mismatch")
    require(stats["evaluation_users"] == len(eligible_users), "Evaluation user count mismatch")
    actual_counts = Counter(row["split"] for row in rows)
    require(actual_counts == Counter(stats["split_sessions"]), "Split statistics mismatch")
    counts = stats["records"]
    require(counts["records_read"] == len(rows) + sum(value for key, value in counts.items() if key != "records_read"), "Record accounting mismatch")
    if "missing_features" in stats:
        require(stats["missing_features"] == {field: sum(not row[field] for row in rows) for field in FEATURES}, "Missing feature counts mismatch")
    if "duration_percentiles_seconds" in stats:
        require(stats["duration_percentiles_seconds"] == percentiles([float(row["duration_seconds"]) for row in rows]), "Duration percentiles mismatch")
    return dict(verified_sessions=len(rows), verified_users=len(groups), evaluation_users=len(eligible_users))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    print(json.dumps(validate(args.output), indent=2))


if __name__ == "__main__":
    main()
