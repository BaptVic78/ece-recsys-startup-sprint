"""Prepare chronological running histories from the public Endomondo export."""

import argparse
import ast
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


FIELDS = ["user_id", "workout_id", "sport", "start_timestamp", "duration_seconds", "split"]


def normalize(record):
    if not isinstance(record, dict):
        raise ValueError("invalid_record")
    if record.get("sport") != "run":
        return None
    identifiers = [record.get("userId"), record.get("id")]
    if any(isinstance(value, bool) or not isinstance(value, (int, str)) or not str(value).strip() for value in identifiers):
        raise ValueError("invalid_identifier")
    timestamps = record.get("timestamp")
    if not isinstance(timestamps, list) or len(timestamps) < 2:
        raise ValueError("missing_timestamps")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in timestamps):
        raise ValueError("invalid_timestamps")
    if any(right < left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("unordered_timestamps")
    duration = timestamps[-1] - timestamps[0]
    if duration <= 0:
        raise ValueError("nonpositive_duration")
    return dict(user_id=str(identifiers[0]), workout_id=str(identifiers[1]), sport="run",
                start_timestamp=timestamps[0], duration_seconds=duration)


def prepare(source, output):
    histories = defaultdict(list)
    counts = Counter()
    seen = set()
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            counts["records_read"] += 1
            try:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    record = ast.literal_eval(line)
                row = normalize(record)
            except (ValueError, SyntaxError, TypeError, RecursionError) as error:
                reason = str(error) if str(error) in {"invalid_record", "invalid_identifier", "missing_timestamps", "invalid_timestamps", "unordered_timestamps", "nonpositive_duration"} else "malformed_record"
                counts[reason] += 1
                continue
            if row is None:
                counts["non_running"] += 1
                continue
            if row["workout_id"] in seen:
                counts["duplicate_workout"] += 1
                continue
            seen.add(row["workout_id"])
            histories[row["user_id"]].append(row)

    output.mkdir(parents=True, exist_ok=True)
    split_counts = Counter()
    user_counts = []
    all_rows = []
    for user_id in sorted(histories):
        rows = sorted(histories[user_id], key=lambda row: (row["start_timestamp"], row["workout_id"]))
        # Hold out complete timestamp groups so tied starts never cross splits.
        times = sorted({row["start_timestamp"] for row in rows})
        eligible = len(times) >= 3
        for row in rows:
            split = "train"
            if eligible and row["start_timestamp"] == times[-1]:
                split = "test"
            elif eligible and row["start_timestamp"] == times[-2]:
                split = "validation"
            row["split"] = split
            split_counts[split] += 1
        user_counts.append(dict(user_id=user_id, sessions=len(rows), evaluation_eligible=eligible))
        all_rows.extend(rows)

    for name in ("sessions", "train", "validation", "test"):
        with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(row for row in all_rows if name == "sessions" or row["split"] == name)
    with (output / "users.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["user_id", "sessions", "evaluation_eligible"])
        writer.writeheader()
        writer.writerows(user_counts)
    sizes = [row["sessions"] for row in user_counts]
    stats = dict(source=str(source), records=dict(counts), retained_users=len(sizes),
                 retained_sessions=len(all_rows), split_sessions=dict(split_counts),
                 evaluation_users=sum(row["evaluation_eligible"] for row in user_counts),
                 sessions_per_user=dict(min=min(sizes, default=0), max=max(sizes, default=0),
                                        mean=sum(sizes) / len(sizes) if sizes else 0),
                 duration_definition="last recorded timestamp minus first recorded timestamp, in seconds",
                 split_policy="last timestamp group test, penultimate validation, earlier train; users with fewer than three distinct starts are train-only")
    (output / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    if not all_rows:
        raise ValueError("No valid running sessions retained; inspect stats.json")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    prepare(args.input, args.output)


if __name__ == "__main__":
    main()
