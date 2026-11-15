"""Prepare chronological running histories from the public Endomondo export."""

import argparse
import ast
import csv
import gzip
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


FEATURES = ["distance_meters", "avg_speed_mps", "avg_heart_rate_bpm", "elevation_gain_meters", "gender"]
FIELDS = ["user_id", "workout_id", "sport", "start_timestamp", "duration_seconds", *FEATURES, "split"]


def finite_number(value):
    try:
        return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    except OverflowError:
        return False


def aligned_series(record, field, length, lower, upper):
    values = record.get(field)
    if not isinstance(values, list) or len(values) != length:
        return None
    if not all(finite_number(value) and lower <= value <= upper for value in values):
        return None
    return values


def workout_features(record, duration, length):
    latitude = aligned_series(record, "latitude", length, -90, 90)
    longitude = aligned_series(record, "longitude", length, -180, 180)
    distance = None
    if latitude is not None and longitude is not None:
        distance = 0.0
        points = list(zip(map(math.radians, latitude), map(math.radians, longitude)))
        for (lat1, lon1), (lat2, lon2) in zip(points, points[1:]):
            arc = math.sin((lat2 - lat1) / 2) ** 2
            arc += math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
            distance += 6371000 * 2 * math.asin(math.sqrt(min(1.0, max(0.0, arc))))
    heart_rates = aligned_series(record, "heart_rate", length, 1, 300)
    altitude = aligned_series(record, "altitude", length, -12000, 10000)
    gender = record.get("gender")
    return dict(
        distance_meters=distance,
        avg_speed_mps=distance / duration if distance is not None else None,
        avg_heart_rate_bpm=sum(heart_rates) / length if heart_rates is not None else None,
        elevation_gain_meters=sum(max(0, right - left) for left, right in zip(altitude, altitude[1:])) if altitude is not None else None,
        gender=gender if isinstance(gender, str) and gender.strip() else None,
    )


def percentiles(values):
    ordered = sorted(values)
    result = {}
    for percentile in (0, 1, 5, 50, 95, 99, 100):
        if not ordered:
            result[f"p{percentile}"] = None
            continue
        position = (len(ordered) - 1) * percentile / 100
        lower = math.floor(position)
        upper = math.ceil(position)
        result[f"p{percentile}"] = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return result


def normalize(record, min_duration=300, max_duration=21600):
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
    if any(not finite_number(value) or value <= 0 for value in timestamps):
        raise ValueError("invalid_timestamps")
    if any(right < left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("unordered_timestamps")
    duration = timestamps[-1] - timestamps[0]
    if duration <= 0:
        raise ValueError("nonpositive_duration")
    if not min_duration <= duration <= max_duration:
        raise ValueError("duration_out_of_range")
    return dict(user_id=str(identifiers[0]), workout_id=str(identifiers[1]), sport="run",
                start_timestamp=timestamps[0], duration_seconds=duration,
                **workout_features(record, duration, len(timestamps)))


def prepare(source, output, min_duration=300, max_duration=21600, min_train_sessions=5):
    if not finite_number(min_duration) or not finite_number(max_duration) or not 0 < min_duration <= max_duration:
        raise ValueError("Duration bounds must be finite, positive and ordered")
    if isinstance(min_train_sessions, bool) or not isinstance(min_train_sessions, int) or min_train_sessions < 1:
        raise ValueError("min_train_sessions must be a positive integer")
    histories = defaultdict(list)
    counts = Counter()
    seen = set()
    preferred_parser = json.loads
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            counts["records_read"] += 1
            if counts["records_read"] % 10000 == 0:
                print(f"Processed {counts['records_read']} records", file=sys.stderr, flush=True)
            try:
                try:
                    record = preferred_parser(line)
                except (ValueError, SyntaxError):
                    fallback = ast.literal_eval if preferred_parser is json.loads else json.loads
                    record = fallback(line)
                    preferred_parser = fallback
                row = normalize(record, min_duration, max_duration)
            except (ValueError, SyntaxError, TypeError, RecursionError, OverflowError) as error:
                reason = str(error) if str(error) in {"invalid_record", "invalid_identifier", "missing_timestamps", "invalid_timestamps", "unordered_timestamps", "nonpositive_duration", "duration_out_of_range"} else "malformed_record"
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
        eligible = len(times) >= min_train_sessions + 2
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
                 duration_bounds_seconds=dict(min=min_duration, max=max_duration),
                 duration_percentiles_seconds=percentiles([row["duration_seconds"] for row in all_rows]),
                 missing_features={field: sum(row[field] is None for row in all_rows) for field in FEATURES},
                 min_train_sessions=min_train_sessions,
                 duration_definition="last recorded timestamp minus first recorded timestamp, in seconds",
                 split_policy="last timestamp group test, penultimate validation, earlier train; evaluation requires min_train_sessions distinct earlier starts, otherwise train-only")
    (output / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    if not all_rows:
        raise ValueError("No valid running sessions retained; inspect stats.json")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--min-duration", type=float, default=300, help="Minimum recorded duration in seconds (inclusive)")
    parser.add_argument("--max-duration", type=float, default=21600, help="Maximum recorded duration in seconds (inclusive)")
    parser.add_argument("--min-train-sessions", type=int, default=5, help="Minimum distinct training start times for held-out users")
    args = parser.parse_args()
    prepare(args.input, args.output, args.min_duration, args.max_duration, args.min_train_sessions)


if __name__ == "__main__":
    main()
