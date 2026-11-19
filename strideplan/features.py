"""Session preprocessing shared by training and serving. Fit only on training rows."""

import bisect
import math

import numpy as np


FEATURE_NAMES = ["log_duration_minutes", "speed_mps", "log_gap_days", "heart_rate_bpm", "heart_rate_missing"]


def parse_session(record, max_speed=10.0):
    """Return (row, None) for a usable run or (None, reason) for a quality exclusion.

    Records use the processed-CSV field names. Values may be strings or numbers.
    """
    try:
        speed = float(record["avg_speed_mps"])
    except (TypeError, ValueError):
        return None, "missing_or_invalid_speed"
    if not math.isfinite(speed) or not 0 < speed <= max_speed:
        return None, "missing_or_invalid_speed"
    raw_heart_rate = record.get("avg_heart_rate_bpm")
    try:
        heart_rate = float(raw_heart_rate) if raw_heart_rate not in (None, "") else float("nan")
    except (TypeError, ValueError):
        heart_rate = float("nan")
    if not math.isfinite(heart_rate) or not 0 < heart_rate <= 300:
        heart_rate = float("nan")
    return {"user": record.get("user_id"), "id": str(record.get("workout_id") or ""),
            "time": float(record["start_timestamp"]), "duration": float(record["duration_seconds"]),
            "speed": speed, "heart_rate": heart_rate, "split": record.get("split")}, None


def add_raw_features(rows):
    """Sort one runner's rows chronologically and attach unscaled features in place."""
    rows.sort(key=lambda row: (row["time"], row["id"]))
    previous_time = None
    for row in rows:
        gap_days = (row["time"] - previous_time) / 86400 if previous_time is not None else 0
        row["raw_features"] = [math.log1p(row["duration"] / 60), row["speed"], math.log1p(gap_days),
                               row["heart_rate"], float(not math.isfinite(row["heart_rate"]))]
        previous_time = row["time"]
    return rows


def fit_catalogue(training_rows, duration_bins=6, speed_bins=4):
    if not training_rows or min(duration_bins, speed_bins) < 1:
        raise ValueError("A nonempty training set and positive bin counts are required")
    durations = [row["duration"] for row in training_rows]
    speeds = [row["speed"] for row in training_rows]
    duration_edges = [float(edge) for edge in np.unique(np.quantile(durations, np.linspace(0, 1, duration_bins + 1)[1:-1]))
                      if min(durations) < edge < max(durations)]
    speed_edges = [float(edge) for edge in np.unique(np.quantile(speeds, np.linspace(0, 1, speed_bins + 1)[1:-1]))
                   if min(speeds) < edge < max(speeds)]
    return {"duration_edges_seconds": duration_edges, "speed_edges_mps": speed_edges,
            "num_templates": (len(duration_edges) + 1) * (len(speed_edges) + 1),
            "duration_range_seconds": [float(min(durations)), float(max(durations))],
            "speed_range_mps": [float(min(speeds)), float(max(speeds))]}


def template_id(row, catalogue):
    duration_band = bisect.bisect_right(catalogue["duration_edges_seconds"], row["duration"])
    speed_band = bisect.bisect_right(catalogue["speed_edges_mps"], row["speed"])
    return 1 + duration_band * (len(catalogue["speed_edges_mps"]) + 1) + speed_band


def fit_scaler(training_rows):
    values = np.array([row["raw_features"] for row in training_rows], dtype=np.float64)
    medians = np.array([np.median(column[np.isfinite(column)]) if np.isfinite(column).any() else 0.0
                        for column in values.T])
    filled = np.where(np.isfinite(values), values, medians)
    means = filled.mean(axis=0)
    scales = filled.std(axis=0)
    scales[scales < 1e-8] = 1.0
    return {"features": FEATURE_NAMES, "medians": medians.tolist(),
            "means": means.tolist(), "scales": scales.tolist()}


def scale_features(raw_features, scaler):
    features = np.array(raw_features, dtype=np.float64)
    filled = np.where(np.isfinite(features), features, scaler["medians"])
    return ((filled - scaler["means"]) / scaler["scales"]).astype(np.float32)


def describe_template(template, catalogue):
    """Human-readable duration (minutes) and speed (km/h) ranges of one category.

    Outer bands are bounded by the training minimum and maximum.
    """
    if not 1 <= template <= catalogue["num_templates"]:
        raise ValueError("Template ID is outside the catalogue")
    speed_count = len(catalogue["speed_edges_mps"]) + 1
    duration_band, speed_band = divmod(template - 1, speed_count)
    durations = [catalogue["duration_range_seconds"][0], *catalogue["duration_edges_seconds"],
                 catalogue["duration_range_seconds"][1]]
    speeds = [catalogue["speed_range_mps"][0], *catalogue["speed_edges_mps"], catalogue["speed_range_mps"][1]]
    low_minutes, high_minutes = durations[duration_band] / 60, durations[duration_band + 1] / 60
    low_kmh, high_kmh = speeds[speed_band] * 3.6, speeds[speed_band + 1] * 3.6
    return {"template_id": template,
            "duration_min_minutes": round(low_minutes, 1), "duration_max_minutes": round(high_minutes, 1),
            "speed_min_kmh": round(low_kmh, 1), "speed_max_kmh": round(high_kmh, 1),
            "title": f"Run {low_minutes:.0f}-{high_minutes:.0f} min at {low_kmh:.1f}-{high_kmh:.1f} km/h"}
