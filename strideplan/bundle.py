"""Save and load everything needed to serve one trained ranker."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from strideplan.features import FEATURE_NAMES
from strideplan.model import WorkoutRanker


BUNDLE_FILE = "bundle.json"
WEIGHTS_FILE = "ranker.pt"
BUNDLE_FORMAT = 1


@dataclass(frozen=True)
class Bundle:
    model: WorkoutRanker
    catalogue: dict
    scaler: dict
    popularity: list
    max_history: int
    max_speed: float
    version: str


def model_version(weights_path):
    return "gru-" + hashlib.sha256(Path(weights_path).read_bytes()).hexdigest()[:12]


def write_bundle(directory, model, catalogue, scaler, popularity, max_history, max_speed):
    """Describe the weights already saved in directory/ranker.pt. Inputs must come from training data only."""
    directory = Path(directory)
    weights = directory / WEIGHTS_FILE
    if not weights.is_file():
        raise FileNotFoundError(f"Missing {weights}")
    if sorted(popularity) != list(range(1, catalogue["num_templates"] + 1)):
        raise ValueError("Popularity must rank every catalogue template exactly once")
    version = model_version(weights)
    payload = {"format": BUNDLE_FORMAT, "model_version": version, "weights": WEIGHTS_FILE,
               "architecture": {"num_templates": model.num_templates, "feature_size": model.feature_size,
                                "hidden_size": model.encoder.hidden_size},
               "catalogue": catalogue, "preprocessing": scaler, "global_popularity": list(popularity),
               "max_history": max_history, "max_speed_mps": max_speed, "effort_head_trained": False}
    (directory / BUNDLE_FILE).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return version


def load_bundle(directory):
    directory = Path(directory)
    payload = json.loads((directory / BUNDLE_FILE).read_text(encoding="utf-8"))
    if payload.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"Unsupported bundle format {payload.get('format')!r}")
    if payload["preprocessing"]["features"] != FEATURE_NAMES:
        raise ValueError("Bundle features do not match the serving preprocessing")
    architecture = payload["architecture"]
    if architecture["num_templates"] != payload["catalogue"]["num_templates"]:
        raise ValueError("Model and catalogue sizes differ")
    weights = directory / payload["weights"]
    version = model_version(weights)
    if version != payload["model_version"]:
        raise ValueError("Weights changed after the bundle was written")
    model = WorkoutRanker(**architecture)
    model.load_state_dict(torch.load(weights, weights_only=True))
    model.eval()
    return Bundle(model=model, catalogue=payload["catalogue"], scaler=payload["preprocessing"],
                  popularity=payload["global_popularity"], max_history=payload["max_history"],
                  max_speed=payload["max_speed_mps"], version=version)
