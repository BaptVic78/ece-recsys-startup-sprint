"""First real-data next-template experiment with training-only preprocessing."""

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from strideplan.bundle import write_bundle
from strideplan.evaluation import evaluate, most_popular
from strideplan.features import (FEATURE_NAMES, add_raw_features, fit_catalogue, fit_scaler, parse_session,
                                 scale_features, template_id)
from strideplan.model import WorkoutRanker
from strideplan_pipeline.validate import validate


def prepare_rows(path, max_speed=10.0):
    if not math.isfinite(max_speed) or max_speed <= 0:
        raise ValueError("max_speed must be finite and positive")
    histories = defaultdict(list)
    rejected = Counter()
    with path.open(newline="", encoding="utf-8") as stream:
        for record in csv.DictReader(stream):
            row, reason = parse_session(record, max_speed)
            if row is None:
                rejected[reason] += 1
                continue
            histories[row["user"]].append(row)
    for rows in histories.values():
        add_raw_features(rows)
    training_rows = [row for rows in histories.values() for row in rows if row["split"] == "train"]
    catalogue = fit_catalogue(training_rows)
    scaler = fit_scaler(training_rows)
    for rows in histories.values():
        for row in rows:
            row["template"] = template_id(row, catalogue)
            row["features"] = scale_features(row["raw_features"], scaler)
    return histories, catalogue, scaler, dict(rejected)


def examples(histories, split, max_history=10, minimum_training_dates=5):
    """Every context ends strictly before its target; test context may use validation."""
    if max_history < 1:
        raise ValueError("max_history must be positive")
    result = []
    for user in sorted(histories):
        rows = histories[user]
        if split != "train" and len({row["time"] for row in rows if row["split"] == "train"}) < minimum_training_dates:
            continue
        for target in rows:
            if target["split"] != split:
                continue
            allowed_splits = {"train", "validation"} if split == "test" else {"train"}
            context = [row for row in rows if row["time"] < target["time"] and row["split"] in allowed_splits]
            if not context:
                continue
            result.append({"user": user, "target": target["template"], "target_time": target["time"],
                           "context": context[-max_history:]})
    return result


def tensors(cases, max_history):
    if not cases:
        raise ValueError("No examples available for this split")
    items = torch.zeros(len(cases), max_history, dtype=torch.long)
    features = torch.zeros(len(cases), max_history, len(FEATURE_NAMES))
    lengths = torch.zeros(len(cases), dtype=torch.long)
    targets = torch.tensor([case["target"] - 1 for case in cases])
    for index, case in enumerate(cases):
        context = case["context"]
        lengths[index] = len(context)
        items[index, :len(context)] = torch.tensor([row["template"] for row in context])
        features[index, :len(context)] = torch.from_numpy(np.stack([row["features"] for row in context]))
    return TensorDataset(items, features, lengths, targets)


def model_predictions(model, dataset, batch_size, empty_history=False):
    model.eval()
    logits = []
    losses = []
    with torch.no_grad():
        for items, features, lengths, targets in DataLoader(dataset, batch_size=batch_size):
            if empty_history:
                items, features, lengths = torch.zeros_like(items), torch.zeros_like(features), torch.zeros_like(lengths)
            ranking_logits, _ = model(items, features, lengths)
            losses.append((F.cross_entropy(ranking_logits, targets).item(), len(targets)))
            # Stable sort makes tied scores resolve by ascending template ID.
            logits.extend((torch.argsort(ranking_logits, descending=True, stable=True) + 1).tolist())
    return logits, sum(loss * size for loss, size in losses) / len(dataset)


def score_rankings(rankings, cases):
    # Group by athlete if the data contains multiple held-out sessions at a tied start.
    per_user = defaultdict(list)
    for index, case in enumerate(cases):
        per_user[case["user"]].append(index)
    scores = [evaluate({i: rankings[i] for i in indexes}, {i: {cases[i]["target"]} for i in indexes}, (1, 3, 10, 20))
              for indexes in per_user.values()]
    result = {key: sum(score[key] for score in scores) / len(scores) for key in scores[0] if key != "evaluated_users"}
    result.update(evaluated_users=len(per_user), evaluated_sessions=len(cases))
    return result


def baseline_predictions(cases, training_sequences, catalogue, seed):
    candidates = list(range(1, catalogue["num_templates"] + 1))
    popular = most_popular(training_sequences, candidates)
    global_position = {item: index for index, item in enumerate(popular)}
    predictions = {"random": [], "global_popularity": [], "personal_popularity": [], "repeat_last": []}
    for case in cases:
        shuffled = candidates.copy()
        random.Random(f"{seed}:{case['user']}:{case['target_time']}").shuffle(shuffled)
        counts = Counter(row["template"] for row in case["context"])
        personal = sorted(candidates, key=lambda item: (-counts[item], global_position[item]))
        last = case["context"][-1]["template"]
        predictions["random"].append(shuffled)
        predictions["global_popularity"].append(popular)
        predictions["personal_popularity"].append(personal)
        predictions["repeat_last"].append([last] + [item for item in popular if item != last])
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/first_results"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--history", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-speed", type=float, default=10.0)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.history) < 1:
        parser.error("epochs, batch-size and history must be positive")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    validate(args.data)
    histories, catalogue, scaler, rejected = prepare_rows(args.data / "sessions.csv", args.max_speed)
    cases = {split: examples(histories, split, args.history) for split in ("train", "validation", "test")}
    datasets = {split: tensors(values, args.history) for split, values in cases.items()}
    training_sequences = [[row["template"] for row in rows if row["split"] == "train"] for rows in histories.values()]
    args.output.mkdir(parents=True, exist_ok=True)
    model = WorkoutRanker(catalogue["num_templates"], feature_size=len(FEATURE_NAMES))
    # No effort targets have been defined yet. Keep this head explicitly untrained.
    for parameter in model.effort_head.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam([parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.001)
    loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    best_score = -1.0
    curve = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for items, features, lengths, targets in loader:
            optimizer.zero_grad()
            ranking_logits, _ = model(items, features, lengths)
            loss = F.cross_entropy(ranking_logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            total_loss += loss.item() * len(targets)
        rankings, val_loss = model_predictions(model, datasets["validation"], args.batch_size)
        validation_metrics = score_rankings(rankings, cases["validation"])
        record = {"epoch": epoch, "train_loss": total_loss / len(datasets["train"]),
                  "validation_loss": val_loss, "validation_ndcg@3": validation_metrics["ndcg@3"]}
        curve.append(record)
        print(json.dumps(record), flush=True)
        if validation_metrics["ndcg@3"] > best_score:
            best_score = validation_metrics["ndcg@3"]
            torch.save(model.state_dict(), args.output / "ranker.pt")
            best_epoch = epoch
    model.load_state_dict(torch.load(args.output / "ranker.pt", weights_only=True))
    popularity = most_popular(training_sequences, list(range(1, catalogue["num_templates"] + 1)))
    version = write_bundle(args.output, model, catalogue, scaler, popularity, args.history, args.max_speed)
    metrics = {}
    qualitative = []
    for split in ("validation", "test"):
        predictions = baseline_predictions(cases[split], training_sequences, catalogue, args.seed)
        predictions["gru"], _ = model_predictions(model, datasets[split], args.batch_size)
        predictions["gru_no_history"], _ = model_predictions(model, datasets[split], args.batch_size, empty_history=True)
        metrics[split] = {name: score_rankings(values, cases[split]) for name, values in predictions.items()}
        if split == "test":
            for index, case in enumerate(cases[split][:3]):
                qualitative.append({"example": index + 1, "history_templates": [row["template"] for row in case["context"]],
                                    "observed_template": case["target"],
                                    "top3": {name: rankings[index][:3] for name, rankings in predictions.items()}})
    occupied = Counter(item for sequence in training_sequences for item in sequence)
    result = {"protocol": "unconstrained next-template prediction; one seed; no effort model or LLM evaluation",
              "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "catalogue": catalogue, "preprocessing": scaler, "quality_exclusions": rejected,
              "training_template_counts": dict(sorted(occupied.items())),
              "examples": {split: len(values) for split, values in cases.items()},
              "best_epoch_by_validation_ndcg3": best_epoch, "training_curve": curve,
              "metrics": metrics, "qualitative_examples": qualitative,
              "effort_head_trained": False, "model_version": version,
              "versions": {"torch": torch.__version__, "numpy": np.__version__}}
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "test": metrics["test"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
