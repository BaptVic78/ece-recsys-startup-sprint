"""Long feedback loop: fine-tune the served ranker on logged feedback and promote it only if it does not regress.

- Completed and accepted workouts are positives; rejected ones are pushed down (unlikelihood loss).
- The three effort buttons train the effort head (too easy 0, ok 0.5, too hard 1).
- Endomondo training examples are replayed so the model does not forget the original task.
- The catalogue and feature scaler of the base bundle are kept, so category IDs stay stable.
- The latest feedback (by time) is held out; the candidate is compared with the current model on it
  and on the original validation split. Rejected candidates are saved but not served.
"""

import argparse
import json
import random
import shutil
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from strideplan.bundle import (BUNDLE_FILE, WEIGHTS_FILE, load_bundle, read_registry, resolve_bundle_dir,
                               write_bundle, write_registry)
from strideplan.experiment import examples, model_predictions, score_rankings, tensors
from strideplan.features import add_raw_features, parse_session, scale_features, template_id
from strideplan.inference import load_histories
from strideplan.model import WorkoutRanker
from strideplan.store import EFFORT_TARGET, Store


@dataclass(frozen=True)
class RetrainConfig:
    epochs: int = 2
    learning_rate: float = 1e-4
    batch_size: int = 256
    feedback_batch_size: int = 64
    feedback_weight: float = 1.0
    holdout_fraction: float = 0.2
    min_feedback: int = 20
    min_effort_labels: int = 20
    tolerance: float = 0.005
    replay_examples: int = 0
    trainable: str = "all"
    anchor_weight: float = 0.0
    seed: int = 42


TRAINABLE = {
    # Which ranking parameters move. The effort head follows its own rule (enough effort labels).
    "all": ("templates", "encoder", "ranking_bias"),
    "embeddings": ("templates", "ranking_bias"),
    "head": ("ranking_bias",),
}


def select_trainable(model, trainable, train_effort):
    if trainable not in TRAINABLE:
        raise ValueError(f"trainable must be one of {sorted(TRAINABLE)}")
    for name, parameter in model.named_parameters():
        group = name.split(".")[0]
        parameter.requires_grad_(train_effort if group == "effort_head" else group in TRAINABLE[trainable])


def featurize_histories(records_by_user, catalogue, scaler, max_speed):
    """Apply the base bundle's fitted preprocessing to Endomondo and logged sessions together."""
    histories = {}
    for user, records in records_by_user.items():
        rows = [row for row, _ in (parse_session(record, max_speed) for record in records) if row is not None]
        add_raw_features(rows)
        for row in rows:
            row["template"] = template_id(row, catalogue)
            row["features"] = scale_features(row["raw_features"], scaler)
        histories[user] = rows
    return histories


def merge_records(base_histories, store):
    merged = defaultdict(list, {user: list(records) for user, records in base_histories.items()})
    for session in store.sessions():
        merged[session["user_id"]].append(session)
    return dict(merged)


def feedback_examples(events, histories, max_history):
    """Each context is the user's history strictly before the recommendation was served."""
    result = []
    for event in events:
        rows = histories.get(event["user_id"], [])
        context = [row for row in rows if row["time"] < event["served_at"]][-max_history:]
        result.append({"context": context, "template": event["template_id"], "action": event["action"],
                       "effort": event["effort"], "time": event["created_at"]})
    return result


def batch(cases, max_history, feature_size):
    items = torch.zeros(len(cases), max_history, dtype=torch.long)
    features = torch.zeros(len(cases), max_history, feature_size)
    lengths = torch.zeros(len(cases), dtype=torch.long)
    for index, case in enumerate(cases):
        context = case["context"]
        lengths[index] = len(context)
        if context:
            items[index, :len(context)] = torch.tensor([row["template"] for row in context])
            features[index, :len(context)] = torch.from_numpy(np.stack([row["features"] for row in context]))
    return items, features, lengths


def feedback_loss(model, cases, max_history, train_effort, weight):
    items, features, lengths = batch(cases, max_history, model.feature_size)
    logits, effort_logits = model(items, features, lengths)
    targets = torch.tensor([case["template"] - 1 for case in cases])
    rows = torch.arange(len(cases))
    log_probabilities = F.log_softmax(logits, dim=1)[rows, targets]
    positive_weight = torch.tensor([{"completed": 1.0, "accepted": 0.5}.get(case["action"], 0.0) for case in cases])
    negative = torch.tensor([case["action"] == "rejected" for case in cases], dtype=torch.float32)
    # Unlikelihood: lower the probability of a rejected category without forcing any other one.
    unlikely = -torch.log1p(-log_probabilities.exp().clamp(max=1 - 1e-6))
    loss = weight * (-(positive_weight * log_probabilities) + negative * unlikely).mean()
    labelled = [index for index, case in enumerate(cases) if case["effort"] is not None]
    if train_effort and labelled:
        index = torch.tensor(labelled)
        effort_targets = torch.tensor([EFFORT_TARGET[cases[i]["effort"]] for i in labelled])
        loss = loss + F.binary_cross_entropy_with_logits(effort_logits[index, targets[index]], effort_targets)
    return loss


def feedback_metrics(model, cases, max_history, effort_trained):
    """Positive MRR, share of rejected categories still in the top 3, and effort error (0.5 if not learned)."""
    if not cases:
        return {"events": 0, "positive_mrr": 0.0, "rejected_in_top3": 0.0, "effort_mae": None, "score": 0.0}
    model.eval()
    with torch.no_grad():
        logits, effort_logits = model(*batch(cases, max_history, model.feature_size))
    order = torch.argsort(logits, dim=1, descending=True, stable=True)
    ranks = (order == torch.tensor([case["template"] - 1 for case in cases])[:, None]).nonzero()[:, 1] + 1
    positives = [1 / float(rank) for case, rank in zip(cases, ranks) if case["action"] != "rejected"]
    rejected = [float(rank <= 3) for case, rank in zip(cases, ranks) if case["action"] == "rejected"]
    errors = []
    for index, case in enumerate(cases):
        if case["effort"] is not None:
            predicted = torch.sigmoid(effort_logits[index, case["template"] - 1]).item() if effort_trained else 0.5
            errors.append(abs(predicted - EFFORT_TARGET[case["effort"]]))
    mrr = sum(positives) / len(positives) if positives else 0.0
    top3 = sum(rejected) / len(rejected) if rejected else 0.0
    return {"events": len(cases), "positive_mrr": mrr, "rejected_in_top3": top3,
            "effort_mae": sum(errors) / len(errors) if errors else None, "score": mrr - top3}


def validation_ndcg(model, cases, max_history, batch_size):
    if not cases:
        return None
    rankings, _ = model_predictions(model, tensors(cases, max_history), batch_size)
    return score_rankings(rankings, cases)["ndcg@3"]


def retrain(bundle, base_histories, store, config=RetrainConfig()):
    """Return (candidate model, report). The report says whether to promote and whether effort is served."""
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    events = store.feedback()
    if len(events) < config.min_feedback:
        raise ValueError(f"Need at least {config.min_feedback} feedback events, found {len(events)}")
    split = int(len(events) * (1 - config.holdout_fraction))
    cutoff = events[split]["created_at"]
    # Keep events sharing the cutoff time together on the holdout side.
    train_events = [event for event in events if event["created_at"] < cutoff]
    holdout_events = [event for event in events if event["created_at"] >= cutoff]
    if not train_events or not holdout_events:
        raise ValueError("Feedback spans too little time to hold out the latest events")
    histories = featurize_histories(merge_records(base_histories, store), bundle.catalogue, bundle.scaler,
                                    bundle.max_speed)
    max_history = bundle.max_history
    train_cases = feedback_examples(train_events, histories, max_history)
    holdout_cases = feedback_examples(holdout_events, histories, max_history)
    replay = examples(histories, "train", max_history)
    if config.replay_examples and len(replay) > config.replay_examples:
        replay = random.Random(config.seed).sample(replay, config.replay_examples)
    validation = examples(histories, "validation", max_history)
    effort_labels = sum(case["effort"] is not None for case in train_cases)
    train_effort = effort_labels >= config.min_effort_labels

    candidate = WorkoutRanker(bundle.model.num_templates, bundle.model.feature_size, bundle.model.encoder.hidden_size)
    candidate.load_state_dict(bundle.model.state_dict())
    select_trainable(candidate, config.trainable, train_effort or bundle.effort_head_trained)
    trained = [(name, parameter) for name, parameter in candidate.named_parameters() if parameter.requires_grad]
    # Anchor: an L2 pull towards the served weights, so fine-tuning cannot drift far from the validated model.
    anchors = {name: parameter.detach().clone() for name, parameter in trained}
    optimizer = torch.optim.Adam([parameter for _, parameter in trained], lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(tensors(replay, max_history), batch_size=config.batch_size, shuffle=True, generator=generator)
    sampler = random.Random(config.seed)
    curve = []
    for epoch in range(1, config.epochs + 1):
        candidate.train()
        total = 0.0
        for items, features, lengths, targets in loader:
            optimizer.zero_grad()
            logits, _ = candidate(items, features, lengths)
            loss = F.cross_entropy(logits, targets)
            loss = loss + feedback_loss(candidate, sampler.sample(train_cases, min(len(train_cases),
                                                                                  config.feedback_batch_size)),
                                        max_history, train_effort, config.feedback_weight)
            if config.anchor_weight:
                loss = loss + config.anchor_weight * sum(((parameter - anchors[name]) ** 2).sum()
                                                         for name, parameter in trained)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite retraining loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(candidate.parameters(), max_norm=5)
            optimizer.step()
            total += loss.item()
        curve.append({"epoch": epoch, "mean_loss": total / max(1, len(loader))})
    candidate.eval()

    effort_learned = train_effort or bundle.effort_head_trained
    current = {"validation_ndcg@3": validation_ndcg(bundle.model, validation, max_history, config.batch_size),
               "holdout": feedback_metrics(bundle.model, holdout_cases, max_history, bundle.effort_head_trained)}
    proposed = {"validation_ndcg@3": validation_ndcg(candidate, validation, max_history, config.batch_size),
                "holdout": feedback_metrics(candidate, holdout_cases, max_history, effort_learned)}
    keeps_task = (current["validation_ndcg@3"] is None
                  or proposed["validation_ndcg@3"] >= current["validation_ndcg@3"] - config.tolerance)
    fits_feedback = proposed["holdout"]["score"] >= current["holdout"]["score"]
    # Serve effort only if it beats the neutral 0.5 guess on held-out ratings.
    neutral = feedback_metrics(candidate, holdout_cases, max_history, False)["effort_mae"]
    serve_effort = bool(effort_learned and proposed["holdout"]["effort_mae"] is not None
                        and proposed["holdout"]["effort_mae"] < neutral)
    report = {"config": asdict(config), "base_version": bundle.version,
              "feedback": {"train_events": len(train_events), "holdout_events": len(holdout_events),
                           "holdout_from": cutoff, "effort_labels": effort_labels},
              "replay_examples": len(replay), "effort_head_trained": train_effort, "serve_effort": serve_effort,
              "curve": curve, "current": current, "candidate": proposed,
              "checks": {"validation_within_tolerance": keeps_task, "holdout_not_worse": fits_feedback},
              "promoted": keeps_task and fits_feedback}
    return candidate, report


def save_candidate(models_dir, candidate, bundle, report):
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=models_dir) as staging:
        torch.save(candidate.state_dict(), Path(staging) / WEIGHTS_FILE)
        version = write_bundle(staging, candidate, bundle.catalogue, bundle.scaler, bundle.popularity,
                               bundle.max_history, bundle.max_speed, effort_head_trained=report["serve_effort"],
                               lineage=(*bundle.lineage, bundle.version), extra={"retrain": report})
        target = models_dir / version
        if not target.exists():
            shutil.copytree(staging, target)
    return version


def register_base(models_dir, base_dir, bundle):
    """Copy the first served bundle into the registry so it can be rolled back to."""
    registry = read_registry(models_dir)
    if registry["current"] is None:
        target = Path(models_dir) / bundle.version
        if not target.exists():
            target.mkdir(parents=True)
            for name in (BUNDLE_FILE, WEIGHTS_FILE):
                shutil.copy2(Path(base_dir) / name, target / name)
        registry = {"current": bundle.version, "history": [{"version": bundle.version, "event": "base"}]}
        write_registry(models_dir, registry)
    return registry


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--base", type=Path, default=Path("artifacts/first_results"),
                        help="Bundle used when the registry does not exist yet")
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--db", type=Path, default=Path("artifacts/feedback/feedback.sqlite3"))
    parser.add_argument("--epochs", type=int, default=RetrainConfig.epochs)
    parser.add_argument("--replay-examples", type=int, default=RetrainConfig.replay_examples,
                        help="Sample of Endomondo training examples to replay; 0 replays all")
    parser.add_argument("--min-feedback", type=int, default=RetrainConfig.min_feedback)
    parser.add_argument("--trainable", choices=sorted(TRAINABLE), default=RetrainConfig.trainable)
    parser.add_argument("--learning-rate", type=float, default=RetrainConfig.learning_rate)
    parser.add_argument("--anchor-weight", type=float, default=RetrainConfig.anchor_weight)
    parser.add_argument("--rollback", action="store_true", help="Serve the previously promoted model again")
    args = parser.parse_args()
    if args.rollback:
        registry = read_registry(args.models)
        promoted = [entry["version"] for entry in registry["history"] if entry["event"] in ("base", "promoted")]
        if len(promoted) < 2 or registry["current"] != promoted[-1]:
            parser.error("Nothing to roll back to")
        registry["current"] = promoted[-2]
        registry["history"].append({"version": promoted[-2], "event": "rollback", "from": promoted[-1]})
        write_registry(args.models, registry)
        print(json.dumps({"current": registry["current"]}))
        return
    torch.set_num_threads(2)
    base_dir = resolve_bundle_dir(args.models) if (args.models / "registry.json").is_file() else args.base
    bundle = load_bundle(base_dir)
    register_base(args.models, base_dir, bundle)
    store = Store(args.db)
    sessions = args.data / "sessions.csv"
    base_histories = load_histories(sessions) if sessions.is_file() else {}
    config = RetrainConfig(epochs=args.epochs, replay_examples=args.replay_examples, min_feedback=args.min_feedback,
                           trainable=args.trainable, learning_rate=args.learning_rate,
                           anchor_weight=args.anchor_weight)
    candidate, report = retrain(bundle, base_histories, store, config)
    version = save_candidate(args.models, candidate, bundle, report)
    registry = read_registry(args.models)
    registry["history"].append({"version": version, "event": "promoted" if report["promoted"] else "rejected",
                                "base": bundle.version})
    if report["promoted"]:
        registry["current"] = version
    write_registry(args.models, registry)
    print(json.dumps({"version": version, "promoted": report["promoted"], "serve_effort": report["serve_effort"],
                      "current": report["current"], "candidate": report["candidate"]}, indent=2))
    print("Restart the API to serve the promoted model." if report["promoted"] else "The current model stays served.")


if __name__ == "__main__":
    main()
