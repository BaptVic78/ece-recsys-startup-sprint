"""Train the context-aware recommender, evaluate it on the temporal test split and save the model.

Training targets are sessions before 1 August 2019, with histories that end before them.
The checkpoint is chosen on validation NDCG@10 (August). Test (from September) is used once,
for the final report. `--no-race-context` trains the ablation with everything else unchanged.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from marathon.data import NUM_ITEMS
from marathon.evaluate import BASELINES, Baselines, exposure_metrics, macro_average, phase_metrics, target_ranks
from marathon.model import ContextCrossAttentionRecommender
from marathon.sequences import MAX_HISTORY, item_attributes, load_store

CONFIG = dict(dim=64, heads=2, layers=2, dropout=0.1)


def to_torch(batch):
    return {key: torch.as_tensor(value) for key, value in batch.items()}


def model_scores(model, store, targets, batch_size=4096):
    model.eval()
    scores, losses = [], []
    with torch.no_grad():
        for start in range(0, len(targets), batch_size):
            batch = to_torch(store.batch(targets[start:start + batch_size]))
            logits = model(batch)
            losses.append(F.cross_entropy(logits, batch["labels"], reduction="sum").item())
            scores.append(logits.numpy())
    return np.concatenate(scores), sum(losses) / len(targets)


def evaluate_split(store, targets, scores):
    context = store.target_context(targets)
    ranks = target_ranks(scores, context["labels"])
    users = store.sessions["athlete"].to_numpy()[targets]
    return ranks, users, context


def baseline_ranks(store, baselines, name, targets, chunk=50_000):
    """Ranks of a baseline over all targets, computed chunk by chunk to bound memory."""
    ranks = []
    for start in range(0, len(targets), chunk):
        part = targets[start:start + chunk]
        batch = store.batch(part)
        ranks.append(target_ranks(baselines.scores(name, batch, part), batch["labels"]))
    return np.concatenate(ranks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/marathon/sessions.parquet"))
    parser.add_argument("--output", type=Path, default=Path("models/marathon"))
    parser.add_argument("--report", type=Path, default=Path("results/marathon"))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-sample", type=int, default=400_000, help="training targets drawn per epoch")
    parser.add_argument("--validation-sample", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--no-race-context", action="store_true", help="train the ablation")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    name = "no_race_context" if args.no_race_context else "full"
    store = load_store(args.data)
    train, validation, test = (store.targets(split) for split in ("train", "validation", "test"))
    validation_sample = np.sort(rng.choice(validation, min(args.validation_sample, len(validation)), replace=False))
    model = ContextCrossAttentionRecommender(NUM_ITEMS, item_attributes(), MAX_HISTORY,
                                             use_race_context=not args.no_race_context, **CONFIG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / f"{name}.pt"
    curve, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        order = rng.choice(train, min(args.train_sample, len(train)), replace=False)
        total = 0.0
        for start in range(0, len(order), args.batch_size):
            batch = to_torch(store.batch(order[start:start + args.batch_size]))
            optimizer.zero_grad()
            loss = F.cross_entropy(model(batch), batch["labels"])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(batch["labels"])
        scores, validation_loss = model_scores(model, store, validation_sample)
        ranks, users, _ = evaluate_split(store, validation_sample, scores)
        metrics = macro_average(ranks, users)
        record = {"epoch": epoch, "train_loss": total / len(order), "validation_loss": validation_loss,
                  "validation_ndcg@10": metrics["ndcg@10"], "validation_recall@20": metrics["recall@20"],
                  "seconds": round(time.time() - started, 1)}
        curve.append(record)
        print(json.dumps(record), flush=True)
        if metrics["ndcg@10"] > best:
            best, best_epoch = metrics["ndcg@10"], epoch
            torch.save(model.state_dict(), checkpoint)
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    config = {"model": name, "use_race_context": not args.no_race_context, "max_history": MAX_HISTORY,
              "num_items": NUM_ITEMS, **CONFIG, "best_epoch": best_epoch, "training": vars(args) | {
                  "data": str(args.data), "output": str(args.output), "report": str(args.report)}}
    (args.output / f"{name}.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    train_context = store.target_context(train)
    baselines = Baselines(train_context["labels"], train_context["target_race"], NUM_ITEMS, args.seed)
    scores, test_loss = model_scores(model, store, test)
    ranks, users, context = evaluate_split(store, test, scores)
    # Exposure is measured on a fixed sample of test targets, the same for every method.
    exposure_sample = np.sort(rng.choice(len(test), min(50_000, len(test)), replace=False))
    report = {"model": name, "best_epoch": best_epoch, "curve": curve, "test_loss": test_loss,
              "test": {name: macro_average(ranks, users)},
              "phases": {name: phase_metrics(ranks, users, context["target_race"], context["target_recovery"])},
              "exposure": {name: exposure_metrics(scores[exposure_sample], baselines.popularity)}}
    del scores
    if not args.no_race_context:
        sample_batch = store.batch(test[exposure_sample])
        for baseline in BASELINES:
            ranks_b = baseline_ranks(store, baselines, baseline, test)
            report["test"][baseline] = macro_average(ranks_b, users)
            report["phases"][baseline] = phase_metrics(ranks_b, users, context["target_race"], context["target_recovery"])
            report["exposure"][baseline] = exposure_metrics(
                baselines.scores(baseline, sample_batch, test[exposure_sample]), baselines.popularity)
    args.report.mkdir(parents=True, exist_ok=True)
    (args.report / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["test"], indent=2))


if __name__ == "__main__":
    main()
