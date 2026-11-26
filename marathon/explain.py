"""Deletion test: do the attention explanations point at the sessions the model actually relies on?

For each sampled test target, take the served model's top-ranked session type and the three
history sessions its cross-attention weights most (the `because_of` sessions). Remove them from
the history and measure how much that type's score drops. Compare with removing three random
history sessions (averaged over several draws) and with removing the three most recent ones,
the explanation a runner would guess without the model. A faithful explanation should cause a
larger drop than random removal. The result is reported whatever it shows.

Removing a session masks it out of both the self-attention and the cross-attention, exactly like
padding. Targets keep at least MIN_HISTORY sessions after removal, as in training.

    python -m marathon.explain           # results/marathon/EXPLAIN.md and explain_deletion.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from marathon.data import NUM_ITEMS
from marathon.model import ContextCrossAttentionRecommender
from marathon.sequences import MIN_HISTORY, item_attributes, load_store

REMOVED = 3
RANDOM_DRAWS = 5


def load_model(directory):
    config = json.loads((directory / "full.json").read_text(encoding="utf-8"))
    model = ContextCrossAttentionRecommender(config["num_items"], item_attributes(), config["max_history"],
                                             dim=config["dim"], heads=config["heads"], layers=config["layers"],
                                             dropout=config["dropout"], use_race_context=config["use_race_context"])
    model.load_state_dict(torch.load(directory / "full.pt", weights_only=True))
    return model.eval(), config


def sample_targets(store, max_history, sample, rng):
    """One test target per runner, with at least MIN_HISTORY sessions left after removing REMOVED."""
    test = store.targets("test")
    history = np.minimum(store.position[test], max_history)
    test = test[history >= MIN_HISTORY + REMOVED]
    athletes = store.sessions["athlete"].to_numpy()[test]
    order = rng.permutation(len(test))
    _, first = np.unique(athletes[order], return_index=True)
    per_runner = test[order[first]]
    return np.sort(rng.choice(per_runner, min(sample, len(per_runner)), replace=False))


def top_slots(keys, mask, count=REMOVED):
    """Boolean [batch, history] selecting the `count` valid slots with the highest key."""
    keys = np.where(mask, keys, -np.inf)
    chosen = np.argsort(-keys, axis=1, kind="stable")[:, :count]
    selected = np.zeros_like(mask)
    np.put_along_axis(selected, chosen, True, axis=1)
    return selected & mask


def effect(model, inputs, removed, items, base_logit, base_probability):
    """Drop of each target's top item when the `removed` slots are masked out of the history."""
    changed = dict(inputs)
    changed["mask"] = inputs["mask"] & ~torch.as_tensor(removed)
    with torch.no_grad():
        logits = model(changed).numpy()
    rows = np.arange(len(items))
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    rank = (logits > logits[rows, items][:, None]).sum(axis=1)
    return {"logit_drop": base_logit - logits[rows, items], "probability_drop": base_probability - probabilities[rows, items],
            "top1_kept": (rank == 0).astype(float), "new_rank": rank + 1.0}


def paired(a, b, rng, samples=1000):
    difference = np.asarray(a) - np.asarray(b)
    means = np.array([difference[rng.integers(0, len(difference), len(difference))].mean() for _ in range(samples)])
    return {"mean": float(difference.mean()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "share_larger": float((difference > 0).mean())}


def run(store, model, max_history, targets, rng, batch_size=512):
    results = {name: {key: [] for key in ("logit_drop", "probability_drop", "top1_kept", "new_rank")}
               for name in ("attention", "random", "recent")}
    attention_share, overlap_recent = [], []
    for start in range(0, len(targets), batch_size):
        batch = store.batch(targets[start:start + batch_size], max_history=max_history)
        inputs = {key: torch.as_tensor(value) for key, value in batch.items()}
        mask = batch["mask"]
        with torch.no_grad():
            logits = model(inputs).numpy()
            weights = model(inputs, return_attention=True)[1].numpy()
        rows = np.arange(len(logits))
        items = logits.argmax(axis=1)
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        base_logit, base_probability = logits[rows, items], probabilities[rows, items]
        item_weights = weights[rows, items]  # [batch, history]: attention of the top item

        attended = top_slots(item_weights, mask)
        recent = top_slots(np.broadcast_to(np.arange(max_history), mask.shape), mask)  # histories are left-padded
        attention_share.append((item_weights * attended).sum(axis=1))
        overlap_recent.append((attended & recent).sum(axis=1) / REMOVED)
        for name, removed in (("attention", attended), ("recent", recent)):
            for key, value in effect(model, inputs, removed, items, base_logit, base_probability).items():
                results[name][key].append(value)
        draws = [effect(model, inputs, top_slots(rng.random(mask.shape), mask), items, base_logit, base_probability)
                 for _ in range(RANDOM_DRAWS)]
        for key in draws[0]:
            results["random"][key].append(np.mean([draw[key] for draw in draws], axis=0))
        print(f"{min(start + batch_size, len(targets)):,} / {len(targets):,} targets", flush=True)
    per_target = {name: {key: np.concatenate(values) for key, values in metrics.items()} for name, metrics in results.items()}
    return per_target, np.concatenate(attention_share), np.concatenate(overlap_recent)


def report(per_target, attention_share, overlap_recent, targets, rng, config):
    summary = {name: {key: float(values.mean()) for key, values in metrics.items()} for name, metrics in per_target.items()}
    comparisons = {f"{key}: attention - {other}": paired(per_target["attention"][key], per_target[other][key], rng)
                   for other in ("random", "recent") for key in ("logit_drop", "probability_drop")}
    result = {"model": f"{config['model']}-epoch{config['best_epoch']}", "targets": int(len(targets)), "removed": REMOVED,
              "random_draws": RANDOM_DRAWS, "mean": summary, "comparisons": comparisons,
              "attention_share_of_top3": float(attention_share.mean()),
              "overlap_with_3_most_recent": float(overlap_recent.mean())}
    versus_random = comparisons["logit_drop: attention - random"]
    faithful = versus_random["ci95"][0] > 0
    versus_recent = comparisons["logit_drop: attention - recent"]
    names = {"attention": "Top-3 attended (`because_of`)", "random": f"3 random sessions (mean of {RANDOM_DRAWS} draws)",
             "recent": "3 most recent sessions"}
    lines = ["# Explanations: deletion test", "",
             f"{len(targets):,} test targets (September to December 2019), one per runner, each with at least "
             f"{MIN_HISTORY + REMOVED} history sessions. For each, the served model's top-ranked session type is "
             f"explained by the {REMOVED} history sessions its cross-attention weights most. We remove {REMOVED} "
             "sessions from the history (masked like padding) and measure how much that type's score drops.", "",
             "| Sessions removed | Logit drop | Probability drop | Still ranked first | Mean new rank |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name in ("attention", "random", "recent"):
        row = summary[name]
        lines.append(f"| {names[name]} | {row['logit_drop']:+.4f} | {row['probability_drop']:+.4f} | "
                     f"{row['top1_kept']:.1%} | {row['new_rank']:.2f} |")
    lines += ["", "Paired bootstrap over targets (1,000 resamples):", "",
              "| Comparison | Mean difference | 95% interval | Targets where attention removal hurts more |",
              "| --- | ---: | ---: | ---: |"]
    lines += [f"| {key} | {value['mean']:+.4f} | [{value['ci95'][0]:+.4f}, {value['ci95'][1]:+.4f}] | "
              f"{value['share_larger']:.1%} |" for key, value in comparisons.items()]
    lines += ["", "## Reading this", "",
              f"- The three `because_of` sessions carry {attention_share.mean():.1%} of the top item's attention on "
              f"average; {overlap_recent.mean():.1%} of them are among the three most recent sessions.",
              ("- **Faithful against random:** removing the attended sessions lowers the recommended type's score more "
               "than removing random sessions, and the 95% interval excludes 0." if faithful else
               "- **Not shown to be faithful against random:** the 95% interval of the difference with random removal "
               "does not exclude 0 or is negative, so these explanations should be presented as what the model looked "
               "at, not as proven causes."),
              ("- Removing the attended sessions also hurts more than removing the three most recent ones, so the "
               "explanation adds information beyond recency." if versus_recent["ci95"][0] > 0 else
               "- Removing the three most recent sessions hurts at least as much as removing the attended ones, so "
               "recency alone explains the recommendation about as well; attention is not more informative than "
               "\"your last runs\"."),
              "- Attention weights show what the model looked at; this test checks whether that matters for the score. "
              "Masking keeps every other session in its slot, but their own features (such as the gap since the "
              "previous run) still reflect the removed sessions, so this is an approximation of a history without them.", "",
              "Reproduce with `python -m marathon.explain`.", ""]
    return result, "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/processed/marathon/sessions.parquet"))
    parser.add_argument("--models", type=Path, default=Path("models/marathon"))
    parser.add_argument("--results", type=Path, default=Path("results/marathon"))
    parser.add_argument("--sample", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    model, config = load_model(args.models)
    if config["num_items"] != NUM_ITEMS:
        raise ValueError("The saved model does not match the session catalogue")
    store = load_store(args.data)
    targets = sample_targets(store, config["max_history"], args.sample, rng)
    per_target, share, overlap = run(store, model, config["max_history"], targets, rng)
    result, text = report(per_target, share, overlap, targets, rng, config)
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "explain_deletion.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.results / "EXPLAIN.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
