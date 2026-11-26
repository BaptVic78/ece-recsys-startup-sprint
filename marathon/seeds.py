"""Repeat training over several seeds and report mean ± std, so results do not rest on one run.

The paired bootstrap in MARATHON_RESULTS.md measures variation across runners for seed 42 only.
This module measures training randomness: the full model and the no-race-context ablation are
retrained with other seeds, with the hyperparameters of the served seed-42 model, and evaluated
on the same temporal test split by the same code (`marathon.train`). Checkpoints stay selected on
validation NDCG@10; the test split is only read for the final report of each run.

    python -m marathon.seeds train      # 2 extra seeds x 2 models, about 1 hour each
    python -m marathon.seeds report     # results/marathon/SEEDS.md and seeds.json

Seed 42 is read from results/marathon/{full,no_race_context}.json. Extra checkpoints go to
artifacts/ (not committed), so models/marathon/full.pt stays the served model.
"""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

MODELS = ("full", "no_race_context")
SERVED_SEED = 42
EXTRA_SEEDS = (7, 123)
TAPER = "taper (race in 0-3 weeks)"
NO_RACE = "no race context"


def served_settings(models_dir):
    """Training hyperparameters of the served model, so every seed is trained the same way."""
    training = json.loads((models_dir / "full.json").read_text(encoding="utf-8"))["training"]
    return {key: training[key] for key in ("epochs", "batch_size", "learning_rate", "train_sample",
                                           "validation_sample")}


def report_path(results, seed, name):
    if seed == SERVED_SEED:
        return results / f"{name}.json"
    return results / "seeds" / f"seed_{seed}" / f"{name}.json"


def train_command(seed, name, settings, data, checkpoints, results, threads):
    if seed == SERVED_SEED:
        raise ValueError("Seed 42 is the served model; retraining it here would overwrite its report")
    command = [sys.executable, "-m", "marathon.train", "--seed", str(seed), "--data", str(data),
               "--output", str(checkpoints / f"seed_{seed}"),
               "--report", str(report_path(results, seed, name).parent), "--threads", str(threads)]
    for key, value in settings.items():
        command += [f"--{key.replace('_', '-')}", str(value)]
    if name == "no_race_context":
        command.append("--no-race-context")
    return command


def train(args):
    settings = served_settings(args.models)
    print(f"Settings from {args.models / 'full.json'}: {settings}", flush=True)
    for seed in args.seeds:
        for name in MODELS:
            if report_path(args.results, seed, name).exists() and not args.force:
                print(f"seed {seed} {name}: report exists, skipped", flush=True)
                continue
            command = train_command(seed, name, settings, args.data, args.checkpoints, args.results, args.threads)
            print(f"seed {seed} {name}: {' '.join(command)}", flush=True)
            subprocess.run(command, check=True)


def seed_metrics(full, ablation):
    """The numbers compared across seeds, from one seed's two `marathon.train` reports."""
    ours, abl = full["test"]["full"], ablation["test"]["no_race_context"]
    personal = full["test"]["personal_popular"]
    phases_ours, phases_abl = full["phases"]["full"], ablation["phases"]["no_race_context"]
    row = {"best_epoch_full": full["best_epoch"], "best_epoch_ablation": ablation["best_epoch"]}
    for metric in ("ndcg@10", "recall@20"):
        row[f"ours {metric}"] = ours[metric]
        row[f"ablation {metric}"] = abl[metric]
        row[f"ours - personal popularity, {metric}"] = ours[metric] - personal[metric]
        row[f"ours - ablation, {metric}"] = ours[metric] - abl[metric]
    for phase in phases_ours:
        row[f"ours ndcg@10, {phase}"] = phases_ours[phase]["ndcg@10"]
        row[f"ablation ndcg@10, {phase}"] = phases_abl[phase]["ndcg@10"]
        row[f"race-context gain ndcg@10, {phase}"] = phases_ours[phase]["ndcg@10"] - phases_abl[phase]["ndcg@10"]
    row["taper gain recall@20"] = phases_ours[TAPER]["recall@20"] - phases_abl[TAPER]["recall@20"]
    return row


def summarise(rows):
    """Mean and sample standard deviation (n - 1) of every metric across seeds."""
    summary = {}
    for key in rows[next(iter(rows))]:
        values = [row[key] for row in rows.values()]
        summary[key] = {"mean": statistics.fmean(values),
                        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                        "min": min(values), "max": max(values)}
    return summary


# Claims made in the README, each checked on every seed: (label, metric key, must be > 0).
CLAIMS = [
    ("Ours beats personal popularity, NDCG@10", "ours - personal popularity, ndcg@10"),
    ("Ours beats personal popularity, Recall@20", "ours - personal popularity, recall@20"),
    ("Race context helps in the taper, NDCG@10", f"race-context gain ndcg@10, {TAPER}"),
    ("Race context helps in the taper, Recall@20", "taper gain recall@20"),
    ("Race context helps overall, Recall@20", "ours - ablation, recall@20"),
]


def markdown(rows, summary):
    seeds = list(rows)
    def cells(key, sign=False):
        fmt = "{:+.4f}" if sign else "{:.4f}"
        return [fmt.format(rows[seed][key]) for seed in seeds] + [
            f"{fmt.format(summary[key]['mean'])} ± {summary[key]['std']:.4f}"]
    def table(header, body):
        lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |"]
        return lines + ["| " + " | ".join(row) + " |" for row in body]
    header = ["Metric"] + [f"Seed {seed}" for seed in seeds] + ["Mean ± std"]
    phases = [key.split(", ", 1)[1] for key in summary if key.startswith("race-context gain ndcg@10, ")]
    lines = ["# Marathon recommender: results over training seeds", "",
             f"The full model and the no-race-context ablation were each trained with seeds {', '.join(map(str, seeds))}, "
             "with the same hyperparameters as the served seed-42 model, and evaluated on the same temporal test split "
             "(September to December 2019) by `marathon.train`. Checkpoints were selected on validation NDCG@10 (August). "
             "Std is the sample standard deviation over seeds (n - 1). With only "
             f"{len(seeds)} seeds it is a rough estimate of training variance; the per-seed columns show the spread directly. "
             "The seed-42 paired bootstrap in [MARATHON_RESULTS.md](MARATHON_RESULTS.md) measures variation across runners, "
             "which is a different source of uncertainty.", "",
             "## Do the claims hold for every seed?", ""]
    body = []
    for label, key in CLAIMS:
        held = sum(rows[seed][key] > 0 for seed in seeds)
        verdict = "yes" if held == len(seeds) else f"**no: {held} of {len(seeds)} seeds**"
        body.append([label] + cells(key, sign=True)[:-1] + [f"{summary[key]['mean']:+.4f} ± {summary[key]['std']:.4f}", verdict])
    lines += table(["Claim (difference)"] + [f"Seed {seed}" for seed in seeds] + ["Mean ± std", "Holds for every seed"], body)
    lines += ["", "## Overall test metrics", ""]
    lines += table(header, [[label] + cells(key) for label, key in (
        ("Ours, NDCG@10", "ours ndcg@10"), ("Ablation, NDCG@10", "ablation ndcg@10"),
        ("Ours, Recall@20", "ours recall@20"), ("Ablation, Recall@20", "ablation recall@20"))]
        + [[label] + cells(key, sign=True) for label, key in (
            ("Ours - ablation, NDCG@10", "ours - ablation, ndcg@10"),
            ("Ours - ablation, Recall@20", "ours - ablation, recall@20"))])
    lines += ["", "## NDCG@10 by preparation phase", ""]
    body = []
    for phase in phases:
        body.append([f"Ours, {phase}"] + cells(f"ours ndcg@10, {phase}"))
        body.append([f"Ablation, {phase}"] + cells(f"ablation ndcg@10, {phase}"))
        body.append([f"Race-context gain, {phase}"] + cells(f"race-context gain ndcg@10, {phase}", sign=True))
    lines += table(header, body)
    lines += ["", "Best epoch (selected on validation): full " +
              ", ".join(f"seed {seed}: {rows[seed]['best_epoch_full']}" for seed in seeds) + "; ablation " +
              ", ".join(f"seed {seed}: {rows[seed]['best_epoch_ablation']}" for seed in seeds) + ".", "",
              "The served model is the seed-42 checkpoint, `models/marathon/full.pt`. Reproduce with "
              "`python -m marathon.seeds train` then `python -m marathon.seeds report`.", ""]
    return "\n".join(lines)


def report(args):
    seeds = [SERVED_SEED] + [seed for seed in args.seeds if seed != SERVED_SEED]
    missing = [str(report_path(args.results, seed, name)) for seed in seeds for name in MODELS
               if not report_path(args.results, seed, name).exists()]
    if missing:
        raise SystemExit("Missing reports, run `python -m marathon.seeds train` first:\n  " + "\n  ".join(missing))
    rows = {}
    for seed in seeds:
        full, ablation = (json.loads(report_path(args.results, seed, name).read_text(encoding="utf-8")) for name in MODELS)
        rows[seed] = seed_metrics(full, ablation)
    summary = summarise(rows)
    (args.results / "seeds.json").write_text(json.dumps({"per_seed": rows, "summary": summary}, indent=2) + "\n",
                                             encoding="utf-8")
    text = markdown(rows, summary)
    (args.results / "SEEDS.md").write_text(text, encoding="utf-8")
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("train", "report"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(EXTRA_SEEDS))
    parser.add_argument("--data", type=Path, default=Path("data/processed/marathon/sessions.parquet"))
    parser.add_argument("--models", type=Path, default=Path("models/marathon"))
    parser.add_argument("--results", type=Path, default=Path("results/marathon"))
    parser.add_argument("--checkpoints", type=Path, default=Path("artifacts/marathon/seeds"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="retrain seeds whose report already exists")
    args = parser.parse_args()
    train(args) if args.command == "train" else report(args)


if __name__ == "__main__":
    main()