"""Compare retraining set-ups inside the feedback simulator.

Each variant runs the full loop on the same runners and seeds as a short-loop-only baseline, and reports
the last-round difference with that baseline plus every retraining's promotion and validation change.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch

from strideplan.bundle import load_bundle, resolve_bundle_dir
from strideplan.inference import load_histories
from strideplan.retrain import RetrainConfig, featurize_histories
from strideplan.simulate import build_runners, run_arm, typical_sessions


VARIANTS = {
    "all, lr 3e-4, once": (dict(learning_rate=3e-4), (4,)),
    "all, lr 3e-4, x3": (dict(learning_rate=3e-4), (2, 4, 6)),
    "all, lr 1e-4, x3": (dict(learning_rate=1e-4), (2, 4, 6)),
    "all, lr 3e-4, L2 anchor 1, x3": (dict(learning_rate=3e-4, anchor_weight=1.0), (2, 4, 6)),
    "frozen GRU (embeddings), lr 3e-4, x3": (dict(learning_rate=3e-4, trainable="embeddings"), (2, 4, 6)),
    "bias only (head), lr 1e-3, x3": (dict(learning_rate=1e-3, trainable="head"), (2, 4, 6)),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("artifacts/first_results"))
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("results/retrain_ablation.json"))
    parser.add_argument("--runners", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    args = parser.parse_args()
    torch.set_num_threads(2)
    bundle = load_bundle(resolve_bundle_dir(args.bundle))
    base = load_histories(args.data / "sessions.csv")
    rows = featurize_histories(base, bundle.catalogue, bundle.scaler, bundle.max_speed)
    typical = typical_sessions(rows)
    rounds = 8
    runs = {name: [] for name in VARIANTS}
    for scenario, shift in (("stable habits", 0.0), ("wants lighter runs", -1.0)):
        for seed in args.seeds:
            runners = build_runners(rows, bundle.catalogue, typical, args.runners, shift, seed)
            short = run_arm("short_loop", bundle, base, runners, rounds, seed, 3, RetrainConfig(), typical)
            for name, (options, retrain_rounds) in VARIANTS.items():
                config = RetrainConfig(epochs=1, replay_examples=10000, **options)
                full = run_arm("full_loop", bundle, base, runners, rounds, seed, 3, config, typical,
                               retrain_rounds=retrain_rounds)
                last, reference = full["rounds"][-1], short["rounds"][-1]
                reports = [report for report in full["retrains"] if "skipped" not in report]
                runs[name].append({
                    "scenario": scenario, "seed": seed,
                    "delta": {metric: last[metric] - reference[metric]
                              for metric in ("list_quality", "mrr", "too_hard_share", "acceptance")},
                    "retrains": [{"promoted": report["promoted"],
                                  "validation_delta": report["candidate"]["validation_ndcg@3"]
                                  - report["current"]["validation_ndcg@3"],
                                  "holdout_delta": report["candidate"]["holdout"]["score"]
                                  - report["current"]["holdout"]["score"]} for report in reports]})
                print(json.dumps({"variant": name, **runs[name][-1]}), flush=True)
    summary = {}
    for name, results in runs.items():
        retrains = [report for result in results for report in result["retrains"]]
        summary[name] = {**{f"mean_delta_{metric}": statistics.mean(r["delta"][metric] for r in results)
                            for metric in results[0]["delta"]},
                         "promoted": sum(r["promoted"] for r in retrains), "retrains": len(retrains),
                         "mean_validation_delta": statistics.mean(r["validation_delta"] for r in retrains)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"runners": args.runners, "seeds": args.seeds, "summary": summary,
                                       "runs": runs}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
