"""Print the results of every training seed, read straight from the JSON reports.

Run from the repository root:  python print_seed_results.py
Seed 42 is read from results/marathon/, the other seeds from results/marathon/seeds/seed_*/.
"""

import json
from pathlib import Path

RESULTS = Path("results/marathon")


def load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def find_seeds():
    seeds = {42: RESULTS}
    for folder in sorted((RESULTS / "seeds").glob("seed_*")):
        seeds[int(folder.name.split("_")[1])] = folder
    return seeds


def main():
    seeds = find_seeds()
    reports = {}
    for seed, folder in seeds.items():
        full, ablation = load(folder / "full.json"), load(folder / "no_race_context.json")
        if full is None or ablation is None:
            print(f"Seed {seed}: missing report in {folder} (full: {full is not None}, ablation: {ablation is not None})")
            continue
        reports[seed] = (full, ablation)

    for seed, (full, ablation) in reports.items():
        ours, abl = full["test"]["full"], ablation["test"]["no_race_context"]
        print(f"\n===== Seed {seed} =====")
        print(f"Best epoch: full {full['best_epoch']}, ablation {ablation['best_epoch']}")
        print("Validation NDCG@10 per epoch (full):    "
              + "  ".join(f"{row['validation_ndcg@10']:.4f}" for row in full["curve"]))
        print("Validation NDCG@10 per epoch (ablation): "
              + "  ".join(f"{row['validation_ndcg@10']:.4f}" for row in ablation["curve"]))

        print(f"\n{'Test, all targets':<40}{'NDCG@10':>10}{'Recall@20':>11}")
        rows = [("Ours (full)", ours), ("Ablation (no race context)", abl)]
        rows += [(name, metrics) for name, metrics in full["test"].items() if name != "full"]
        for name, metrics in rows:
            print(f"{name:<40}{metrics['ndcg@10']:>10.4f}{metrics['recall@20']:>11.4f}")

        print(f"\n{'NDCG@10 by phase':<40}{'Ours':>9}{'Ablation':>10}{'Gain':>9}")
        for phase, metrics in full["phases"]["full"].items():
            other = ablation["phases"]["no_race_context"][phase]["ndcg@10"]
            print(f"{phase:<40}{metrics['ndcg@10']:>9.4f}{other:>10.4f}{metrics['ndcg@10'] - other:>+9.4f}")

    if len(reports) > 1:
        print("\n===== Key differences across seeds =====")
        taper = "taper (race in 0-3 weeks)"
        checks = {
            "Ours - personal popularity, NDCG@10":
                lambda f, a: f["test"]["full"]["ndcg@10"] - f["test"]["personal_popular"]["ndcg@10"],
            "Ours - ablation, NDCG@10 (all)":
                lambda f, a: f["test"]["full"]["ndcg@10"] - a["test"]["no_race_context"]["ndcg@10"],
            "Ours - ablation, Recall@20 (all)":
                lambda f, a: f["test"]["full"]["recall@20"] - a["test"]["no_race_context"]["recall@20"],
            "Taper gain, NDCG@10":
                lambda f, a: f["phases"]["full"][taper]["ndcg@10"] - a["phases"]["no_race_context"][taper]["ndcg@10"],
        }
        print(f"{'':<40}" + "".join(f"{'seed ' + str(seed):>11}" for seed in reports) + f"{'positive?':>12}")
        for label, compute in checks.items():
            values = [compute(*reports[seed]) for seed in reports]
            positive = sum(value > 0 for value in values)
            print(f"{label:<40}" + "".join(f"{value:>+11.4f}" for value in values)
                  + f"{f'{positive}/{len(values)}':>12}")


if __name__ == "__main__":
    main()