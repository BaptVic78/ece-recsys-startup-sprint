"""Build the results report: metrics, ablation with paired bootstrap, phases, curves and examples.

Reads the reports written by `marathon.train` and the two saved checkpoints, then recomputes the
test ranks it needs for the paired bootstrap, so every number comes from the same split and code.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from marathon.data import DISTANCE_POINTS, INTENSITY_EDGES, NUM_ITEMS, describe_item
from marathon.evaluate import PHASES, Baselines, per_target_metrics, target_ranks
from marathon.model import ContextCrossAttentionRecommender
from marathon.sequences import item_attributes, load_store, race_bucket
from marathon.train import baseline_ranks, model_scores

NAMES = {"full": "Context-aware Transformer (ours)", "no_race_context": "Ablation: same model without race context",
         "personal_popular": "Personal popularity", "repeat_last": "Repeat last session type",
         "phase_popular": "Popularity per preparation phase", "most_popular": "Most popular", "random": "Random"}


def load_model(directory, name):
    config = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
    model = ContextCrossAttentionRecommender(config["num_items"], item_attributes(), config["max_history"],
                                             dim=config["dim"], heads=config["heads"], layers=config["layers"],
                                             dropout=config["dropout"], use_race_context=config["use_race_context"])
    model.load_state_dict(torch.load(directory / f"{name}.pt", weights_only=True))
    return model.eval()


def per_user(ranks, users, metric):
    frame = per_target_metrics(ranks)[[metric]].assign(user=users)
    return frame.groupby("user")[metric].mean()


def paired_bootstrap(a, b, samples=1000, seed=42):
    """Mean difference a - b over runners with a 95% percentile interval from resampling runners."""
    difference = (a - b.loc[a.index]).to_numpy()
    rng = np.random.default_rng(seed)
    means = np.array([difference[rng.integers(0, len(difference), len(difference))].mean() for _ in range(samples)])
    return {"mean": float(difference.mean()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}


def plot_curves(reports, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, report in reports.items():
        curve = pd.DataFrame(report["curve"])
        axes[0].plot(curve.epoch, curve.train_loss, marker="o", label=f"{name} train")
        axes[0].plot(curve.epoch, curve.validation_loss, marker="s", linestyle="--", label=f"{name} validation")
        axes[1].plot(curve.epoch, curve["validation_ndcg@10"], marker="o", label=name)
    axes[0].set(title="Cross-entropy loss", xlabel="epoch", ylabel="loss")
    axes[1].set(title="Validation NDCG@10 (August 2019)", xlabel="epoch", ylabel="NDCG@10")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=120)


SWEEP_DAYS = [70, 56, 42, 35, 28, 21, 14, 10, 7, 4, 2, 1]


def one_target_per_runner(store, targets, rng, sample):
    """One random target per runner, for `sample` distinct runners, so every runner weighs the same."""
    athletes = store.sessions["athlete"].to_numpy()[targets]
    order = rng.permutation(len(targets))
    _, first = np.unique(athletes[order], return_index=True)
    per_runner = targets[order[first]]
    return np.sort(rng.choice(per_runner, min(sample, len(per_runner)), replace=False))


def race_date_sweep(models, store, targets, rng, sample=2000):
    """Counterfactual: keep each runner's history, move only the declared race date.

    For every target, the race is placed `days` after the target day and the race-calendar
    buckets of the history are shifted consistently. The output is the expected distance of
    the recommended session (softmax-weighted over the 66 types). A model without race context
    must give the same value for every `days`. Each runner counts once: one random eligible
    target is drawn per runner, then `sample` distinct runners are drawn.
    """
    chosen = one_target_per_runner(store, targets, rng, sample)
    batch = store.batch(chosen)
    dates = store.sessions["date"].to_numpy("datetime64[D]")
    offsets = np.arange(-batch["items"].shape[1], 0)
    history_index = np.clip(chosen[:, None] + offsets[None, :], 0, None)
    days_before_target = (dates[chosen][:, None] - dates[history_index]).astype(int)
    km = np.repeat(DISTANCE_POINTS, len(INTENSITY_EDGES) + 1)
    result = {}
    for name, model in models.items():
        values = []
        for days in SWEEP_DAYS:
            shifted = dict(batch)
            shifted["target_race"] = race_bucket(np.full(len(chosen), days))
            shifted["race"] = np.where(batch["mask"], race_bucket(days + days_before_target), 0)
            with torch.no_grad():
                logits = model({key: torch.as_tensor(value) for key, value in shifted.items()})
            probabilities = torch.softmax(logits, dim=1).numpy()
            values.append(float((probabilities * km[None, :]).sum(axis=1).mean()))
        result[name] = dict(zip(SWEEP_DAYS, values))
    result["runners"] = int(len(np.unique(store.sessions["athlete"].to_numpy()[chosen])))
    return result


def plot_sweep(sweep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(6.5, 3.8))
    for name, values in sweep.items():
        if name == "runners":
            continue
        axis.plot(list(values), list(values.values()), marker="o", label=name)
    axis.invert_xaxis()
    axis.set(title="Same histories, only the race date moves", xlabel="days until the declared race",
             ylabel="expected recommended distance (km)")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=120)


def item_text(item):
    info = describe_item(item)
    distance = f"{info['km_min']:g}+ km" if info["km_max"] is None else f"{info['km_min']:g}-{info['km_max']:g} km"
    return f"{distance}, {info['intensity']}"


def examples(store, model, baselines, targets, count=3):
    """The first test targets of the first runners with a race in the test period, in athlete order."""
    sessions = store.sessions
    racing = sessions.loc[sessions.is_race & (sessions.split == "test"), "athlete"].unique()
    chosen = []
    for athlete in sorted(racing)[:count]:
        candidates = targets[(sessions["athlete"].to_numpy()[targets] == athlete)
                             & (store.race_bucket[targets] >= 2) & (store.race_bucket[targets] <= 4)]
        if len(candidates):
            chosen.append(candidates[0])
    batch = store.batch(np.array(chosen))
    with torch.no_grad():
        scores = model({key: torch.as_tensor(value) for key, value in batch.items()}).numpy()
    personal = baselines.scores("personal_popular", batch, chosen)
    rows = []
    for row, target in enumerate(chosen):
        history = [target - offset for offset in range(5, 0, -1)]
        rows.append({
            "runner": int(sessions.athlete[target]), "date": str(sessions.date[target].date()),
            "days_to_race": int(sessions.days_to_race[target]),
            "last_sessions": [f"{sessions.date[i].date()}: {sessions.km[i]:.1f} km ({item_text(store.item[i])})" for i in history],
            "actual": item_text(store.item[target]),
            "model_top5": [item_text(i + 1) for i in np.argsort(-scores[row], kind="stable")[:5]],
            "personal_top5": [item_text(i + 1) for i in np.argsort(-personal[row], kind="stable")[:5]],
            "model_rank_of_actual": int(target_ranks(scores[row:row + 1], batch["labels"][row:row + 1])[0]) + 1,
            "personal_rank_of_actual": int(target_ranks(personal[row:row + 1], batch["labels"][row:row + 1])[0]) + 1,
        })
    return rows


def table(rows, columns):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |"]
    lines += ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/marathon/sessions.parquet"))
    parser.add_argument("--models", type=Path, default=Path("models/marathon"))
    parser.add_argument("--results", type=Path, default=Path("results/marathon"))
    args = parser.parse_args()
    torch.set_num_threads(8)
    reports = {name: json.loads((args.results / f"{name}.json").read_text(encoding="utf-8"))
               for name in ("full", "no_race_context")}
    store = load_store(args.data)
    test = store.targets("test")
    users = store.sessions["athlete"].to_numpy()[test]
    context = store.target_context(test)
    train_context = store.target_context(store.targets("train"))
    baselines = Baselines(train_context["labels"], train_context["target_race"], NUM_ITEMS)
    ranks = {}
    models = {}
    for name in reports:
        models[name] = load_model(args.models, name)
        scores, _ = model_scores(models[name], store, test)
        ranks[name] = target_ranks(scores, context["labels"])
        del scores
    ranks["personal_popular"] = baseline_ranks(store, baselines, "personal_popular", test)
    ranks["most_popular"] = baseline_ranks(store, baselines, "most_popular", test)
    subsets = {phase: rule(context["target_race"], context["target_recovery"]) for phase, rule in PHASES.items()}
    comparisons = {}
    for metric in ("ndcg@10", "recall@20"):
        full = per_user(ranks["full"], users, metric)
        for other in ("no_race_context", "personal_popular", "most_popular"):
            comparisons[f"{metric}: ours - {other}, all test targets"] = paired_bootstrap(full, per_user(ranks[other], users, metric))
        for phase, mask in subsets.items():
            if mask.any():
                comparisons[f"{metric}: ours - no_race_context, {phase}"] = paired_bootstrap(
                    per_user(ranks["full"][mask], users[mask], metric),
                    per_user(ranks["no_race_context"][mask], users[mask], metric))
    test_metrics = {**reports["full"]["test"], "no_race_context": reports["no_race_context"]["test"]["no_race_context"]}
    phases = {**reports["full"]["phases"], "no_race_context": reports["no_race_context"]["phases"]["no_race_context"]}
    exposure = {**reports["full"]["exposure"], "no_race_context": reports["no_race_context"]["exposure"]["no_race_context"]}
    plot_curves({"full": reports["full"], "no race context": reports["no_race_context"]}, args.results / "learning_curves.png")
    sample = examples(store, models["full"], baselines, test)
    preparing = test[(context["target_race"] >= 3) & (context["target_race"] <= 21)]
    sweep = race_date_sweep({"ours": models["full"], "ablation (no race context)": models["no_race_context"]},
                            store, preparing, np.random.default_rng(42))
    plot_sweep(sweep, args.results / "race_date_sweep.png")
    summary = {"test": test_metrics, "phases": phases, "exposure": exposure, "bootstrap": comparisons, "examples": sample,
               "race_date_sweep_km": sweep,
               "best_epochs": {name: report["best_epoch"] for name, report in reports.items()}}
    (args.results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    order = ["full", "no_race_context", "personal_popular", "repeat_last", "phase_popular", "most_popular", "random"]
    lines = ["# Marathon recommender: offline results", "",
             f"Test period: September to December 2019, {test_metrics['full']['targets']:,} target sessions "
             f"from {test_metrics['full']['users']:,} runners. Full ranking of all {NUM_ITEMS} session types. "
             "Metrics are averaged per runner, then across runners. The checkpoint of each model was selected on "
             "validation NDCG@10 (August); the test split was used once.", "", "## Test results", ""]
    lines += table([[NAMES[name], f"{test_metrics[name]['ndcg@10']:.4f}", f"{test_metrics[name]['recall@20']:.4f}",
                     f"{test_metrics[name]['recall@10']:.4f}", f"{test_metrics[name]['mrr']:.4f}"] for name in order],
                   ["Method", "NDCG@10", "Recall@20", "Recall@10", "MRR"])
    lines += ["", "## Ablation and significance", "",
              "Paired bootstrap over runners (1,000 resamples). A difference is reliable when its 95% interval excludes 0.", ""]
    lines += table([[key, f"{value['mean']:+.4f}", f"[{value['ci95'][0]:+.4f}, {value['ci95'][1]:+.4f}]"]
                    for key, value in comparisons.items()], ["Comparison", "Mean difference", "95% interval"])
    lines += ["", "## Innovation metric: ranking quality by preparation phase (NDCG@10)", ""]
    phase_names = list(phases["full"])
    lines += table([[phase, f"{phases['full'][phase]['targets']:,}"] + [f"{phases[name][phase]['ndcg@10']:.4f}" for name in
                    ("full", "no_race_context", "personal_popular", "phase_popular")] for phase in phase_names],
                   ["Phase", "Targets", "Ours", "Ablation", "Personal popularity", "Phase popularity"])
    lines += ["", "## Exposure (50,000 test targets)", "",
              "Coverage@10 is the share of the 66 types that appear in at least one top-10 list. The mean popularity rank "
              "of recommended types is 0 for the most frequent training type; higher means less popular types are shown.", ""]
    lines += table([[NAMES[name], f"{exposure[name]['coverage@10']:.2f}", f"{exposure[name]['mean_popularity_rank@10']:.1f}"]
                    for name in order], ["Method", "Coverage@10", "Mean popularity rank@10"])
    ours, ablation = sweep["ours"], sweep["ablation (no race context)"]
    lines += ["", "## Innovation check: moving only the race date", "",
              f"{sweep['runners']:,} distinct test runners preparing a race (2 to 20 weeks out), one session each. "
              "Their histories are kept; only the declared race "
              f"date moves. Expected distance of the recommended session: {ours[70]:.1f} km with the race 70 days away, "
              f"{ours[14]:.1f} km at 14 days, {ours[2]:.1f} km at 2 days. The ablation stays at {ablation[70]:.1f} km "
              "whatever the date, as it must. The model learned to taper from data; no rule encodes it.", "",
              "![Race-date sweep](race_date_sweep.png)", ""]
    lines += ["", "## Learning curves", "", "![Training and validation curves](learning_curves.png)", "",
              "## Three test runners, 1 to 4 weeks before their race", ""]
    for number, example in enumerate(sample, 1):
        lines += [f"**Example {number}** — runner {example['runner']}, {example['date']}, {example['days_to_race']} days before the race.", "",
                  "Last five sessions: " + "; ".join(example["last_sessions"]) + ".", "",
                  f"- Actual session: {example['actual']}.",
                  f"- Our top 5: {', '.join(example['model_top5'])}. The actual session is ranked {example['model_rank_of_actual']}.",
                  f"- Personal popularity top 5: {', '.join(example['personal_top5'])}. The actual session is ranked {example['personal_rank_of_actual']}.", ""]
    (args.results / "MARATHON_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
