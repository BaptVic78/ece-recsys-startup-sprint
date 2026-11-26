"""Compare generated training weeks with the weeks test runners actually ran.

For each sampled test runner we take a random 7-day window from the test period with at least
three runs and no race inside it. The planner gets the runner's history before the window, the
real run dates and the declared race date, then fills those dates with `greedy` (top-ranked
session every day) and `sample` (nucleus sampling from the model). Each generated week is
compared with the real week: variety of session types, repetition, weekly volume, spread
between the longest and shortest run, and per-day agreement with the real session type.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from marathon.data import NUM_ITEMS
from marathon.serving import Recommender

STRATEGIES = ("greedy", "sample")


def sample_weeks(sessions, races, count, rng, min_runs=3, min_history=5):
    """One random eligible test week per runner, for up to `count` runners."""
    sessions = sessions.sort_values(["athlete", "date"], kind="stable")
    test = sessions[(sessions["split"] == "test") & ~sessions["is_race"]]
    races = races.sort_values("race_date")
    weeks = []
    for athlete in rng.permutation(test["athlete"].unique()):
        if len(weeks) >= count:
            break
        runner = sessions[sessions["athlete"] == athlete]
        starts = runner.loc[(runner["split"] == "test") & ~runner["is_race"], "date"].to_numpy()
        for start in rng.permutation(starts):
            start = pd.Timestamp(start)
            window = runner[(runner["date"] >= start) & (runner["date"] < start + pd.Timedelta(days=7))]
            history = runner[runner["date"] < start]
            if len(window) < min_runs or window["is_race"].any() or len(history) < min_history:
                continue
            upcoming = races[(races["athlete"] == athlete) & (races["race_date"] > start)]
            weeks.append({"athlete": int(athlete), "start": start, "history": history[["date", "km", "minutes"]],
                          "real": window, "race_date": upcoming["race_date"].iloc[0] if len(upcoming) else None})
            break
    return weeks


def week_metrics(items, km):
    items = np.asarray(items)
    km = np.asarray(km, dtype=float)
    return {"distinct_types_share": len(set(items.tolist())) / len(items),
            "consecutive_repeat_rate": float((items[1:] == items[:-1]).mean()) if len(items) > 1 else 0.0,
            "weekly_km": float(km.sum()), "longest_over_shortest": float(km.max() / km.min()),
            "has_run_of_18km_or_more": float(km.max() >= 18)}


def type_histogram(frame_items):
    counts = np.bincount(np.asarray(frame_items, dtype=int) - 1, minlength=NUM_ITEMS).astype(float)
    return counts / counts.sum()


def jensen_shannon(p, q):
    m = (p + q) / 2

    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())

    return (kl(p, m) + kl(q, m)) / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/marathon"))
    parser.add_argument("--models", type=Path, default=Path("models/marathon"))
    parser.add_argument("--results", type=Path, default=Path("results/marathon"))
    parser.add_argument("--runners", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    sessions = pd.read_parquet(args.data / "sessions.parquet")
    races = pd.read_csv(args.data / "races.csv", parse_dates=["race_date"])
    weeks = sample_weeks(sessions, races, args.runners, rng)
    recommender = Recommender(args.models)
    items = {method: [] for method in ("real", *STRATEGIES)}
    rows = []
    for number, week in enumerate(weeks):
        real = week["real"]
        dates = real["date"].tolist()
        real_items = real["item"].to_numpy()
        items["real"] += real_items.tolist()
        rows.append({"method": "real", **week_metrics(real_items, real["km"]), "type_accuracy": 1.0,
                     "weekly_km_abs_error": 0.0})
        history = week["history"].to_dict(orient="records")
        for strategy in STRATEGIES:
            days = recommender.plan(history, dates, week["race_date"], top_k=1, strategy=strategy,
                                    seed=args.seed * 1_000_003 + number)
            planned = [day["planned"] for day in days]
            planned_items = [item["item"] for item in planned]
            items[strategy] += planned_items
            metrics = week_metrics(planned_items, [item["km"] for item in planned])
            rows.append({"method": strategy, **metrics,
                         "type_accuracy": float((np.array(planned_items) == real_items).mean()),
                         "weekly_km_abs_error": abs(metrics["weekly_km"] - float(real["km"].sum()))})
    summary = pd.DataFrame(rows).groupby("method").mean().loc[["real", *STRATEGIES]]
    real_histogram = type_histogram(items["real"])
    divergence = {method: jensen_shannon(type_histogram(items[method]), real_histogram) for method in STRATEGIES}
    result = {"weeks": len(weeks), "runners": len({week["athlete"] for week in weeks}),
              "sessions": len(items["real"]), "summary": summary.round(4).to_dict(orient="index"),
              "js_divergence_to_real_type_distribution": divergence,
              "distinct_types_used": {method: len(set(values)) for method, values in items.items()}}
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "plan_diversity.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    names = {"real": "Real weeks (what runners did)", "greedy": "Greedy plan (top session each day)",
             "sample": "Sampled plan (nucleus p=0.9, default)"}
    lines = ["# Weekly plans: greedy versus sampled", "",
             f"{result['weeks']} test weeks, one per runner ({result['sessions']:,} real sessions), each a random 7-day "
             "window from September-December 2019 with at least three runs and no race. The planner receives the "
             "runner's history before the window, the real run dates and the declared race date.", "",
             "| Method | Distinct types / runs | Consecutive repeats | Weekly km | Longest / shortest | Week with 18+ km run | "
             "Same type as real day | Weekly km abs. error |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for method, values in summary.iterrows():
        lines.append(f"| {names[method]} | {values['distinct_types_share']:.2f} | {values['consecutive_repeat_rate']:.2f} | "
                     f"{values['weekly_km']:.1f} | {values['longest_over_shortest']:.2f} | "
                     f"{values['has_run_of_18km_or_more']:.2f} | {values['type_accuracy']:.3f} | "
                     f"{values['weekly_km_abs_error']:.1f} |")
    lines += ["", "Jensen-Shannon divergence between the session types used across all generated weeks and those of "
              "the real weeks (0 = identical, 1 = disjoint): "
              + ", ".join(f"{method} {value:.3f}" for method, value in divergence.items())
              + f". Distinct session types used: " + ", ".join(
                  f"{method} {count}" for method, count in result["distinct_types_used"].items()) + ".", "",
              "Greedy decoding always returns the single most likely type, then feeds it back as if done, so weeks "
              "repeat one session. Sampling from the same model reproduces the variety of real weeks. Per-day type "
              "accuracy is low for both because a runner's exact session on a given day is hard to predict; the "
              "ranking quality of the model itself is measured in MARATHON_RESULTS.md.", "",
              "Reproduce: `python -m marathon.plan_eval` (needs the processed data)."]
    (args.results / "PLAN_DIVERSITY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
