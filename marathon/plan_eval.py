"""Compare generated training weeks with the weeks test runners actually ran.

For each sampled test runner we take a random 7-day window from the test period with at least
three runs and no race inside it. The planner gets the runner's history before the window, the
real run dates and the declared race date, then fills those dates with `greedy` (top-ranked
session every day) and `sample` (nucleus sampling from the model). Each generated week is
compared with the real week: variety of session types, repetition, weekly volume, spread
between the longest and shortest run, and per-day agreement with the real session type.

A third method, `phase_sample`, draws each day from the training frequencies of its preparation
phase (the cold-start fallback), which ignores the runner. It shows which metrics reward realism
alone: variety and the overall type mix, which it matches too. Per-day agreement and weekly volume
need the runner's history.

`--demo` runs the same comparison on the committed demo runners (no download needed): weeks
starting 28, 21, 14 and 7 days before their race, with their real run dates.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from marathon.data import NUM_ITEMS, annotate_sessions
from marathon.sequences import race_bucket
from marathon.serving import Recommender, day_rng, prescription

STRATEGIES = ("greedy", "sample")
METHODS = (*STRATEGIES, "phase_sample")


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


def demo_weeks(demo_dir, min_runs=3):
    """Weeks of the committed demo runners: four 7-day windows from their demo date, before the race."""
    runners = pd.read_csv(demo_dir / "runners.csv", parse_dates=["race_date", "today"])
    sessions = pd.read_csv(demo_dir / "sessions.csv", parse_dates=["date"])
    weeks = []
    for runner in runners.itertuples():
        own = annotate_sessions(sessions[sessions["runner"] == runner.runner].rename(columns={"runner": "athlete"}))
        for offset in range(0, 28, 7):
            start = runner.today + pd.Timedelta(days=offset)
            real = own[(own["date"] >= start) & (own["date"] < start + pd.Timedelta(days=7))
                       & (own["date"] < runner.race_date)]
            if len(real) >= min_runs:
                weeks.append({"athlete": int(runner.runner), "start": start, "real": real,
                              "history": own.loc[own["date"] < start, ["date", "km", "minutes"]],
                              "race_date": runner.race_date})
    return weeks


def phase_sample_plan(recommender, history, dates, race_date, seed):
    """Baseline that ignores the runner: draw each day from the training frequencies of its phase."""
    history = list(history)
    planned = []
    for date in dates:
        _, info = recommender.rank(history, date, race_date)
        days = info["days_to_race"]
        counts = recommender.phase_popularity[int(race_bucket([np.nan if days is None else days])[0])]
        chosen = int(day_rng(seed, date).choice(NUM_ITEMS, p=counts / counts.sum()))
        session = prescription(chosen + 1, info["reference_pace"])
        planned.append(session)
        history.append({"date": date, "km": session["km"], "minutes": session["minutes"]})
    return planned


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
    parser.add_argument("--demo", action="store_true", help="use the committed demo runners instead of the test set")
    parser.add_argument("--demo-dir", type=Path, default=Path("demo"))
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    if args.demo:
        weeks = demo_weeks(args.demo_dir)
    else:
        sessions = pd.read_parquet(args.data / "sessions.parquet")
        races = pd.read_csv(args.data / "races.csv", parse_dates=["race_date"])
        weeks = sample_weeks(sessions, races, args.runners, rng)
    recommender = Recommender(args.models)
    items = {method: [] for method in ("real", *METHODS)}
    rows = []
    for number, week in enumerate(weeks):
        real = week["real"]
        dates = real["date"].tolist()
        real_items = real["item"].to_numpy()
        items["real"] += real_items.tolist()
        rows.append({"method": "real", **week_metrics(real_items, real["km"]), "type_accuracy": 1.0,
                     "weekly_km_abs_error": 0.0})
        history = week["history"].to_dict(orient="records")
        seed = args.seed * 1_000_003 + number
        for strategy in METHODS:
            if strategy == "phase_sample":
                planned = phase_sample_plan(recommender, history, dates, week["race_date"], seed)
            else:
                days = recommender.plan(history, dates, week["race_date"], top_k=1, strategy=strategy, seed=seed)
                planned = [day["planned"] for day in days]
            planned_items = [item["item"] for item in planned]
            items[strategy] += planned_items
            metrics = week_metrics(planned_items, [item["km"] for item in planned])
            rows.append({"method": strategy, **metrics,
                         "type_accuracy": float((np.array(planned_items) == real_items).mean()),
                         "weekly_km_abs_error": abs(metrics["weekly_km"] - float(real["km"].sum()))})
    summary = pd.DataFrame(rows).groupby("method").mean().loc[["real", *METHODS]]
    real_histogram = type_histogram(items["real"])
    divergence = {method: jensen_shannon(type_histogram(items[method]), real_histogram) for method in METHODS}
    result = {"weeks": len(weeks), "runners": len({week["athlete"] for week in weeks}),
              "sessions": len(items["real"]), "summary": summary.round(4).to_dict(orient="index"),
              "js_divergence_to_real_type_distribution": divergence,
              "distinct_types_used": {method: len(set(values)) for method, values in items.items()}}
    suffix = "_demo" if args.demo else ""
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / f"plan_diversity{suffix}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    names = {"real": "Real weeks (what runners did)", "greedy": "Greedy plan (top session each day)",
             "sample": "Sampled plan (nucleus p=0.9, default)",
             "phase_sample": "Phase popularity sampled (ignores the runner)"}
    source = ("demo runners (committed sample), weeks starting 28, 21, 14 and 7 days before their race"
              if args.demo else "test weeks, one per runner, each a random 7-day window from September-December "
                                "2019 with at least three runs and no race")
    lines = ["# Weekly plans: greedy versus sampled", "",
             f"{result['weeks']} weeks from {result['runners']} runners ({result['sessions']:,} real sessions): {source}. "
             "The planner receives the runner's history before the window, the real run dates and the declared race "
             "date.", "",
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
              "Greedy decoding returns the single most likely type every day. No type is very likely and the most "
              "likely one changes little within a week, so greedy weeks repeat one session. Sampling from the same "
              "model reproduces the variety of real weeks but matches the session actually run on fewer days. "
              "Sampling the phase popularity, which ignores the runner, also looks varied: variety and type mix "
              "measure realism, while per-day agreement and weekly km error show what the runner's history adds. "
              "The ranking quality of the model itself is measured in MARATHON_RESULTS.md.", "",
              "Reproduce: `python -m marathon.plan_eval" + (" --demo`." if args.demo else "` (needs the processed data).")]
    (args.results / f"PLAN_DIVERSITY{suffix.upper()}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
