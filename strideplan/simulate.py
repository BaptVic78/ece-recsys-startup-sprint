"""Simulate the feedback loop with virtual runners built from Endomondo histories.

Each virtual runner has a category taste (smoothed counts of their last 30 runs) and a comfort intensity
(median intensity of those runs, optionally shifted to model a runner who now wants lighter or harder work).
Shown a ranked list, the runner examines it in order and picks a category with probability
taste(t) / max taste x exp(-(intensity(t) - comfort)^2 / (2 x 0.35^2)). Examined categories before the pick are
skipped (rejected). A picked category is completed and rated too_easy / ok / too_hard against the comfort level.

Three arms share the same runners, budgets and random draws:
  no_loop     the model only, no feedback used and completed runs not logged
  short_loop  feedback re-ranking, logged runs enter the history, 10% exploration
  full_loop   short_loop plus one retraining (strideplan.retrain) halfway, with the promotion gate

This is a behavioural model, not real users: it shows whether the loop adapts, not by how much real runners gain.
"""

import argparse
import json
import math
import random
import statistics
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch

from strideplan.bundle import Bundle, load_bundle, resolve_bundle_dir
from strideplan.inference import Recommender, load_histories
from strideplan.reranker import NO_FEEDBACK, Policy, intensity
from strideplan.retrain import RetrainConfig, featurize_histories, retrain
from strideplan.service import StridePlanService
from strideplan.store import Store


ARMS = ("no_loop", "short_loop", "full_loop")
COMFORT_WIDTH = 0.35
EFFORT_MARGIN = 0.25
BUDGETS = (30, 45, 60, 90, 120)


def build_runners(histories, catalogue, count, comfort_shift, seed, minimum_sessions=10):
    candidates = sorted(user for user, rows in histories.items() if len(rows) >= minimum_sessions)
    chosen = sorted(random.Random(seed).sample(candidates, min(count, len(candidates))))
    runners = {}
    for user in chosen:
        recent = [row["template"] for row in histories[user][-30:]]
        counts = Counter(recent)
        taste = {t: (counts[t] + 0.5) for t in range(1, catalogue["num_templates"] + 1)}
        top = max(taste.values())
        comfort = statistics.median(intensity(t, catalogue) for t in recent) + comfort_shift
        runners[user] = {"taste": {t: value / top for t, value in taste.items()},
                         "comfort": max(-1.0, min(1.0, comfort))}
    return runners


def appeal(runner, template, catalogue):
    distance = intensity(template, catalogue) - runner["comfort"]
    return runner["taste"][template] * math.exp(-distance ** 2 / (2 * COMFORT_WIDTH ** 2))


def effort_rating(runner, template, catalogue, draw):
    distance = intensity(template, catalogue) - runner["comfort"]
    if draw < 0.1:
        return "ok"
    return "too_hard" if distance > EFFORT_MARGIN else "too_easy" if distance < -EFFORT_MARGIN else "ok"


def session_for(template, recommender, start):
    item = recommender.templates[template]
    minutes = (item["duration_min_minutes"] + item["duration_max_minutes"]) / 2
    speed = (item["speed_min_kmh"] + item["speed_max_kmh"]) / 2 / 3.6
    return {"start_timestamp": start, "duration_seconds": minutes * 60, "avg_speed_mps": speed}


def list_quality(runner, served, eligible, catalogue):
    """NDCG of the runner's appeal over the served list, against the best list from eligible categories."""
    gains = [appeal(runner, template, catalogue) for template in served]
    ideal = sorted((appeal(runner, template, catalogue) for template in eligible), reverse=True)[:len(served)]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    best = sum(gain / math.log2(index + 2) for index, gain in enumerate(ideal))
    return dcg / best if best else 0.0


def run_arm(arm, bundle, base_histories, runners, rounds, seed, top_k, start_time, retrain_config):
    store = Store(":memory:")
    policy = NO_FEEDBACK if arm == "no_loop" else Policy()
    service = StridePlanService(Recommender(bundle), store, base_histories, policy, seed=seed)
    catalogue = bundle.catalogue
    per_round = []
    retrain_report = None
    for round_index in range(rounds):
        if arm == "full_loop" and round_index == rounds // 2:
            candidate, retrain_report = retrain(service.recommender.bundle, base_histories, store, retrain_config)
            if retrain_report["promoted"]:
                served = service.recommender.bundle
                service.recommender = Recommender(Bundle(
                    model=candidate, catalogue=served.catalogue, scaler=served.scaler, popularity=served.popularity,
                    max_history=served.max_history, max_speed=served.max_speed, version="retrained",
                    effort_head_trained=retrain_report["serve_effort"], lineage=(served.version,)))
        totals = Counter()
        quality = []
        for user, runner in runners.items():
            now = start_time + round_index * 3 * 86400 + 3600
            budget = random.Random(f"{seed}:budget:{user}:{round_index}").choice(BUDGETS)
            result = service.recommend(user, budget, top_k, now=now)
            served = [item["template_id"] for item in result["recommendations"]]
            eligible = [item["template_id"] for item in service.recommender.within_budget(
                [{"template_id": t, "score": None} for t in range(1, catalogue["num_templates"] + 1)], budget)]
            quality.append(list_quality(runner, served, eligible, catalogue))
            picked = None
            for position, template in enumerate(served):
                draw = random.Random(f"{seed}:{user}:{round_index}:{template}").random()
                if draw < appeal(runner, template, catalogue):
                    picked = (position, template)
                    break
            examined = served[:picked[0]] if picked else served
            for offset, template in enumerate(examined):
                service.record_feedback(result["rec_id"], template, "rejected", now=now + 60 + offset)
            totals["lists"] += 1
            if picked is None:
                continue
            position, template = picked
            rating = effort_rating(runner, template, catalogue,
                                   random.Random(f"{seed}:effort:{user}:{round_index}").random())
            session = session_for(template, service.recommender, now + 1800) if arm != "no_loop" else None
            service.record_feedback(result["rec_id"], template, "completed", rating, session, now=now + 7200)
            totals["picked"] += 1
            totals["reciprocal_rank"] += 1 / (position + 1)
            totals[rating] += 1
        picked = totals["picked"]
        per_round.append({"round": round_index + 1,
                          "acceptance": picked / totals["lists"],
                          "mrr": totals["reciprocal_rank"] / totals["lists"],
                          "comfortable_share": totals["ok"] / picked if picked else 0.0,
                          "too_hard_share": totals["too_hard"] / picked if picked else 0.0,
                          "list_quality": sum(quality) / len(quality)})
    store.close()
    return {"rounds": per_round, "retrain": retrain_report}


def summarise(runs):
    """Mean and standard deviation across seeds of each arm's last-round metrics and of round 1."""
    summary = {}
    for arm in ARMS:
        summary[arm] = {}
        for label, index in (("first_round", 0), ("last_round", -1)):
            values = [run[arm]["rounds"][index] for run in runs]
            summary[arm][label] = {metric: {"mean": statistics.mean(v[metric] for v in values),
                                            "std": statistics.pstdev(v[metric] for v in values)}
                                   for metric in values[0] if metric != "round"}
    return summary


def write_report(output, result):
    output.mkdir(parents=True, exist_ok=True)
    (output / "feedback_simulation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                                     encoding="utf-8")
    names = {"no_loop": "No loop", "short_loop": "Short loop", "full_loop": "Short + retraining"}
    metrics = {"acceptance": "Acceptance", "mrr": "MRR", "list_quality": "List quality (NDCG)",
               "comfortable_share": "Rated OK", "too_hard_share": "Rated too hard"}
    config = result["config"]
    lines = ["# Feedback-loop simulation", "",
             f"{config['runners']} virtual runners, {config['rounds']} rounds of {config['top_k']} recommendations, "
             f"{len(config['seeds'])} seeds. Comfort shifts: "
             + ", ".join(f"{name} {shift:+.2f}" for name, shift in config["scenarios"].items()) + ". "
             "Values are means across seeds of each scenario's last round, with standard deviation in brackets.", ""]
    for scenario, summary in result["summary"].items():
        lines += [f"## Scenario: {scenario}", "", "| Arm | " + " | ".join(metrics.values()) + " |",
                  "| --- |" + " ---: |" * len(metrics)]
        for arm in ARMS:
            last = summary[arm]["last_round"]
            lines.append(f"| {names[arm]} | " + " | ".join(
                f"{last[m]['mean']:.3f} ({last[m]['std']:.3f})" for m in metrics) + " |")
        reports = [run["full_loop"]["retrain"] for run in result["runs"][scenario] if run["full_loop"]["retrain"]]
        promoted = sum(report["promoted"] for report in reports)
        effort = sum(report["serve_effort"] for report in reports)
        lines += ["", f"Retraining promoted the candidate in {promoted} of {len(reports)} seeds; "
                      f"the learned effort head was served in {effort}.", ""]
    lines += ["## Reading this", "",
              "- *Acceptance* is the share of lists where the runner picked a category; *MRR* also rewards picking "
              "it high in the list.",
              "- *List quality* is the NDCG of the runner's hidden appeal over the served list, against the best "
              "list allowed by the time budget. It needs no random draw.",
              "- *Rated OK* and *Rated too hard* use the three effort buttons of completed workouts.",
              "- Runners and their reactions are simulated from Endomondo histories, so the GRU saw the same "
              "habits during training. The comparison shows whether the loop adapts; it is not a measured gain "
              "on real users.", "",
              "## Reproduce", "", "```text", "python -m strideplan.simulate", "```", ""]
    (output / "FEEDBACK_SIMULATION.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, default=Path("artifacts/first_results"))
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--runners", type=int, default=300)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--replay-examples", type=int, default=10000)
    args = parser.parse_args()
    if args.rounds < 2 or args.runners < 1:
        parser.error("Use at least 2 rounds and 1 runner")
    torch.set_num_threads(2)
    bundle = load_bundle(resolve_bundle_dir(args.bundle))
    base_histories = load_histories(args.data / "sessions.csv")
    rows = featurize_histories(base_histories, bundle.catalogue, bundle.scaler, bundle.max_speed)
    start_time = max(row["time"] for user_rows in rows.values() for row in user_rows) + 86400
    retrain_config = RetrainConfig(epochs=1, replay_examples=args.replay_examples)
    scenarios = {"stable habits": 0.0, "wants lighter runs": -0.5}
    result = {"config": {"runners": args.runners, "rounds": args.rounds, "top_k": args.top_k, "seeds": args.seeds,
                         "scenarios": scenarios, "base_version": bundle.version, "retrain": asdict(retrain_config)},
              "runs": {}, "summary": {}}
    for scenario, shift in scenarios.items():
        runs = []
        for seed in args.seeds:
            runners = build_runners(rows, bundle.catalogue, args.runners, shift, seed)
            run = {arm: run_arm(arm, bundle, base_histories, runners, args.rounds, seed, args.top_k, start_time,
                                retrain_config) for arm in ARMS}
            runs.append(run)
            print(json.dumps({"scenario": scenario, "seed": seed,
                              **{arm: run[arm]["rounds"][-1] for arm in ARMS}}), flush=True)
        result["runs"][scenario] = runs
        result["summary"][scenario] = summarise(runs)
    write_report(args.output, result)
    print(f"Wrote {args.output / 'FEEDBACK_SIMULATION.md'}")


if __name__ == "__main__":
    main()
