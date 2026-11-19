# Feedback loop

Runners can now react to each recommendation. Their feedback changes the next recommendation immediately and is used to retrain the model.

```text
 /recommend ──► served list + rec_id ──► runner skips / does a workout ──► /feedback
      ▲                                                                       │
      │  short loop: re-ranking + new session in the GRU history (instant)    │
      └───────────────────────────────────────────────────────────────────────┤
      │  long loop: strideplan.retrain → promotion gate → model registry      │
      └───────────────────────────────────────────────────────────────────────┘
```

## Feedback that is collected

| Signal | How it is sent | Used by |
| --- | --- | --- |
| Skipped category | `action: "rejected"` | Re-ranking penalty, then unlikelihood loss in retraining |
| Chosen category | `action: "accepted"` | Positive with half weight in retraining |
| Done workout | `action: "completed"`, optional `session` | GRU history, positive in retraining |
| Effort | Three buttons on a completed workout: `too_easy`, `ok`, `too_hard` | Intensity shift, then effort head training |

Every served list is logged under a `rec_id` with the model version, the budget, each category's position and score, and whether it was an exploration slot. Feedback is refused if the `rec_id` is unknown, if the category was not in that list, or if effort is sent for anything but a completed workout. The log is SQLite (`STRIDEPLAN_DB`, default `artifacts/feedback/feedback.sqlite3`, excluded from Git).

## Short loop: the next recommendation changes at once

`strideplan/reranker.py` adjusts the model's log-probabilities using the last 90 days of a runner's feedback. Each change is listed in the item's `reasons`.

- **Skips.** A skipped category loses 1.5 log-units. The penalty halves every 7 days.
- **Effort.** The last 5 rated workouts give a bias. Each "too hard" counts −1 and each "too easy" +1, halving every 14 days. The bias shifts every category in proportion to its intensity: −1 for the shortest and slowest band, +1 for the longest and fastest.
- **Learned effort.** Once a retrained model serves its effort head, categories predicted far from "ok" lose up to 1 log-unit.
- **History.** A completed workout with its `session` enters the runner's history, so the GRU input changes too. Almost every training run has heart rate. When a logged session has none, the runner's recent median is used. Otherwise the missing-heart-rate flag would mark the run as unusual to the GRU. In the first simulation this flag alone made the loop worse than no loop at all.
- **Exploration.** In 10% of responses (`STRIDEPLAN_EPSILON`), the last slot shows another eligible category, marked `explored: true`. This limits the loop's tendency to show only what was already shown.

## Long loop: retraining with a promotion gate

```text
python -m strideplan.retrain
python -m strideplan.retrain --rollback
```

Options: `--learning-rate` (default 1e-4), `--trainable all|embeddings|head` (which ranking weights move), `--anchor-weight` (L2 pull towards the served weights). The defaults come from the retraining comparison below.

- The model is fine-tuned from the served weights. The catalogue and scaler are kept, so category IDs never change.
- Completed and accepted workouts are positives (cross-entropy). Skips lower the skipped category's probability (unlikelihood). Endomondo training examples are replayed alongside the feedback so the model does not forget the original task.
- The effort head is trained only once there are at least 20 effort ratings. It is served only if it beats a constant "ok" guess on held-out ratings.
- The latest 20% of feedback (by time) is held out. Every training context is the runner's history strictly before the recommendation was served.
- **Promotion** needs two things: Endomondo validation NDCG@3 within 0.005 of the current model, and a held-out feedback score (positive MRR minus the share of skipped categories still in the top 3) at least as good. A rejected candidate is saved but not served.
- `artifacts/models/registry.json` records the base, promoted, rejected and rolled-back versions. Each version folder holds its weights, `bundle.json` with lineage and the full retraining report. The API serves the registry's current version on restart.

## Try it

```text
python -m strideplan.experiment
python -m uvicorn strideplan.api:app --host 127.0.0.1 --port 8000
```

In `/docs`:

1. `POST /recommend` with `{"user_id": "<an Endomondo user_id>", "max_duration_minutes": 60, "top_k": 3}`. Keep the `rec_id`.
2. `POST /feedback` with `{"rec_id": "...", "template_id": "<first category>", "action": "rejected"}`.
3. `POST /recommend` again. The skipped category moves down, and `feedback_applied` is true.
4. `POST /feedback` with `"action": "completed"`, `"effort": "too_hard"` and a `session`. The next list shifts towards lighter categories.
5. `GET /users/<user_id>/history` shows logged sessions and feedback.

`POST /sessions` logs a workout that was not recommended.

## Simulation

```text
python -m strideplan.simulate
python -m strideplan.retrain_ablation
```

Real users are not available, so `strideplan/simulate.py` builds virtual runners from Endomondo histories.

- **Taste.** Each runner's taste comes from the categories of their last 30 runs.
- **Personal load.** A category's load for a runner is the mean z-score of its typical training session (log duration and log speed) against that runner's own median and spread.
- **Effort.** Ratings come from that load. It deliberately does not use the re-ranker's band-intensity scale. An earlier version did, which made part of the effort gain true by construction.

Three arms share the same runners, budgets and random draws:
- **No loop**: the model only;
- **Short loop**: re-ranking, logged history and exploration;
- **Short + retraining**: the short loop, plus retraining before rounds 3, 5 and 7, each through the promotion gate.

There are two scenarios: stable habits, and runners who now want runs one spread lighter than usual. See [the simulation report](results/FEEDBACK_SIMULATION.md).

### Results (300 runners, 8 rounds, 3 seeds, model `gru-498e31cda864`)

| Last round | No loop | Short loop | Short + retraining |
| --- | ---: | ---: | ---: |
| List quality, stable habits | 0.813 | **0.822** | 0.821 |
| MRR, stable habits | 0.618 | **0.661** | 0.659 |
| List quality, wants lighter runs | 0.799 | 0.799 | **0.808** |
| MRR, wants lighter runs | 0.481 | 0.510 | **0.521** |
| Rated "too hard", wants lighter runs | 20.5% | 15.6% | **15.0%** |

Seed standard deviations are about 0.001 to 0.007 for list quality and 3 to 4 points for "too hard". See the report.

- **The short loop helps, modestly.**
  - It raises MRR in both scenarios: runners find a category they want higher in the list.
  - It cuts "too hard" ratings by about 5 points for runners who want lighter work, about 1.4 standard deviations.
  - List quality moves little.
- **Retraining helps only when needs change.** With lighter runs, it adds about 0.009 list quality over the short loop. With stable habits, it adds nothing, and the gate promoted 2 of 9 candidates.
- **The learned effort head was never served.** Most ratings are "ok", and a constant "ok" guess stays hard to beat.

### Retraining set-ups

`strideplan.retrain_ablation` compares retraining variants against the short loop on the same runners: 2 scenarios × 2 seeds, 300 runners. The full table is in `results/retrain_ablation.json`.

| Variant | List quality vs short loop | "Too hard" | Promoted | Validation NDCG@3 change |
| --- | ---: | ---: | ---: | ---: |
| All weights, lr 3e-4, once (previous default) | +0.002 | −0.2 pt | 1/4 | −0.005 |
| All weights, lr 3e-4, 3 times | +0.006 | −0.6 pt | 2/12 | −0.005 |
| **All weights, lr 1e-4, 3 times (default)** | **+0.006** | −0.4 pt | **5/12** | **−0.003** |
| L2 anchor to served weights | +0.003 | −1.0 pt | 2/12 | −0.005 |
| Frozen GRU (`--trainable embeddings`) | +0.004 | −0.4 pt | 5/12 | −0.000 |
| Ranking bias only (`--trainable head`) | +0.001 | −0.1 pt | 7/12 | +0.001 |

All gains are within seed noise.

- Fine-tuning the whole network keeps drifting away from the original task: validation NDCG@3 drops by 0.003 to 0.005. The best base checkpoint was selected on validation, so further training on replayed data tends to overfit.
- Freezing the GRU, or training only the ranking bias, preserves validation. They also learn less from feedback.
- The default is now a lower learning rate (1e-4) with repeated retraining.
- The honest conclusion is that the short loop delivers most of the benefit. Retraining is a safe pipeline: the gate never served a candidate that lost more than 0.005 validation NDCG@3. It does not yet bring a measurable gain.

### Bugs the simulation found

- **Missing heart rate in logged sessions.** It made the loop worse than no loop (list quality 0.57 against 0.76). Logged sessions now take the runner's recent median heart rate.
- **Circular effort model.** An effort model that reused the re-ranker's intensity scale inflated the early results: list quality 0.76 to 0.81, "too hard" 39% to 31%. The figures above use the independent personal-load model.

## Limits

- The re-ranking weights are hand-set defaults (`Policy`), not tuned on real users.
- Feedback is stored with the raw `user_id`. Pseudonymise it before any real deployment.
- The API loads a promoted model only on restart.
- One process owns the SQLite file. Several API workers would need a server database.
