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
```

Real users are not available, so `strideplan/simulate.py` builds virtual runners from Endomondo histories. Each has a category taste and a comfort intensity, and examines lists, skips, completes and rates workouts. Three arms share the same runners, budgets and random draws:

- **No loop**: the model only;
- **Short loop**: re-ranking, logged history and exploration;
- **Short + retraining**: the short loop, plus one retraining halfway through.

There are two scenarios: stable habits, and runners who now want lighter runs than their history. See [the simulation report](results/FEEDBACK_SIMULATION.md). The simulator is a behavioural model, not a measure of real-user benefit.

### Results (300 runners, 8 rounds, 3 seeds, model `gru-498e31cda864`)

| Last round | No loop | Short loop | Short + retraining |
| --- | ---: | ---: | ---: |
| List quality, stable habits | 0.757 | **0.811** | 0.811 |
| Rated "too hard", stable habits | 3.7% | **2.3%** | 2.3% |
| List quality, wants lighter runs | 0.750 | 0.769 | **0.770** |
| Rated "too hard", wants lighter runs | 39.0% | 31.2% | **30.1%** |

- **The short loop does most of the work.** In both scenarios it beats the model alone by more than the spread across seeds. It raises acceptance and MRR, and lowers the share of workouts rated too hard.
- **Retraining adds little so far, and the gate is strict.** In the stable scenario, no candidate was promoted in 3 seeds: validation NDCG@3 dropped by more than 0.005 (0.360 to about 0.350). With lighter runs, 2 of 3 seeds were promoted, with a small held-out gain. The learned effort head was served once, when it beat the constant "ok" guess. Retraining needs more feedback, or a gentler fine-tuning, before it pays off.
- **The first run of this simulation found a real serving bug.** Logged sessions without heart rate made the loop worse than no loop (list quality 0.57). They are now imputed, as described above.

## Limits

- The re-ranking weights are hand-set defaults (`Policy`), not tuned on real users.
- Feedback is stored with the raw `user_id`. Pseudonymise it before any real deployment.
- The API loads a promoted model only on restart.
- One process owns the SQLite file. Several API workers would need a server database.
