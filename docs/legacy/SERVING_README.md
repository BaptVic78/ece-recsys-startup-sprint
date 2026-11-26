# Serving the trained ranker

The API now serves the GRU trained by `strideplan.experiment`, instead of the Week 2 mock catalogue. This is the base for the planned feedback loop: each response carries the model version and every category's ID, so later feedback can refer to them.

## What is served

- **Bundle.** Training writes `bundle.json` next to `ranker.pt`. It holds the architecture, the training-only catalogue boundaries and feature scaler, the global popularity order, the history length and the speed filter. `model_version` is a hash of the weights. Loading fails if the weights or the feature list no longer match.
- **Shared preprocessing.** `strideplan/features.py` is used by both training and serving. A test checks that a served history gets the same categories and scaled features as in training.
- **Ranking.** The latest 10 usable sessions go through the GRU. Every category is ranked by softmax probability, with ties broken by category ID as in offline evaluation. Sessions without a valid GPS speed are ignored, as in training.
- **Cold start.** A user with no usable session gets global training popularity (`strategy: "global_popularity"`, `personalized: false`, `score: null`).
- **Time budget.** A category is kept if its shortest duration fits `max_duration_minutes`. Its upper duration is then capped to the budget. Outer bands are bounded by the training minimum and maximum.
- **Effort.** The effort head is still untrained. Its output is not served.

## Run it

Python 3.12+, with the dependencies from the [Week 2 guide](WEEK2_README.md). A run made before this change has no `bundle.json`. Rerun the experiment: it is seeded and deterministic.

```text
python -m strideplan.experiment
python -m uvicorn strideplan.api:app --host 127.0.0.1 --port 8000
```

| Variable | Default | Role |
| --- | --- | --- |
| `STRIDEPLAN_BUNDLE` | `artifacts/first_results` | Bundle directory. If `bundle.json` is missing, the API serves the mock catalogue with `mode: "mock"` |
| `STRIDEPLAN_SESSIONS` | `data/processed/sessions.csv` | Known Endomondo histories, looked up by `user_id`. Optional |

Serving uses every split of a known user's history, because it predicts the session after the dataset ends. Offline evaluation keeps its own leakage rules.

## Endpoints

- `GET /health` shows the mode, the model version and the number of known users.
- `GET /catalogue` lists every category with its duration and speed ranges.
- `POST /recommend` takes a `user_id`, an inline `history`, or both. An inline history replaces the stored one.

```json
{"user_id": "demo", "max_duration_minutes": 45, "top_k": 3,
 "history": [{"start_timestamp": 1400000000, "duration_seconds": 2400, "avg_speed_mps": 2.9, "avg_heart_rate_bpm": 150}]}
```

Each response has `mode`, `model_version`, `personalized`, `strategy`, `history_sessions_used`, and `recommendations` with a category ID, title, ranges and score.

## Tests

```text
python -m unittest discover -s tests
```

`MockApiTest` needs only FastAPI. The inference tests and `ModelApiTest` need PyTorch and build a small untrained bundle from synthetic sessions.

## Feedback

Responses now carry a `rec_id`, and feedback is described in [the feedback-loop guide](FEEDBACK_README.md). When `artifacts/models/registry.json` exists, the API serves its current version.
