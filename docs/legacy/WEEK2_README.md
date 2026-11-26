# Week 2 first build

This adds the first executable model, ranking metrics, and API contract for StridePlan.

- **Model** — a GRU reads earlier session template IDs and numeric features. It outputs ranking logits and relative-effort logits for each candidate. The smoke command checks a forward and backward pass on synthetic histories, including a new user with zero sessions.
- **Evaluation** — binary-relevance Recall@K and NDCG@K, macro-averaged across users, with known-answer tests. A training-frequency baseline is included. Empty relevance sets must be excluded explicitly by the evaluation protocol.
- **API** — `POST /recommend` returns clearly marked mock sessions that respect the requested duration and `top_k`. `/health`, `/docs`, and `/openapi.json` expose the contract.

## Run it

Requires Python 3.12+ for the pinned model/API dependencies. Verified under Ubuntu. The standalone data pipeline still supports Python 3.9+. Create and activate a virtual environment using your shell, then run these commands from the repository root. On Linux, use `python3` if needed.

```text
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m strideplan.smoke
python -m unittest discover -s tests
python -m unittest discover -s strideplan_pipeline -p "test_*.py"
python -m uvicorn strideplan.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` and try `POST /recommend` with this body.

```json
{"user_id": "demo", "sport": "run", "max_duration_minutes": 40, "top_k": 5}
```

The response has `mode: "mock"`, `personalized: false`, and two demo runs of 20 and 30 minutes. Budgets below 20 minutes return an empty list.

## Status for the report

The data pipeline has processed the full dataset. The model forward pass, metric harness, and mock API can now be executed. Report their checks alongside the Week 1 schedule status.

The next step is to build the data-driven template catalogue and chronological training examples from the prepared CSVs, then train both prediction heads. Feature scaling and template boundaries must be fitted on training data only. The smoke inputs and metric values are synthetic. The API uses a fixed demo catalogue and does not call the model yet. Effort calibration, conformal risk control, LLM parsing, interaction logging, and real offline performance remain pending.

The implementation uses PyTorch's [GRU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html) and FastAPI's [testing tools](https://fastapi.tiangolo.com/tutorial/testing/).

## Evaluation and training decisions

The default metric cutoffs are @1 and @3. Pass larger cutoffs explicitly when the catalogue size justifies them, and report eligible candidate counts. With one relevant next template, Recall@K is also hit rate. The demo template IDs are synthetic placeholders. Duration, pace and terrain grouping for the real catalogue must be agreed before training. Missing sensor values and GPS outliers also need a training-only preprocessing policy.

Tests are split by component. The metrics need only Python and can be run with python -m unittest discover -s tests -p test_evaluation.py. API tests are in test_api.py and require FastAPI and its test client. Model tests are in test_model.py and require PyTorch. The pinned NumPy dependency provides PyTorch array interoperability.
