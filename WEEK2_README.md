# Week 2 first build

This adds the first executable model, ranking metrics, and API contract for StridePlan.

- **Model** — a GRU reads earlier session template IDs and numeric features. It outputs ranking logits and relative-effort logits for each candidate. The smoke command checks a forward and backward pass on synthetic histories, including a new user with zero sessions.
- **Evaluation** — binary-relevance Recall@K and NDCG@K, macro-averaged across users, with known-answer tests. A training-frequency baseline is included. Empty relevance sets must be excluded explicitly by the evaluation protocol.
- **API** — `POST /recommend` returns clearly marked mock sessions that respect the requested duration and `top_k`. `/health`, `/docs`, and `/openapi.json` expose the contract.

## Run it

Tested on Python 3.12 under Ubuntu. Create and activate a virtual environment using your shell, then run these commands from the repository root. On Linux, use `python3` if needed.

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
