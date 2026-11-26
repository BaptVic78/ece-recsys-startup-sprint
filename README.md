# StridePlan: a marathon training recommender

[![Tests](https://github.com/BaptVic78/ece-recsys-startup-sprint/actions/workflows/tests.yml/badge.svg)](https://github.com/BaptVic78/ece-recsys-startup-sprint/actions/workflows/tests.yml)

StridePlan recommends a marathon runner's next training sessions. You declare your race date. Every day you plan to run, the app ranks 66 session types for you and turns the best ones into concrete sessions (distance, pace, duration). It also builds the coming week. Log a completed, skipped or missed session and the next recommendations change.

The model is a **context-aware sequential Transformer** after CARCA (Rashed, Elsayed & Schmidt-Thieme, *Context and Attribute-Aware Sequential Recommendation via Cross-Attention*, RecSys 2022). It was trained on 1.2 million sessions of 15,518 runners who completed a 2019 World Marathon Major. The data is the public [BMClab dataset](https://figshare.com/articles/dataset/A_public_dataset_on_long-distance_running_training_in_2019_and_2020/16620238) (CC BY 4.0). The innovation is the **race-calendar context**: the model knows how many weeks remain before your race and how long ago your last race was. On the test period (September to December 2019), the model reaches NDCG@10 0.3225 and Recall@20 0.7953. Most-popular reaches 0.1992 and 0.6240; personal popularity, the strongest baseline, reaches 0.2555 and 0.6636. In the taper weeks, the race context adds +0.037 NDCG@10 over the same model without it, with a paired-bootstrap 95% interval of [+0.035, +0.039]. Moving only the race date makes the recommended distance fall from 14.8 km three weeks out to 7.9 km two days out: the model learned to taper from data.

**Results:** see [results/marathon/MARATHON_RESULTS.md](results/marathon/MARATHON_RESULTS.md). **Method, data and leakage policy:** see [MARATHON_README.md](MARATHON_README.md).

## Quick start (Docker, < 5 minutes)

Install and start [Docker Desktop](https://www.docker.com/products/docker-desktop/), then run:

```text
git clone https://github.com/BaptVic78/ece-recsys-startup-sprint.git
cd ece-recsys-startup-sprint
docker compose up -d --wait
```

Open http://localhost:8501 and load a demo runner in the sidebar.
The API docs are at http://localhost:8000/docs.

Compose downloads the prebuilt image from GHCR. If it is unavailable, it builds
the same image locally. A first source build can take longer, depending on the network.
The trained model and demo runners are included: no dataset download or training is needed.

If a port is busy, copy `.env.example` to `.env` and change `API_PORT` or
`FRONTEND_PORT`. For example, `API_PORT=8010` moves the API to
http://localhost:8010; the frontend still connects to it inside Docker.

- Stop the app: `docker compose down`.
- Start it again: `docker compose up -d --wait`.
- See logs: `docker compose logs --tail=50`.
- Build local changes: `docker compose build api`, then
  `docker compose up -d --no-build --pull never --wait`.

Done, Skip and Missed are stored in the `interactions` Docker volume.
Restarting or recreating the containers keeps this data. `docker compose down -v`
deletes it.

## Team

| Name | Role | Responsibility |
| --- | --- | --- |
| ROULEAU Victor | Project Lead | Plan, schedule, Git repository; owns the evaluation harness and model-specific preprocessing |
| LE POEZELLER Thaddée | Data Engineer | Data pipeline and storage; `/log_interaction` endpoint and database |
| SOUBAGNE Thomas | Lead ML Engineer | Core model implementation |
| GIRAUD Nils | ML Engineer (Ops) | Training and validation loop, model artifacts, predict function |
| ALACACHE Axel | Systems Engineer | FastAPI backend, Streamlit frontend, deployment |

## Run without Docker (Python alternative)

The trained model (`models/marathon/`) and a demo sample of 12 test-period runners (`demo/`) are in the repository, so running the app needs no dataset download.

1. Clone the repository and enter it.

   ```text
   git clone https://github.com/BaptVic78/ece-recsys-startup-sprint.git
   cd ece-recsys-startup-sprint
   ```

2. Create a virtual environment with Python 3.12 or newer, and activate it.

   ```text
   python -m venv .venv
   .venv\Scripts\activate        (Windows)
   source .venv/bin/activate     (macOS / Linux)
   ```

3. Install the dependencies. PyTorch comes first from its CPU index (about 200 MB).

   ```text
   python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
   python -m pip install -r requirements.txt
   ```

4. Start the API in a first terminal.

   ```text
   python -m uvicorn marathon.api:app --host 0.0.0.0 --port 8000
   ```

   The API docs are at http://localhost:8000/docs.

5. Start the frontend in a second terminal, with the virtual environment activated.

   ```text
   python -m streamlit run frontend/app.py
   ```

6. Open http://localhost:8501. In the sidebar, load a demo runner (or create your own profile), then look at today's ranked sessions and the 7-day plan. Click **Done**, **Skip top pick** or **Missed the day** and the recommendations update. Every click is a `POST /log_interaction`, stored in `artifacts/marathon/interactions.sqlite3`.

If port 8000 is already taken (Docker, WSL or VS Code port forwarding often use it), start the API with `--port 8010` and run the frontend with `MARATHON_API=http://localhost:8010` set.

To use the app from another machine on the same Wi-Fi, run step 5 with `MARATHON_API=http://<server-ip>:8000` set, then open `http://<server-ip>:8501`.

## API

| Endpoint | Role |
| --- | --- |
| `GET /health` | Status and model version |
| `GET /catalogue` | The 66 session types |
| `POST /users` | Create a runner: race date, optional goal time, recent sessions |
| `GET /users/{user_id}` | Profile, sessions and interaction log |
| `POST /recommend` | Ranked session types for one day, from the trained model, with their probability, and the session `planned` for that day |
| `POST /plan` | Next days on the chosen weekdays, rolled forward with the model. Each day's `planned` session is drawn from the model's distribution (`strategy: "sample"`, the default, seeded by runner and date, so it matches `/recommend` and stays put while the plan is followed), or set `strategy: "greedy"` for the top session every day |
| `POST /log_interaction` | `completed` (with km, minutes and optional effort), `skipped` or `missed`; written to SQLite |
| `GET /demo/runners`, `POST /demo/{runner}/load` | Demo runners from the test period |

Runners with fewer than 5 logged sessions get the training popularity of their preparation phase (`strategy: "phase_popularity"`), because the model was trained with at least 5 sessions of history. Everyone else gets the model (`strategy: "model"`).

## Retrain the model (optional, about 2 hours on a laptop CPU)

```text
python -m marathon.download          # 155 MB, checked against the published MD5
python -m marathon.data              # sessions, session types, race calendar, temporal split
python -m marathon.validate          # leakage and consistency checks
python -m marathon.train             # full model, saved to models/marathon/full.pt
python -m marathon.train --no-race-context   # ablation
python -m marathon.serving           # cold-start fallback from training data only
python -m marathon.report            # results/marathon/MARATHON_RESULTS.md and learning curves
python -m marathon.demo              # demo sample
```

## Tests

```text
python -m unittest discover -s tests
```

The tests cover leakage (temporal split, histories strictly before targets, a causal pace reference), the ablation being blind to the race calendar, ranking ties, the baselines, and train/serve consistency. They also run the API end to end with SQLite and check latency, and run the Streamlit page against the committed model (`tests/test_frontend.py`).

## Docker and CI checks

Run both test suites inside the image:

```text
docker compose run --rm api python -m unittest discover -s tests
docker compose run --rm api python -m unittest discover -s strideplan_pipeline -p "test_*.py"
```

With the app running, `python scripts/check_docker.py` checks real recommendations,
all three interaction types, a container restart, saved data and the next plan.
It needs only Python's standard library and restarts both services.
For custom ports, pass `--api http://localhost:8010 --frontend http://localhost:8501`.

GitHub Actions runs the tests on each PR and push to `main`.
The image workflow also builds and tests the container on PRs. On `main`, it
publishes the tested image to `ghcr.io/baptvic78/ece-recsys-startup-sprint`,
with `latest` and commit tags.

After the first publication, a package owner must set the GHCR package visibility
to **Public** and check that it can be pulled without a login.
For the CI failure check, temporarily add a failing test on the PR, record the red
run, remove the test and record the green run. Keep no failing test on `main`.

For Week 3 Q6, a teammate who did not write the setup must time clone-to-first-
recommendation on a clean machine and attach the timing and a screenshot to the PR.
Local checks are recorded in [docs/DOCKER_CHECKS.md](docs/DOCKER_CHECKS.md).

## Repository

| Path | Content |
| --- | --- |
| `marathon/` | Data pipeline, model, training, evaluation, serving, API |
| `frontend/app.py` | Streamlit frontend |
| `models/marathon/` | Trained checkpoints and their configurations |
| `demo/` | Demo sample of 12 test-period runners (BMClab, CC BY 4.0) |
| `results/marathon/` | Offline results, learning curves, ablation |
| `tests/` | Unit and end-to-end tests |
| `strideplan/`, `strideplan_pipeline/` | Earlier Endomondo prototype (Weeks 1–3), kept for reference: [first results](results/FIRST_RESULTS.md), [serving](SERVING_README.md), [feedback loop](FEEDBACK_README.md) |

Data: Afonseca, Watanabe & Duarte (2022), *A public dataset on long-distance running training in 2019 and 2020*, figshare, CC BY 4.0, [PeerJ 10:e13192](https://peerj.com/articles/13192).
