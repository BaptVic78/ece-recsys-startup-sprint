# StridePlan data pipeline

We prepare each runner's Endomondo history for the workout recommender. The pipeline keeps valid runs, removes duplicates, sorts sessions by date, and exports train/validation/test CSVs.

Each session includes recorded duration, GPS distance (meters), average speed (m/s), average heart rate (bpm), elevation gain (meters), and the recorded gender.

- Distance sums the GPS segments. Average speed is that distance divided by recorded duration.
- Heart rate is the sample average. Elevation gain sums positive altitude changes.
- Missing, invalid, or misaligned sensor series produce empty feature cells. Counts are in `stats.json`.
- GPS noise can affect distance and elevation. Features describe the recorded trace, which may cover only part of a workout.

## Filtering and splits

By default, we keep recorded durations from 5 minutes to 6 hours. These are configurable quality filters, and `stats.json` reports rejection counts and retained duration percentiles.

The latest start-time group goes to test, the previous group to validation, and earlier sessions to training. Evaluation requires at least five distinct training start times. Runners with shorter histories stay in training only. Tied start times stay together.

Fit future model features on training data only to avoid leakage.

## Run it

Python 3.9+ with no extra packages. These commands work in PowerShell and Bash when `python` points to Python 3. On Linux, you can use `python3`.

```text
python -m strideplan_pipeline.download
python -m strideplan_pipeline.pipeline --input data/raw/endomondoHR_proper.json
python -m strideplan_pipeline.validate
python -m unittest discover -s strideplan_pipeline -p "test_*.py"
```

The [official download](https://cseweb.ucsd.edu/~jmcauley/datasets/fitrec.html) is about 4.9 GB. Skip downloading if you already have it. An interrupted download restarts from the beginning.

Optional pipeline flags are `--min-duration 300`, `--max-duration 21600` (seconds), `--min-train-sessions 5`, and `--output data/processed`.

Outputs in `data/processed/` include all sessions, the three split CSVs, `users.csv`, and `stats.json`. Data files are excluded from Git. The validation command checks unique IDs, positive bounded durations, split coverage and chronology, minimum history, and summary counts without rereading the raw download.

## Full run after review

The full 167,783-record release produced 70,559 runs for 807 runners. The splits contain 69,229 training, 665 validation and 665 test sessions. Heart-rate averages are missing for 116 runs. The other added features are present for every retained run. All export checks and seven automated tests pass.
