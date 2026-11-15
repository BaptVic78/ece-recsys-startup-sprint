# StridePlan data pipeline

Python 3.10+ and the standard library are sufficient. Run commands from the repository root.

## Download and prepare

The official [UCSD FitRec release](https://cseweb.ucsd.edu/~jmcauley/datasets/fitrec.html)
contains Endomondo workout logs. The filtered `endomondoHR_proper.json` download
is approximately 4.9 GB. Its counts differ from the original 253,020-workout collection.

```bash
mkdir -p data/raw
curl --fail --location --retry 3 --continue-at - \
  --output data/raw/endomondoHR_proper.json.part \
  https://mcauleylab.ucsd.edu/public_datasets/gdrive/fitrec/endomondoHR_proper.json
mv data/raw/endomondoHR_proper.json.part data/raw/endomondoHR_proper.json
python3 -m strideplan_pipeline.pipeline --input data/raw/endomondoHR_proper.json
python3 -m unittest discover -s strideplan_pipeline -p 'test_*.py'
```

Only rename the partial download after curl succeeds. Raw and generated data are ignored by Git.
The loader streams JSON lines or Python dictionary literals using `ast.literal_eval`.
It also accepts gzip files. Only small normalized session records remain in memory.

## Validation and exports

Only `sport == "run"` is retained. Each running session requires user and workout
identifiers and at least two finite, positive, nondecreasing Unix timestamps.
The duration is the final timestamp minus the first, in seconds. This measures
the recorded interval, which can differ from the complete workout duration if
the source recording was truncated. Normalized `time_elapsed` values are not used.
Zero-duration sessions and repeated workout IDs are discarded. Rejection counts
are included in `stats.json`.

Outputs in `data/processed/` are `sessions.csv`, `train.csv`, `validation.csv`,
`test.csv`, `users.csv`, and `stats.json`. Override the directory with `--output`.
Session rows include user, workout, sport, start timestamp, duration, and split.
User statistics include retained sessions and eligibility for evaluation.

Sessions are ordered per user. With at least three distinct start timestamps,
the latest timestamp group goes to test, the preceding group to validation,
and all earlier groups to training. Tied starts remain together. Users with fewer
than three distinct starts contribute only to training and are excluded from
held-out evaluation. The split is per user, not a global calendar cutoff.

Build future learned features, categories, popularity scores and normalization
statistics from training only. Validation can select hyperparameters. When
evaluating the test session, its chronological context may include validation
sessions, but must exclude the test group and future events. These exports
prepare histories and splits. A recommendation model and ranking evaluation
remain subsequent work.
