# StridePlan data pipeline

This pipeline prepares Endomondo running histories for our workout recommender.
It keeps valid runs, removes duplicates, calculates recorded duration, and sorts each runner's sessions by date.

The latest session goes to test, the previous one to validation, and the rest to training.
Sessions with the same start time stay together. Runners with fewer than three distinct start times contribute only to training.

## Results

The full download was processed on November 15, 2026.

| Output | Count |
| --- | ---: |
| Records read | 167,783 |
| Valid running sessions | 70,559 |
| Runners | 807 |
| Training sessions | 69,095 |
| Validation sessions | 732 |
| Test sessions | 732 |

We excluded 97,192 non-running records and 32 runs with unordered timestamps.
Both automated tests passed. Checks on the full output confirmed unique sessions, positive durations, and chronological splits with no overlap.

## Run it

Requires Python 3.10+ with no extra packages. From the repository root, download the [official FitRec dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/fitrec.html) (about 4.9 GB), then prepare it.

```bash
mkdir -p data/raw
curl --fail --location --retry 3 --continue-at - \
  --output data/raw/endomondoHR_proper.json.part \
  https://mcauleylab.ucsd.edu/public_datasets/gdrive/fitrec/endomondoHR_proper.json && \
mv data/raw/endomondoHR_proper.json.part data/raw/endomondoHR_proper.json
python3 -m strideplan_pipeline.pipeline --input data/raw/endomondoHR_proper.json
python3 -m unittest discover -s strideplan_pipeline -p 'test_*.py'
```

Outputs are saved in `data/processed/`: all sessions, the three split CSVs, `users.csv`, and `stats.json`. Data files are excluded from Git.

Duration measures the recorded trace, which may cover only part of a workout. Future model features must be fitted on training data only to avoid data leakage.
