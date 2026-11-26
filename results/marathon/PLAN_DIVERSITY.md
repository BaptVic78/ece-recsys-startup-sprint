# Weekly plans: greedy versus sampled

500 test weeks, one per runner (2,012 real sessions), each a random 7-day window from September-December 2019 with at least three runs and no race. The planner receives the runner's history before the window, the real run dates and the declared race date.

| Method | Distinct types / runs | Consecutive repeats | Weekly km | Longest / shortest | Week with 18+ km run | Same type as real day | Weekly km abs. error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Real weeks (what runners did) | 0.92 | 0.07 | 47.4 | 3.38 | 0.45 | 1.000 | 0.0 |
| Greedy plan (top session each day) | 0.60 | 0.44 | 45.4 | 2.44 | 0.43 | 0.104 | 11.0 |
| Sampled plan (nucleus p=0.9, default) | 0.93 | 0.06 | 46.6 | 3.01 | 0.43 | 0.043 | 11.1 |

Jensen-Shannon divergence between the session types used across all generated weeks and those of the real weeks (0 = identical, 1 = disjoint): greedy 0.170, sample 0.013. Distinct session types used: real 65, greedy 53, sample 66.

Greedy decoding returns the single most likely type every day. No type is very likely and the most likely one changes little within a week, so greedy weeks repeat one session. Sampling from the same model reproduces the variety of real weeks but matches the session actually run on fewer days. Per-day type accuracy is low for both because a runner's exact session on a given day is hard to predict; the ranking quality of the model itself is measured in MARATHON_RESULTS.md.

Reproduce: `python -m marathon.plan_eval` (needs the processed data).

These figures were produced at 5594ff4, before per-day seeding and without the phase-popularity baseline. Rerun the command above to refresh them.
