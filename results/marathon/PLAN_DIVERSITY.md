# Weekly plans: greedy versus sampled

500 weeks from 500 runners (2,012 real sessions): test weeks, one per runner, each a random 7-day window from September-December 2019 with at least three runs and no race. The planner receives the runner's history before the window, the real run dates and the declared race date.

| Method | Distinct types / runs | Consecutive repeats | Weekly km | Longest / shortest | Week with 18+ km run | Same type as real day | Weekly km abs. error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Real weeks (what runners did) | 0.92 | 0.07 | 47.4 | 3.38 | 0.45 | 1.000 | 0.0 |
| Greedy plan (top session each day) | 0.60 | 0.44 | 45.4 | 2.44 | 0.43 | 0.104 | 11.0 |
| Sampled plan (nucleus p=0.9, default) | 0.92 | 0.06 | 47.2 | 2.98 | 0.44 | 0.059 | 11.0 |
| Phase popularity sampled (ignores the runner) | 0.96 | 0.03 | 49.0 | 3.69 | 0.50 | 0.023 | 17.9 |

Jensen-Shannon divergence between the session types used across all generated weeks and those of the real weeks (0 = identical, 1 = disjoint): greedy 0.170, sample 0.014, phase_sample 0.016. Distinct session types used: real 65, greedy 53, sample 65, phase_sample 66.

Greedy decoding returns the single most likely type every day. No type is very likely and the most likely one changes little within a week, so greedy weeks repeat one session. Sampling from the same model reproduces the variety of real weeks but matches the session actually run on fewer days. Sampling the phase popularity, which ignores the runner, also looks varied: variety and type mix measure realism, while per-day agreement and weekly km error show what the runner's history adds. The ranking quality of the model itself is measured in MARATHON_RESULTS.md.

Reproduce: `python -m marathon.plan_eval` (needs the processed data).
