# Weekly plans: greedy versus sampled

43 weeks from 12 runners (180 real sessions): demo runners (committed sample), weeks starting 28, 21, 14 and 7 days before their race. The planner receives the runner's history before the window, the real run dates and the declared race date.

| Method | Distinct types / runs | Consecutive repeats | Weekly km | Longest / shortest | Week with 18+ km run | Same type as real day | Weekly km abs. error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Real weeks (what runners did) | 0.95 | 0.03 | 50.2 | 3.46 | 0.63 | 1.000 | 0.0 |
| Greedy plan (top session each day) | 0.70 | 0.33 | 51.0 | 3.10 | 0.67 | 0.149 | 8.5 |
| Sampled plan (nucleus p=0.9, default) | 0.93 | 0.08 | 54.5 | 3.63 | 0.67 | 0.061 | 12.0 |
| Phase popularity sampled (ignores the runner) | 0.96 | 0.01 | 56.2 | 3.47 | 0.56 | 0.018 | 16.6 |

Jensen-Shannon divergence between the session types used across all generated weeks and those of the real weeks (0 = identical, 1 = disjoint): greedy 0.208, sample 0.143, phase_sample 0.133. Distinct session types used: real 48, greedy 34, sample 52, phase_sample 53.

Greedy decoding returns the single most likely type every day. No type is very likely and the most likely one changes little within a week, so greedy weeks repeat one session. Sampling from the same model reproduces the variety of real weeks but matches the session actually run on fewer days. Sampling the phase popularity, which ignores the runner, also looks varied: variety and type mix measure realism, while per-day agreement and weekly km error show what the runner's history adds. The ranking quality of the model itself is measured in MARATHON_RESULTS.md.

Reproduce: `python -m marathon.plan_eval --demo`.
