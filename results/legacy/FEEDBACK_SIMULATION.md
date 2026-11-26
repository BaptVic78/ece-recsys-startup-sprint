# Feedback-loop simulation

300 virtual runners, 8 rounds of 3 recommendations, 3 seeds. Comfort shifts: stable habits +0.00, wants lighter runs -1.00. Values are means across seeds of each scenario's last round, with standard deviation in brackets.

## Scenario: stable habits

| Arm | Acceptance | MRR | List quality (NDCG) | Rated OK | Rated too hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| No loop | 0.717 (0.019) | 0.618 (0.023) | 0.813 (0.005) | 0.974 (0.011) | 0.002 (0.002) |
| Short loop | 0.719 (0.004) | 0.661 (0.015) | 0.822 (0.001) | 0.981 (0.010) | 0.002 (0.002) |
| Short + retraining | 0.714 (0.010) | 0.659 (0.019) | 0.821 (0.001) | 0.983 (0.009) | 0.002 (0.002) |

Retraining promoted 2 of 9 candidates; the learned effort head was served in 0.

## Scenario: wants lighter runs

| Arm | Acceptance | MRR | List quality (NDCG) | Rated OK | Rated too hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| No loop | 0.610 (0.024) | 0.481 (0.023) | 0.799 (0.007) | 0.789 (0.032) | 0.205 (0.035) |
| Short loop | 0.598 (0.024) | 0.510 (0.026) | 0.799 (0.004) | 0.839 (0.038) | 0.156 (0.036) |
| Short + retraining | 0.602 (0.036) | 0.521 (0.035) | 0.808 (0.004) | 0.845 (0.043) | 0.150 (0.041) |

Retraining promoted 4 of 9 candidates; the learned effort head was served in 0.

## Reading this

- *Acceptance* is the share of lists where the runner picked a category; *MRR* also rewards picking it high in the list.
- *List quality* is the NDCG of the runner's hidden appeal over the served list, against the best list allowed by the time budget. It needs no random draw.
- *Rated OK* and *Rated too hard* use the three effort buttons of completed workouts.
- Runners and their reactions are simulated from Endomondo histories, so the GRU saw the same habits during training. The comparison shows whether the loop adapts; it is not a measured gain on real users.

## Reproduce

```text
python -m strideplan.simulate
```
