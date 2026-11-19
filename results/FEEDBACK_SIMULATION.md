# Feedback-loop simulation

300 virtual runners, 8 rounds of 3 recommendations, 3 seeds. Comfort shifts: stable habits +0.00, wants lighter runs -0.50. Values are means across seeds of each scenario's last round, with standard deviation in brackets.

## Scenario: stable habits

| Arm | Acceptance | MRR | List quality (NDCG) | Rated OK | Rated too hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| No loop | 0.657 (0.017) | 0.555 (0.021) | 0.757 (0.005) | 0.828 (0.020) | 0.037 (0.003) |
| Short loop | 0.677 (0.012) | 0.626 (0.011) | 0.811 (0.009) | 0.893 (0.018) | 0.023 (0.012) |
| Short + retraining | 0.677 (0.012) | 0.626 (0.011) | 0.811 (0.009) | 0.893 (0.018) | 0.023 (0.012) |

Retraining promoted the candidate in 0 of 3 seeds; the learned effort head was served in 0.

## Scenario: wants lighter runs

| Arm | Acceptance | MRR | List quality (NDCG) | Rated OK | Rated too hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| No loop | 0.534 (0.032) | 0.406 (0.028) | 0.750 (0.005) | 0.581 (0.026) | 0.390 (0.025) |
| Short loop | 0.553 (0.029) | 0.449 (0.027) | 0.769 (0.007) | 0.668 (0.026) | 0.312 (0.018) |
| Short + retraining | 0.557 (0.035) | 0.450 (0.032) | 0.770 (0.009) | 0.676 (0.015) | 0.301 (0.011) |

Retraining promoted the candidate in 2 of 3 seeds; the learned effort head was served in 1.

## Reading this

- *Acceptance* is the share of lists where the runner picked a category; *MRR* also rewards picking it high in the list.
- *List quality* is the NDCG of the runner's hidden appeal over the served list, against the best list allowed by the time budget. It needs no random draw.
- *Rated OK* and *Rated too hard* use the three effort buttons of completed workouts.
- Runners and their reactions are simulated from Endomondo histories, so the GRU saw the same habits during training. The comparison shows whether the loop adapts; it is not a measured gain on real users.

## Reproduce

```text
python -m strideplan.simulate
```
