# Marathon recommender: results over training seeds

The full model and the no-race-context ablation were each trained with seeds 42, 7, 123, with the same hyperparameters as the served seed-42 model, and evaluated on the same temporal test split (September to December 2019) by `marathon.train`. Checkpoints were selected on validation NDCG@10 (August). Std is the sample standard deviation over seeds (n - 1). With only 3 seeds it is a rough estimate of training variance; the per-seed columns show the spread directly. The seed-42 paired bootstrap in [MARATHON_RESULTS.md](MARATHON_RESULTS.md) measures variation across runners, which is a different source of uncertainty.

## Do the claims hold for every seed?

| Claim (difference) | Seed 42 | Seed 7 | Seed 123 | Mean ± std | Holds for every seed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ours beats personal popularity, NDCG@10 | +0.0670 | +0.0649 | +0.0651 | +0.0657 ± 0.0012 | yes |
| Ours beats personal popularity, Recall@20 | +0.1318 | +0.1306 | +0.1288 | +0.1304 ± 0.0015 | yes |
| Race context helps in the taper, NDCG@10 | +0.0370 | +0.0304 | +0.0315 | +0.0329 ± 0.0035 | yes |
| Race context helps in the taper, Recall@20 | +0.0600 | +0.0530 | +0.0498 | +0.0543 ± 0.0052 | yes |
| Race context helps overall, Recall@20 | +0.0049 | +0.0040 | +0.0050 | +0.0046 ± 0.0006 | yes |

## Overall test metrics

| Metric | Seed 42 | Seed 7 | Seed 123 | Mean ± std |
| --- | ---: | ---: | ---: | ---: |
| Ours, NDCG@10 | 0.3225 | 0.3204 | 0.3206 | 0.3212 ± 0.0012 |
| Ablation, NDCG@10 | 0.3231 | 0.3236 | 0.3196 | 0.3221 ± 0.0022 |
| Ours, Recall@20 | 0.7953 | 0.7941 | 0.7923 | 0.7939 ± 0.0015 |
| Ablation, Recall@20 | 0.7904 | 0.7902 | 0.7874 | 0.7893 ± 0.0017 |
| Ours - ablation, NDCG@10 | -0.0005 | -0.0032 | +0.0010 | -0.0009 ± 0.0021 |
| Ours - ablation, Recall@20 | +0.0049 | +0.0040 | +0.0050 | +0.0046 ± 0.0006 |

## NDCG@10 by preparation phase

| Metric | Seed 42 | Seed 7 | Seed 123 | Mean ± std |
| --- | ---: | ---: | ---: | ---: |
| Ours, taper (race in 0-3 weeks) | 0.3040 | 0.3037 | 0.3029 | 0.3035 ± 0.0006 |
| Ablation, taper (race in 0-3 weeks) | 0.2670 | 0.2733 | 0.2714 | 0.2706 ± 0.0032 |
| Race-context gain, taper (race in 0-3 weeks) | +0.0370 | +0.0304 | +0.0315 | +0.0329 ± 0.0035 |
| Ours, peak (4-10 weeks out) | 0.2731 | 0.2709 | 0.2697 | 0.2712 ± 0.0017 |
| Ablation, peak (4-10 weeks out) | 0.2741 | 0.2732 | 0.2710 | 0.2728 ± 0.0016 |
| Race-context gain, peak (4-10 weeks out) | -0.0010 | -0.0023 | -0.0013 | -0.0015 ± 0.0007 |
| Ours, post-race recovery (0-8 weeks after) | 0.3380 | 0.3325 | 0.3361 | 0.3356 ± 0.0028 |
| Ablation, post-race recovery (0-8 weeks after) | 0.3386 | 0.3395 | 0.3332 | 0.3371 ± 0.0034 |
| Race-context gain, post-race recovery (0-8 weeks after) | -0.0006 | -0.0070 | +0.0029 | -0.0016 ± 0.0050 |
| Ours, no race context | 0.3429 | 0.3399 | 0.3406 | 0.3411 ± 0.0015 |
| Ablation, no race context | 0.3523 | 0.3509 | 0.3453 | 0.3495 ± 0.0037 |
| Race-context gain, no race context | -0.0094 | -0.0110 | -0.0047 | -0.0084 ± 0.0033 |

Best epoch (selected on validation): full seed 42: 6, seed 7: 6, seed 123: 6; ablation seed 42: 6, seed 7: 6, seed 123: 6.

The served model is the seed-42 checkpoint, `models/marathon/full.pt`. Reproduce with `python -m marathon.seeds train` then `python -m marathon.seeds report`.
