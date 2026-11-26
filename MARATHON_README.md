# Marathon training recommender

StridePlan recommends a marathon runner's next training sessions. The runner declares a race and its date. The app ranks 66 session types for their next run and builds the coming week from those rankings. The plan updates after every logged or missed session.

## Approach

The core model is a context-aware sequential Transformer, after **CARCA** (Rashed, Elsayed & Schmidt-Thieme, "Context and Attribute-Aware Sequential Recommendation via Cross-Attention", RecSys 2022). Self-attention encodes the runner's recent sessions with their attributes and context. Cross-attention then scores every candidate session type against that history, given the context of the day being planned.

The innovation is the **race-calendar context**: days until the declared race, together with the weekday and the days since the last run. It tells the model where the runner is in their preparation (build-up, peak, taper or recovery). It is measured with its own metric, ranking quality by preparation phase, and an ablation: the same model retrained without that context. Neither k-NN nor matrix factorisation is used.

Evaluation uses full-catalogue ranking with NDCG@10 and Recall@20 on a temporal split. Random and most-popular rankings form the sanity floor.

## Data

[BMClab long-distance running dataset](https://figshare.com/articles/dataset/A_public_dataset_on_long-distance_running_training_in_2019_and_2020/16620238) (Afonseca, Watanabe & Duarte, [PeerJ 2022](https://peerj.com/articles/13192)), CC BY 4.0. It covers 36,412 Strava runners who had run a World Marathon Major, with one row per runner per day of 2019: distance (km), duration (min), gender, age group, country and the majors they ran (`major`, e.g. `"LONDON 2018,LONDON 2019"`).

- **Users.** Runners with a labelled 2019 major: a 41–47 km day on the race date, finished in 2 to 7 hours. Registered runners without such a day are excluded and counted.
- **Interactions.** Every running day of at least 1 km. The source stores daily totals, so two runs on one day form one session.
- **Session types (items).** 11 distance bands × 6 intensity bands = 66 types.
  - Distance bands, in km: 1–5, 5–8, 8–10, 10–12, 12–15, 15–18, 18–21, 21–25, 25–30, 30–35, 35+.
  - Intensity is the session pace divided by the runner's median pace over the 28 days strictly before it. The bands are below 0.90, 0.90–0.95, 0.95–0.99, 0.99–1.03, 1.03–1.08 and above 1.08.
  - With fewer than three earlier runs in that window, intensity is set to "usual". This affects 3.7% of sessions.
  - The edges are fixed in advance, not fitted on the data.
- **Race calendar.** Each session gets the days until the runner's next race. Race-day sessions stay in the history but are never recommendation targets.
- **Cleaning.** Days with a negative or non-finite value, time without distance, or an average pace outside 2.5–15 min/km are set to zero.

### Leakage policy

- **Temporal split.** One global cutoff: train before 1 August 2019, validation in August, test from 1 September (the autumn majors).
- **Context.** Every input is either known in advance or computed from strictly earlier days. The race date is declared by the runner at sign-up.
- **Race results.** The finish time is never a model input. It would leak, because the autumn results fall in the test period.
- **Checks.** `marathon.validate` checks the split labels, chronological order, the race calendar and the counts.

## Run it

```text
python -m marathon.download
python -m marathon.data
python -m marathon.validate
python -m unittest tests.test_marathon_data
```

The download is 155 MB and is checked against the published MD5. Outputs go to `data/processed/marathon/`, which is excluded from Git: `races.csv`, `sessions.parquet` and `stats.json`.

## Dataset on the full file

| Race | Registered in `major` | Kept |
| --- | ---: | ---: |
| Tokyo | 1,170 | 1,128 |
| Boston | 4,365 | 4,299 |
| London | 2,404 | 2,359 |
| Berlin | 4,545 | 4,478 |
| Chicago | 1,253 | 1,173 |
| New York | 2,751 | 2,684 |

- **Sessions:** 15,518 users and 2,207,158 sessions, including 16,121 race days.
- **Recommendation targets (race days excluded):** 1,282,554 train, 213,107 validation and 695,376 test.
- **Catalogue:** all 66 session types occur in training. The most frequent covers 5.3% of training sessions and the 20 most frequent cover 62%.
- **Test period:** 21.6% of test sessions fall within 20 weeks of a known race.

## Model, training and results

`marathon/model.py` is the CARCA-style recommender. Each past session is encoded as the sum of:
- its session-type embedding;
- a projection of its attributes (log distance, intensity, gap since the previous run);
- embeddings of its weekday, its race-calendar bucket (weeks to the race) and its recovery bucket (weeks since the last race).

Two self-attention layers encode the last 32 sessions. Every candidate type then becomes a query: its embedding, its static attributes and the context of the planned day. Cross-attention scores all 66 candidates at once, and the model is trained with softmax cross-entropy. `--no-race-context` removes the race and recovery embeddings and changes nothing else. That is the ablation.

**Training:** 6 epochs of 250,000 training targets sampled per epoch, AdamW at 1e-3, batch 1,024, seed 42. The checkpoint is chosen on validation NDCG@10 (August). Both models are 133k parameters and live in `models/marathon/`.

**Full report:** [results/marathon/MARATHON_RESULTS.md](results/marathon/MARATHON_RESULTS.md). On the test period (September to December 2019: 694,116 targets, 15,097 runners, full ranking of 66 types):

| Method | NDCG@10 | Recall@20 |
| --- | ---: | ---: |
| Ours | **0.3225** | **0.7953** |
| Ablation without race context | 0.3231 | 0.7904 |
| Personal popularity | 0.2555 | 0.6636 |
| Most popular | 0.1992 | 0.6240 |
| Random | 0.0688 | 0.3040 |

- **Sanity floor.** The model beats random and most-popular by a wide margin. It also beats personal popularity, the strongest baseline: NDCG@10 is +0.067, 95% interval [+0.066, +0.068].
- **Not just popularity.** All 66 types appear in some top-10 list, against 15% of types for most-popular.
- **Innovation.** The race context matters where it should. In the taper (race within 3 weeks), NDCG@10 rises from 0.267 to 0.304, +0.037 with 95% interval [+0.035, +0.039], and Recall@20 gains +0.060. It is neutral in peak and recovery weeks. It costs 0.009 NDCG@10 when no race is planned, so the overall NDCG@10 is unchanged; overall Recall@20 gains +0.005.
- **Taper check.** We take 2,000 distinct test runners, one session each, keep their histories and move only the race date. The expected recommended distance peaks at 14.9 km three weeks out, then falls to 7.7 km two days before the race. The ablation stays flat at 13.6 km. No rule encodes the taper: the model learned it from data.
- **Learning curves.** Training and validation losses decrease together, with no divergence. Validation NDCG@10 was still rising slowly at epoch 6, so more epochs might help both models.

Reproduce with `python -m marathon.train`, `python -m marathon.train --no-race-context` and `python -m marathon.report`. Allow about 1 hour per model on an 8-core laptop when nothing else is running.

## Weekly plans that vary like real training

`POST /plan` rolls the model forward over the chosen run days: each planned session is treated as done before ranking the next day. The first version always took the top-ranked type (greedy decoding), and weeks repeated one session at the same pace.

**Why greedy weeks repeat.** The model is uncertain about any single day: on the demo runners, the top type has a median probability of 9% out of 66, about one point ahead of the second. The most likely type depends mostly on the runner's history, which changes little within a week, so taking it every day repeats it. Feeding the planned session back is not the main cause: ranking each day from the real history only, with nothing appended, still repeats the top type on 33% of consecutive days, against 39% with the rollout. The model predicts the next session a runner like this is likely to do; it is not trained to balance a week.

**What the plan does now.** Each day's session is **drawn from the model's own distribution**, restricted to the types covering 90% of the probability (nucleus sampling, `marathon/serving.py`). Each day has its own generator, seeded by the runner and that day's date, so:
- the same day shows the same session in `POST /recommend` (the Today view) and in `POST /plan` (the week);
- a plan stays the same while the runner follows it; only a different logged session changes the days after it;
- every response gives the planned session's rank and probability, so a reader sees how likely the model found it.

`strategy: "greedy"` is still available on both endpoints.

**What sampling gains and costs.** `python -m marathon.plan_eval` compares generated weeks with what 500 test runners actually ran, filling their real run dates from their real history ([results/marathon/PLAN_DIVERSITY.md](results/marathon/PLAN_DIVERSITY.md)):

| Weeks | Distinct types / runs | Consecutive repeats | Weekly km | JS divergence of session types vs real | Same type as real day |
| --- | ---: | ---: | ---: | ---: | ---: |
| Real | 0.92 | 0.07 | 47.4 | — | 1.000 |
| Greedy | 0.60 | 0.44 | 45.4 | 0.170 | 0.104 |
| Sampled | 0.93 | 0.06 | 46.6 | 0.013 | 0.043 |

- Sampled weeks have the variety, the long/short spread and the overall mix of real training weeks.
- Greedy matches the session actually run on more days (10.4% against 4.3%), because it always bets on the most likely type.
- Variety and mix measure realism, not personalisation: sampling the phase-popularity fallback, which ignores the runner, matches them too. On the 12 committed demo runners (43 real weeks, `python -m marathon.plan_eval --demo`, [results](results/marathon/PLAN_DIVERSITY_DEMO.md)), it reached 0.96 distinct types per run and 1% repeats, like the model's sampled weeks (0.93 and 8%), but matched the real session on 1.8% of days and missed the weekly volume by 16.6 km, against 6.1% and 12.0 km for sampled weeks and 14.9% and 8.5 km for greedy. The runner's history shows in the per-day match and the weekly volume. This sample is small: treat the demo figures as indicative.

The full-test-set numbers in the table were produced before per-day seeding and without the baseline; rerun `python -m marathon.plan_eval` with the processed data to refresh them and add the baseline. The model's ranking quality is unchanged and is reported in MARATHON_RESULTS.md.
