# Marathon plan recommender

StridePlan is being reoriented from next-workout prediction to marathon training plans. A runner gives a race date and a goal time. The system estimates their chance of reaching it and recommends the coming weeks from the training of similar runners who reached a comparable goal.

The approach follows the case-based recommender of [Feely et al., RecSys 2020](https://dl.acm.org/doi/fullHtml/10.1145/3383313.3412220), rebuilt on public data.

## Data

[BMClab long-distance running dataset](https://figshare.com/articles/dataset/A_public_dataset_on_long-distance_running_training_in_2019_and_2020/16620238) (Afonseca, Watanabe & Duarte, [PeerJ 2022](https://peerj.com/articles/13192)), CC BY 4.0. It has 36,412 Strava runners who had run a World Marathon Major, with one row per runner per day of 2019: distance (km), duration (min), gender, age group, country, and the majors they ran (`major`, e.g. `"LONDON 2018,LONDON 2019"`).

- **Label.** A case is a runner's 2019 major. Its label is the duration of a 41–47 km day on the race date, between 2 and 7 hours. The daily total can include a warm-up or a walk back.
- **History.** Daily distance and duration for up to 140 days strictly before race day. The race day is never in the history. Tokyo (3 March) has 8 weeks of 2019 history and Boston (15 April) 14, because the file starts on 1 January.
- **Cleaning.** Days with a negative or non-finite value, time without distance, or an average pace outside 2.5–15 min/km are set to zero. `stats.json` counts them.
- **Splits.** 70/15/15 by athlete with a fixed seed. A runner with two majors keeps both races in one split.
- **Limits.** Only daily totals: no heart rate, no splits, so interval structure is not visible. The population is Strava users who completed a major, so beginners are under-represented.

## Run it

```text
python -m marathon.download
python -m marathon.data
python -m marathon.validate
python -m unittest tests.test_marathon_data
```

The download is 155 MB and is checked against the published MD5. Outputs go to `data/processed/marathon/`, which is excluded from Git: `races.csv` (one row per labelled race), `days.parquet` (daily history), `weeks.csv` (complete weeks before the race) and `stats.json`.

## Result on the full file

| Race | Registered in `major` | Kept |
| --- | ---: | ---: |
| Tokyo | 1,170 | 1,128 |
| Boston | 4,365 | 4,299 |
| London | 2,404 | 2,359 |
| Berlin | 4,545 | 4,478 |
| Chicago | 1,253 | 1,173 |
| New York | 2,751 | 2,684 |

16,121 labelled races from 15,518 runners: 11,296 train, 2,411 validation and 2,414 test races. 10,694 races have 16 complete weeks of history. Finish times: median 3h48, 5th percentile 2h49, 95th percentile 5h34. Registered runners without a marathon-length day on race day (about 2%) are excluded and counted.
