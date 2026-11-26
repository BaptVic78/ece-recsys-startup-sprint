# Weeks 1–3 prototype (Endomondo)

Before the marathon reorientation, StridePlan predicted the category of a runner's next workout on the Endomondo / FitRec dataset with a GRU, served it through a first API and added a feedback loop. The code is still in `strideplan/` and `strideplan_pipeline/`, and its tests still run in CI. The current product, which is the marathon recommender, does not use it.

| Document | Content |
| --- | --- |
| [PIPELINE_README.md](PIPELINE_README.md) | Endomondo download, cleaning and chronological splits |
| [WEEK2_README.md](WEEK2_README.md) | Week 2 GRU skeleton, ranking metrics and mock API |
| [SERVING_README.md](SERVING_README.md) | Serving the trained GRU |
| [FEEDBACK_README.md](FEEDBACK_README.md) | Feedback loop, re-ranking and retraining simulation |
| [../../results/legacy/FIRST_RESULTS.md](../../results/legacy/FIRST_RESULTS.md) | First real-data results of the GRU |
