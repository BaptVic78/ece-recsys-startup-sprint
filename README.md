# RecSys Startup Sprint

Build and demo an MVP based on a technique from a RecSys research paper. Select the paper and dataset before implementing the model.

Plan for offline ranking evaluation, interaction logging, model serving and a four-week demo milestone.

## StridePlan data preparation

See [the pipeline guide](PIPELINE_README.md) for download commands, validation rules, chronological splits, and exports.

## Week 2 model, evaluation and API

See [the Week 2 guide](WEEK2_README.md) to run the model smoke check, ranking metrics and mock API.

The model/API dependency set requires Python 3.12+. The standalone data pipeline supports Python 3.9+.

## First results on real data

See [the first-results report](results/FIRST_RESULTS.md) for the trained GRU, baseline comparisons, learning curves and reproduction commands. The first experiment predicts workout categories without time constraints.

## Serving the trained model

See [the serving guide](SERVING_README.md). The API serves the trained GRU from its saved bundle, filters categories by time budget and falls back to global popularity for new users.
