"""Create a readable first-results report and learning curves from saved metrics."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("artifacts/first_results/results.json"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    result = json.loads(args.input.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "first_results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    metrics = result["metrics"]["test"]
    names = {"random": "Random", "global_popularity": "Global popularity",
             "personal_popularity": "Personal popularity", "repeat_last": "Repeat last category",
             "gru": "GRU", "gru_no_history": "GRU with history removed"}
    lines = ["# First results on real Endomondo data", "", "18 November 2026", "",
             "We trained the GRU to predict the category of a runner's next completed session.",
             "This first experiment uses duration and speed categories. It evaluates ranking without a time-budget constraint.", "",
             "## Test results", "",
             "All methods use the same candidate catalogue and the same test cases.", "",
             "| Method | Recall@1 | Recall@3 | NDCG@3 | NDCG@10 | Recall@20 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for method, name in names.items():
        values = metrics[method]
        lines.append(f"| {name} | {values['recall@1']:.1%} | {values['recall@3']:.1%} | {values['ndcg@3']:.4f} | {values['ndcg@10']:.4f} | {values['recall@20']:.1%} |")
    gru, personal, global_pop = metrics["gru"], metrics["personal_popularity"], metrics["global_popularity"]
    count = gru["evaluated_sessions"]
    lines += ["", f"The GRU retrieves the observed category in its top three for {round(gru['recall@3'] * count)} of {count} test sessions. "
              f"Global popularity retrieves {round(global_pop['recall@3'] * count)} and personal popularity retrieves {round(personal['recall@3'] * count)}.", "",
              "History is useful in this experiment. The GRU is slightly ahead of personal popularity, but the difference is small. "
              "This is one run, so we cannot claim a reliable improvement over that baseline yet.", "",
              "The zero-history GRU is a diagnostic: we remove the input history from the same trained network. "
              "It is not retrained for cold start. The planned new-user fallback remains global popularity.", "",
              "## Protocol", "",
              f"- {result['catalogue']['num_templates']} shared categories, built from up to six training-duration quantile bands and four training-speed quantile bands. "
              f"{len(result['training_template_counts'])} categories occur in training.",
              "- Catalogue boundaries, missing-heart-rate imputation and feature scaling are fitted on training data only.",
              f"- Runs with missing, nonpositive or above-{result['config']['max_speed']:g} m/s GPS average speed are excluded. "
              f"This removes {sum(result['quality_exclusions'].values())} sessions across the source splits.",
              f"- {result['examples']['train']:,} training examples, {result['examples']['validation']} validation sessions "
              f"and {count} test sessions from {gru['evaluated_users']} test runners.",
              f"- Up to {result['config']['history']} earlier sessions per example. The input features are log-duration, speed, log-time-gap, "
              "mean heart rate and a missing-heart-rate flag. Altitude is excluded from this first model because of sensor noise.",
              "- A target never appears in its own context. Validation uses earlier training events. Test context may also include earlier validation events. "
              "Other test events are excluded. Splits are per user, not a global calendar cutoff.",
              "- Global popularity counts all training events. Personal popularity counts the same recent history visible to the GRU, "
              "with global popularity used to break ties. Repeat-last puts the previous category first. Random uses a fixed seed.",
              f"- {result['config']['epochs']} epochs, seed {result['config']['seed']}, Adam at 0.001, batch size {result['config']['batch_size']}. "
              f"The checkpoint from epoch {result['best_epoch_by_validation_ndcg3']} was selected by validation NDCG@3. Test scores were not used to select it.",
              "- Metrics are averaged per runner. Each of the current test runners has one target, so Recall@K is also hit rate.",
              f"- With {result['catalogue']['num_templates']} categories, @20 is a broad cutoff. @1 and @3 are more informative for this first catalogue.", "",
              "## Category boundaries", "",
              "Internal duration boundaries, in minutes: " + ", ".join(f"{value / 60:.2f}" for value in result["catalogue"]["duration_edges_seconds"]) + ".", "",
              "Internal speed boundaries, in km/h: " + ", ".join(f"{value * 3.6:.2f}" for value in result["catalogue"]["speed_edges_mps"]) + ".", "",
              f"A value equal to a boundary goes into the higher band. Template ID = 1 + duration-band-index × {len(result['catalogue']['speed_edges_mps']) + 1} + speed-band-index, with zero-based band indices. "
              "These categories are provisional and describe recorded traces, which can be shorter than complete workouts.", "",
              "## Learning curves", "", "![Training and validation curves](first_results_curves.png)", "",
              "## Three examples", "",
              "These are the first three test cases in user-ID order. IDs below are category IDs, not users.", "",
              "| Example | Observed category | Global top 3 | Personal top 3 | GRU top 3 |", "| --- | ---: | --- | --- | --- |"]
    for example in result["qualitative_examples"]:
        top = example["top3"]
        lines.append(f"| {example['example']} | {example['observed_template']} | {top['global_popularity']} | {top['personal_popularity']} | {top['gru']} |")
    lines += ["", "## Limits and next work", "",
              "This predicts a coarse category of an observed next workout. It does not establish which workout improves fitness. "
              "The small GRU-versus-personal-popularity gap needs repeated seeds and uncertainty estimates. "
              "The GPS speed filter does not remove every possible sensor error. The duration and speed groups also need product-level review.", "",
              "The effort head remains untrained and frozen in this run. Effort calibration, conformal risk control, LLM parsing, "
              "time-constrained ranking and serving the trained checkpoint through the API remain pending.", "",
              "## Reproduce", "", "From the repository root, with the processed dataset already present:", "", "```text",
              "python -m pip install -r requirements-results.txt", "python -m strideplan.experiment",
              "python -m strideplan.report_experiment", "```", "",
              "The checkpoint and local experiment outputs are saved under artifacts/first_results, which is excluded from Git. "
              "The aggregate metrics in first_results.json and this report contain no user identifiers. "
              "Use a different --output directory to preserve this run when starting another experiment.", ""]
    (args.output / "FIRST_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = result["training_curve"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    epochs = [record["epoch"] for record in curve]
    axes[0].plot(epochs, [record["train_loss"] for record in curve], marker="o", label="Training")
    axes[0].plot(epochs, [record["validation_loss"] for record in curve], marker="o", label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Ranking loss")
    axes[0].legend(frameon=False)
    axes[1].plot(epochs, [record["validation_ndcg@3"] for record in curve], marker="o", color="#1b7b74")
    axes[1].axvline(result["best_epoch_by_validation_ndcg3"], linestyle="--", color="gray", label="Selected checkpoint")
    axes[1].set(xlabel="Epoch", ylabel="Validation NDCG@3", title="Checkpoint selection")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.set_xticks(epochs)
    figure.savefig(args.output / "first_results_curves.png", dpi=180)
    plt.close(figure)
    print(args.output / "FIRST_RESULTS.md")


if __name__ == "__main__":
    main()
