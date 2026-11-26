import copy
import json
import shutil
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from marathon import seeds

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "marathon"
MODELS = ROOT / "models" / "marathon"


def shifted(report, name, delta):
    """A copy of a seed-42 report whose model metrics all move by `delta`."""
    report = copy.deepcopy(report)
    for metrics in [report["test"][name], *report["phases"][name].values()]:
        for key in ("ndcg@10", "recall@20"):
            metrics[key] += delta
    return report


PHASES = {seeds.TAPER: (0.3040, 0.2670), "peak (4-10 weeks out)": (0.2731, 0.2741), seeds.NO_RACE: (0.3429, 0.3523)}


def fake_report(name, ndcg, recall):
    """The parts of a `marathon.train` report that marathon.seeds reads, with the seed-42 values."""
    column = 0 if name == "full" else 1
    report = {"model": name, "best_epoch": 6,
              "test": {name: {"ndcg@10": ndcg, "recall@20": recall}},
              "phases": {name: {phase: {"ndcg@10": values[column], "recall@20": 0.79 + 0.06 * (column == 0)}
                                for phase, values in PHASES.items()}}}
    if name == "full":
        report["test"]["personal_popular"] = {"ndcg@10": 0.2555, "recall@20": 0.6636}
    return report


class SeedsTest(unittest.TestCase):
    def setUp(self):
        # Built here rather than read from results/, which is not part of the Docker image.
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory)
        self.full, self.ablation = fake_report("full", 0.3225, 0.7953), fake_report("no_race_context", 0.3231, 0.7904)
        for seed in (42,):
            self.write(seed, "full", self.full)
            self.write(seed, "no_race_context", self.ablation)

    def write(self, seed, name, report):
        path = seeds.report_path(self.directory, seed, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report), encoding="utf-8")

    def run_report(self, extra):
        seeds.report(Namespace(results=self.directory, seeds=list(extra)))
        return (self.directory / "SEEDS.md").read_text(encoding="utf-8"), \
            json.loads((self.directory / "seeds.json").read_text(encoding="utf-8"))

    def test_train_commands_reuse_served_settings_and_never_touch_the_served_model(self):
        settings = seeds.served_settings(MODELS)
        self.assertEqual(settings["train_sample"], 250_000)  # not train.py's 400,000 default
        command = seeds.train_command(7, "no_race_context", settings, Path("data.parquet"),
                                      Path("artifacts/marathon/seeds"), RESULTS, 4)
        joined = " ".join(command)
        self.assertIn("--seed 7", joined)
        self.assertIn("--train-sample 250000", joined)
        self.assertIn("--no-race-context", command)
        output = Path(command[command.index("--output") + 1])
        self.assertNotEqual(output.resolve(), MODELS.resolve())
        self.assertEqual(Path(command[command.index("--report") + 1]), RESULTS / "seeds" / "seed_7")
        with self.assertRaises(ValueError):
            seeds.train_command(42, "full", settings, Path("d"), Path("c"), RESULTS, 4)

    def test_seed_metrics_compute_the_published_differences(self):
        row = seeds.seed_metrics(self.full, self.ablation)
        self.assertAlmostEqual(row["ours ndcg@10"], 0.3225, places=4)
        self.assertAlmostEqual(row["ours - personal popularity, ndcg@10"], 0.0670, places=4)
        self.assertAlmostEqual(row[f"race-context gain ndcg@10, {seeds.TAPER}"], 0.0370, places=4)

    @unittest.skipUnless((RESULTS / "full.json").exists(), "results/ is not in the Docker image")
    def test_committed_seed_42_reports_give_the_published_numbers(self):
        full, ablation = (json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8")) for name in seeds.MODELS)
        row = seeds.seed_metrics(full, ablation)
        self.assertAlmostEqual(row["ours - personal popularity, ndcg@10"], 0.0670, places=4)
        self.assertAlmostEqual(row[f"race-context gain ndcg@10, {seeds.TAPER}"], 0.0370, places=4)

    def test_report_gives_mean_and_std_over_three_seeds(self):
        for seed, delta in ((7, 0.002), (123, -0.002)):
            self.write(seed, "full", shifted(self.full, "full", delta))
            self.write(seed, "no_race_context", self.ablation)
        text, data = self.run_report([7, 123])
        self.assertEqual(list(data["per_seed"]), ["42", "7", "123"])
        ours = data["summary"]["ours ndcg@10"]
        self.assertAlmostEqual(ours["mean"], self.full["test"]["full"]["ndcg@10"])
        self.assertAlmostEqual(ours["std"], 0.002)
        self.assertIn("Seed 42 | Seed 7 | Seed 123 | Mean ± std", text)
        self.assertNotIn("**no", text.split("## Overall")[0].replace("Ours beats personal", ""))

    def test_a_claim_that_fails_for_one_seed_is_reported(self):
        self.write(7, "full", self.full)
        self.write(7, "no_race_context", self.ablation)
        # Seed 123: the ablation catches up in the taper, so the race-context gain disappears.
        ablation = copy.deepcopy(self.ablation)
        ablation["phases"]["no_race_context"][seeds.TAPER]["ndcg@10"] = 0.31
        self.write(123, "full", self.full)
        self.write(123, "no_race_context", ablation)
        text, _ = self.run_report([7, 123])
        taper = [line for line in text.splitlines() if line.startswith("| Race context helps in the taper, NDCG@10")][0]
        self.assertIn("**no: 2 of 3 seeds**", taper)

    def test_missing_reports_stop_with_the_command_to_run(self):
        with self.assertRaises(SystemExit) as stop:
            self.run_report([7])
        self.assertIn("marathon.seeds train", str(stop.exception))


if __name__ == "__main__":
    unittest.main()
