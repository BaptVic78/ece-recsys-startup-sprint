"""Export a small demo sample: test-period runners and their sessions, committed to the repository.

The demo needs no 155 MB download. Runners are taken from the autumn 2019 majors (test period),
so the demo shows recommendations for preparations the model never trained on. Each runner's
demo "today" is 28 days before the race; sessions after it are kept to compare with what the
runner actually did. Source: BMClab dataset, CC BY 4.0 (Afonseca, Watanabe & Duarte, 2022).
"""

import argparse
from pathlib import Path

import pandas as pd

DEMO_RACES = ("BERLIN 2019", "CHICAGO 2019", "NEW YORK 2019")


def export(data_dir, output, per_race=4, seed=7, days_before=28):
    races = pd.read_csv(data_dir / "races.csv", parse_dates=["race_date"])
    sessions = pd.read_parquet(data_dir / "sessions.parquet", columns=["athlete", "date", "km", "minutes"])
    counts = sessions.groupby("athlete").size()
    # Runners with one 2019 race only and a regular history (at least 120 sessions in 2019).
    single = races.groupby("athlete").filter(lambda group: len(group) == 1)
    eligible = single[single["race"].isin(DEMO_RACES) & single["athlete"].map(counts).ge(120)]
    chosen = pd.concat([group.sample(min(per_race, len(group)), random_state=seed)
                        for _, group in eligible.groupby("race")])
    runners = pd.DataFrame({"runner": chosen["athlete"].astype(int), "race": chosen["race"],
                            "race_date": chosen["race_date"].dt.date.astype(str),
                            "today": (chosen["race_date"] - pd.Timedelta(days=days_before)).dt.date.astype(str),
                            "finish_minutes": chosen["finish_minutes"].round(1)})
    selected = sessions[sessions["athlete"].isin(runners["runner"])].merge(
        chosen[["athlete", "race_date"]], on="athlete")
    selected = selected[selected["date"] <= selected["race_date"]]
    demo_sessions = pd.DataFrame({"runner": selected["athlete"].astype(int), "date": selected["date"].dt.date.astype(str),
                                  "km": selected["km"].round(2), "minutes": selected["minutes"].round(2)})
    output.mkdir(parents=True, exist_ok=True)
    runners.sort_values("runner").to_csv(output / "runners.csv", index=False)
    demo_sessions.sort_values(["runner", "date"]).to_csv(output / "sessions.csv", index=False)
    print(f"{len(runners)} runners, {len(demo_sessions)} sessions written to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/marathon"))
    parser.add_argument("--output", type=Path, default=Path("demo"))
    args = parser.parse_args()
    export(args.data, args.output)


if __name__ == "__main__":
    main()
