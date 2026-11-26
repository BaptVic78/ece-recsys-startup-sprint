import os
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

os.environ["MARATHON_SKIP_APP"] = "1"

from fastapi.testclient import TestClient  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from marathon.api import MAX_SESSION_KM, MAX_SESSION_MINUTES, create_app  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "frontend" / "app.py"


class FrontendTest(unittest.TestCase):
    """Runs the Streamlit page against the real committed model, with HTTP routed to the API in memory."""

    def setUp(self):
        self.client = TestClient(create_app(ROOT / "models" / "marathon", ":memory:", ROOT / "demo"))

        def route(method, url, timeout=None, **kwargs):
            path = url.split("://", 1)[1].split("/", 1)[1]
            return self.client.request(method, "/" + path, **kwargs)

        self.patch = mock.patch("requests.request", side_effect=route)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def page(self, user_id, today):
        app = AppTest.from_file(str(APP), default_timeout=60)
        app.session_state["user_id"] = user_id
        app.session_state["today"] = today
        return app.run()

    def test_slow_runner_recommendation_fits_the_completion_widget(self):
        # Reviewer's case: six 35 km runs at 14 min/km, valid under the training pace rule.
        runs = [{"date": str(date(2019, 8, 20) + timedelta(days=2 * i)), "km": 35.0, "minutes": 490.0} for i in range(6)]
        created = self.client.post("/users", json={"user_id": "slow", "race_date": "2019-09-29", "sessions": runs})
        self.assertEqual(created.status_code, 200)
        app = self.page("slow", date(2019, 9, 1))
        self.assertFalse(app.exception, app.exception)
        top = self.client.post("/recommend", json={"user_id": "slow", "date": "2019-09-01", "top_k": 1}).json()
        [button for button in app.button if button.label == "Log run"][0].click().run()
        self.assertFalse(app.exception, app.exception)
        minutes = [widget for widget in app.number_input if widget.label == "Time (min)"][0]
        self.assertEqual(minutes.value, float(top["planned"]["minutes"]))
        self.assertGreater(minutes.value, 400)  # above the old hard-coded maximum
        self.assertEqual(minutes.proto.max, MAX_SESSION_MINUTES)
        self.assertEqual([w for w in app.number_input if w.label == "Distance (km)"][0].proto.max, MAX_SESSION_KM)

    def test_demo_runner_page_renders_and_logs_a_missed_day(self):
        loaded = self.client.post("/demo/4399/load").json()
        app = self.page(loaded["user_id"], date.fromisoformat(loaded["today"]))
        self.assertFalse(app.exception, app.exception)
        self.assertEqual([tab.label for tab in app.tabs], ["Today", "Your week", "History"])
        self.assertFalse(app.number_input)
        [button for button in app.button if "Missed" in button.label][0].click().run()
        self.assertFalse(app.exception, app.exception)
        interactions = self.client.get(f"/users/{loaded['user_id']}").json()["interactions"]
        self.assertEqual([row["type"] for row in interactions], ["missed"])

    def test_completed_run_saves_actual_values_and_moves_to_next_day(self):
        loaded = self.client.post("/demo/4399/load").json()
        today = date.fromisoformat(loaded["today"])
        app = self.page(loaded["user_id"], today)
        [button for button in app.button if button.label == "Log run"][0].click().run()
        [w for w in app.number_input if w.label == "Distance (km)"][0].set_value(8.0)
        [w for w in app.number_input if w.label == "Time (min)"][0].set_value(48.0)
        [button for button in app.button if button.label == "Save run"][0].click().run()
        self.assertFalse(app.exception, app.exception)
        profile = self.client.get(f"/users/{loaded['user_id']}").json()
        self.assertEqual(profile["interactions"][-1]["type"], "completed")
        self.assertEqual(profile["sessions"][-1]["km"], 8.0)
        self.assertEqual(profile["sessions"][-1]["minutes"], 48.0)
        self.assertEqual(app.session_state["today"], today + timedelta(days=1))
        self.assertFalse(app.number_input)

    def test_today_card_matches_the_week_row_for_the_same_day(self):
        for runner in (4302, 4399, 18512):
            with self.subTest(runner=runner):
                loaded = self.client.post(f"/demo/{runner}/load").json()
                today = date.fromisoformat(loaded["today"])
                app = self.page(loaded["user_id"], today)
                self.assertFalse(app.exception, app.exception)
                week = app.dataframe[0].value
                self.assertEqual(week.iloc[0]["Day"], today.strftime("%a %d %b"))  # demo days are Sundays
                card = [element.value for element in app.subheader if " km · " in element.value][0]
                self.assertEqual(week.iloc[0]["Session"], card)

    def test_titles_name_distance_and_pace_and_options_are_visible(self):
        loaded = self.client.post("/demo/4302/load").json()
        today = date.fromisoformat(loaded["today"])
        day = self.client.post("/recommend", json={"user_id": loaded["user_id"], "date": str(today),
                                                   "top_k": 3}).json()
        app = self.page(loaded["user_id"], today)
        self.assertNotIn("Other suggestions", [expander.label for expander in app.expander])
        markdown = [element.value for element in app.markdown]
        self.assertIn("**Other options the model ranks highly**", markdown)
        card = [element.value for element in app.subheader if " km · " in element.value][0]
        options = [value.strip("*") for value in markdown if value.startswith("**") and " km · " in value]
        others = [item for item in day["recommendations"] if item["item"] != day["planned"]["item"]][:2]
        self.assertEqual(len(options), len(others))
        self.assertEqual(len({card, *options}), 1 + len(others))  # different types, different titles
        self.assertTrue(card.startswith(f"{day['planned']['km']:g} km"))
        week = app.dataframe[0].value
        self.assertEqual(list(week.columns), ["Day", "Session", "Pace", "Time", "Model rank", "Top-ranked options"])
        self.assertTrue(all(cell.startswith("#") for cell in week["Model rank"]))

    def test_today_card_says_which_past_runs_drove_the_session(self):
        loaded = self.client.post("/demo/4302/load").json()
        today = date.fromisoformat(loaded["today"])
        planned = self.client.post("/recommend", json={"user_id": loaded["user_id"], "date": str(today),
                                                       "top_k": 3}).json()["planned"]
        app = self.page(loaded["user_id"], today)
        self.assertFalse(app.exception, app.exception)
        lines = [element.value for element in app.markdown if element.value.startswith("Because of: ")]
        self.assertEqual(len(lines), 1)
        expected = ", ".join(f"{date.fromisoformat(run['date']):%a %d %b} {run['km']:g} km"
                             for run in planned["because_of"])
        self.assertEqual(lines[0], "Because of: " + expected)

    def test_skip_logs_the_session_shown_and_moves_to_next_day(self):
        loaded = self.client.post("/demo/4302/load").json()
        today = date.fromisoformat(loaded["today"])
        shown = self.client.post("/recommend", json={"user_id": loaded["user_id"], "date": str(today),
                                                     "top_k": 3}).json()["planned"]
        app = self.page(loaded["user_id"], today)
        [button for button in app.button if button.label.startswith("Skip this session")][0].click().run()
        self.assertFalse(app.exception, app.exception)
        interactions = self.client.get(f"/users/{loaded['user_id']}").json()["interactions"]
        self.assertEqual([(row["type"], row["date"], row["item"]) for row in interactions],
                         [("skipped", str(today), shown["item"])])
        self.assertEqual(app.session_state["today"], today + timedelta(days=1))


if __name__ == "__main__":
    unittest.main()
