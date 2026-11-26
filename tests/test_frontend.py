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
        self.assertEqual(minutes.value, float(top["recommendations"][0]["minutes"]))
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


if __name__ == "__main__":
    unittest.main()
