import json
import os
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

os.environ["MARATHON_SKIP_APP"] = "1"

from fastapi.testclient import TestClient  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from marathon.api import create_app  # noqa: E402
from marathon.data import plausible_pace  # noqa: E402
from marathon.forecast import features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROFILES = json.loads((ROOT / "demo" / "mock_profiles.json").read_text(encoding="utf-8"))["profiles"]


class OnboardingTest(unittest.TestCase):
    """Sign-up, the mock running-app import, the dashboard and the staff page, against the committed models."""

    def setUp(self):
        self.client = TestClient(create_app(ROOT / "models" / "marathon", ":memory:", ROOT / "demo"))

        def route(method, url, timeout=None, **kwargs):
            path = url.split("://", 1)[1].split("/", 1)[1]
            return self.client.request(method, "/" + path, **kwargs)

        self.patch = mock.patch("requests.request", side_effect=route)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def test_mock_profiles_are_valid_marathon_histories(self):
        self.assertEqual(len(PROFILES), 5)
        self.assertEqual(len({p["source"] for p in PROFILES}), 5)
        for profile in PROFILES:
            with self.subTest(profile=profile["id"]):
                sessions = profile["sessions"]
                self.assertTrue(20 <= len(sessions) <= 400)
                self.assertTrue(all(1 <= s["days_ago"] <= 84 for s in sessions))
                self.assertTrue(all(plausible_pace(s["km"], s["minutes"]) for s in sessions))
                self.assertGreaterEqual(max(s["km"] for s in sessions), 25)  # a real long run

    def test_signup_import_and_dashboard(self):
        app = AppTest.from_file(str(ROOT / "frontend" / "app.py"), default_timeout=60).run()
        self.assertFalse(app.exception, app.exception)
        app.text_input[0].set_value("Camille Martin")
        [b for b in app.button if b.label == "Create my account"][0].click().run()
        self.assertFalse(app.exception, app.exception)
        app.selectbox[0].set_value(app.selectbox[0].options[1])  # presenter picks the first profile
        [b for b in app.button if b.label == "Connect Strava"][0].click().run()
        self.assertFalse(app.exception, app.exception)
        user_id = app.session_state["user_id"]
        self.assertTrue(user_id.startswith("camille-martin-"))
        profile = self.client.get(f"/users/{user_id}").json()
        self.assertEqual(len(profile["sessions"]), PROFILES[0]["runs"])
        self.assertTrue(all(s["date"] < str(date.today()) for s in profile["sessions"]))
        self.assertEqual([tab.label for tab in app.tabs], ["Today", "Your week", "History"])
        labels = [metric.label for metric in app.metric]
        self.assertIn("Forecast finish", labels)
        forecast = [m for m in app.metric if m.label == "Forecast finish"][0].value
        self.assertRegex(forecast, r"^\d:\d\d$")
        self.assertTrue(any(" km · " in element.value for element in app.subheader))

    def test_staff_page_explains_a_runner(self):
        loaded = self.client.post("/demo/4399/load").json()
        app = AppTest.from_file(str(ROOT / "frontend" / "pages" / "1_Staff.py"), default_timeout=60)
        app.session_state["user_id"] = loaded["user_id"]
        app.session_state["today"] = date.fromisoformat(loaded["today"])
        app.run()
        self.assertFalse(app.exception, app.exception)
        ranking = app.dataframe[0].value
        self.assertEqual(len(ranking), 66)
        self.assertAlmostEqual(ranking["Probability"].sum(), 1.0, places=3)
        self.assertIn("Forecast", [metric.label for metric in app.metric])

    def test_forecast_endpoint_and_features(self):
        self.assertIsNone(features(["2019-01-01"], [5.0], [30.0], "2019-01-02"))
        loaded = self.client.post("/demo/4399/load").json()
        result = self.client.get(f"/users/{loaded['user_id']}/forecast", params={"date": loaded["today"]}).json()
        self.assertTrue(result["available"])
        self.assertTrue(150 < result["minutes"] < 420)
        self.assertLess(result["low"], result["minutes"])
        self.assertLess(result["minutes"], result["high"])
        self.assertLess(result["test_mae_minutes"], result["test_mae_riegel_minutes"])


if __name__ == "__main__":
    unittest.main()
