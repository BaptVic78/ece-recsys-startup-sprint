"""Check the running Docker app and its saved interactions."""

import argparse
import json
import subprocess
import time
import uuid
from datetime import date, timedelta
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def request(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=10) as response:
        return json.load(response)


def wait_ready(api, frontend):
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            health = request(api + "/health")
            with urlopen(frontend + "/_stcore/health", timeout=3) as response:
                if health["status"] == "ok" and response.status == 200:
                    return health
        except (URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(1)
    raise RuntimeError("The API or frontend did not become healthy")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--frontend", default="http://localhost:8501")
    args = parser.parse_args()
    health = wait_ready(args.api, args.frontend)
    user = "docker-check-" + uuid.uuid4().hex[:10]
    start = date(2019, 8, 20)
    sessions = [
        {"date": str(start + timedelta(days=2 * i)), "km": 10.0, "minutes": 55.0}
        for i in range(6)
    ]
    request(args.api + "/users", {
        "user_id": user, "race_date": "2019-09-29", "sessions": sessions,
    })
    today = date(2019, 9, 1)
    started = time.perf_counter()
    recommendation = request(args.api + "/recommend", {
        "user_id": user, "date": str(today), "top_k": 3,
    })
    latency = time.perf_counter() - started
    assert recommendation["strategy"] == "model", recommendation
    assert len(recommendation["recommendations"]) == 3
    assert latency < 2, f"Recommendation took {latency:.2f} s"

    for offset, action in enumerate(("completed", "skipped", "missed")):
        day = str(today + timedelta(days=offset))
        rec = request(args.api + "/recommend", {"user_id": user, "date": day, "top_k": 3})
        top = rec["recommendations"][0]
        body = {"user_id": user, "date": day, "type": action}
        if action != "missed":
            body.update(rec_id=rec["rec_id"], item=top["item"])
        if action == "completed":
            body.update(km=top["km"], minutes=top["minutes"], effort="ok")
        assert request(args.api + "/log_interaction", body)["stored"]

    before = request(args.api + "/users/" + user)
    assert [row["type"] for row in before["interactions"]] == ["completed", "skipped", "missed"]
    assert len(before["sessions"]) == 7
    subprocess.run(["docker", "compose", "restart"], cwd=ROOT, check=True, timeout=120)
    wait_ready(args.api, args.frontend)
    after = request(args.api + "/users/" + user)
    assert before == after, "Saved data changed after restart"

    plan = request(args.api + "/plan", {"user_id": user, "days": 7})
    assert plan["start_date"] == "2019-09-04"
    assert plan["days"]
    report = {
        "model": health["model"],
        "strategy": recommendation["strategy"],
        "recommendation_seconds": round(latency, 3),
        "saved_actions": [row["type"] for row in after["interactions"]],
        "sessions_after_restart": len(after["sessions"]),
        "restart_persistence": "passed",
        "plan_days": len(plan["days"]),
        "test_user": user,
    }
    output = ROOT / "artifacts" / "docker-checks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
