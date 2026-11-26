"""FastAPI backend of the marathon recommender. Every recommendation comes from the trained model.

Run with `python -m uvicorn marathon.api:app --host 0.0.0.0 --port 8000` and open /docs.
"""

import os
import time
from datetime import date as Date
from datetime import timedelta
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from marathon.data import MIN_SESSION_KM, NUM_ITEMS, PACE_RANGE, describe_item, plausible_pace
from marathon.serving import Recommender
from marathon.store import EFFORTS, INTERACTION_TYPES, InteractionError, Store

MODEL_DIR = Path(os.environ.get("MARATHON_MODEL_DIR", "models/marathon"))
DB_PATH = os.environ.get("MARATHON_DB", "artifacts/marathon/interactions.sqlite3")
DEMO_DIR = Path(os.environ.get("MARATHON_DEMO_DIR", "demo"))
# Input contract for one session; the frontend uses the same bounds.
MAX_SESSION_KM = 100
MAX_SESSION_MINUTES = 900


class Session(BaseModel):
    model_config = ConfigDict(extra="forbid")
    date: Date
    km: float = Field(gt=0, le=MAX_SESSION_KM, allow_inf_nan=False)
    minutes: float = Field(gt=0, le=MAX_SESSION_MINUTES, allow_inf_nan=False)

    @model_validator(mode="after")
    def check_quality(self):
        check_session(self.km, self.minutes)
        return self


class UserRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(min_length=1, max_length=100)
    race_date: Optional[Date] = None
    goal_minutes: Optional[float] = Field(default=None, gt=120, lt=480)
    sessions: list[Session] = Field(default_factory=list, max_length=400)


class RecommendRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(min_length=1, max_length=100)
    date: Optional[Date] = None
    top_k: int = Field(default=10, ge=1, le=NUM_ITEMS, strict=True)


class PlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(min_length=1, max_length=100)
    start_date: Optional[Date] = None
    days: int = Field(default=7, ge=1, le=28, strict=True)
    run_weekdays: list[int] = Field(default=[1, 3, 5, 6], min_length=1, max_length=7,
                                    description="0 = Monday ... 6 = Sunday")
    top_k: int = Field(default=3, ge=1, le=10, strict=True)


class Interaction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(min_length=1, max_length=100)
    type: Literal[INTERACTION_TYPES]
    date: Date
    rec_id: Optional[str] = None
    item: Optional[int] = Field(default=None, ge=1, le=NUM_ITEMS)
    km: Optional[float] = Field(default=None, gt=0, le=MAX_SESSION_KM, allow_inf_nan=False)
    minutes: Optional[float] = Field(default=None, gt=0, le=MAX_SESSION_MINUTES, allow_inf_nan=False)
    effort: Optional[Literal[EFFORTS]] = None

    @model_validator(mode="after")
    def check_quality(self):
        if self.km is not None and self.minutes is not None:
            check_session(self.km, self.minutes)
        return self


def check_session(km, minutes):
    """The training pipeline's quality rule, enforced at the API boundary."""
    if km < MIN_SESSION_KM:
        raise ValueError(f"a session must be at least {MIN_SESSION_KM:g} km")
    if not plausible_pace(km, minutes):
        raise ValueError(f"pace must be between {PACE_RANGE[0]:g} and {PACE_RANGE[1]:g} min/km")


def goal_pace(goal_minutes):
    if goal_minutes is None:
        return None
    seconds = round(goal_minutes / 42.195 * 60)
    return f"{seconds // 60}:{seconds % 60:02d}/km"


def create_app(model_dir=MODEL_DIR, db_path=DB_PATH, demo_dir=DEMO_DIR):
    recommender = Recommender(model_dir)
    store = Store(db_path)
    demo_dir = Path(demo_dir)
    app = FastAPI(title="StridePlan Marathon", version="1.0.0",
                  description="Context-aware sequential recommender of marathon training sessions")
    app.state.store = store
    app.state.recommender = recommender

    def require_user(user_id):
        user = store.user(user_id)
        if user is None:
            raise HTTPException(404, "Unknown user; create it with POST /users")
        return user

    def default_day(user_id):
        sessions = store.sessions(user_id)
        interactions = store.interactions(user_id)
        dates = [pd.Timestamp(row["date"]) for row in sessions + interactions]
        return (max(dates) + timedelta(days=1)).date() if dates else Date.today()

    @app.get("/health")
    def health():
        return {"status": "ok", "model": recommender.name, "items": NUM_ITEMS}

    @app.get("/catalogue")
    def catalogue():
        return [describe_item(item) for item in range(1, NUM_ITEMS + 1)]

    @app.post("/users")
    def create_user(request: UserRequest):
        user = store.upsert_user(request.user_id, request.race_date and str(request.race_date), request.goal_minutes)
        store.add_sessions(request.user_id, [s.model_dump() for s in request.sessions], "onboarding")
        return {**user, "sessions": len(store.sessions(request.user_id))}

    @app.get("/users/{user_id}")
    def get_user(user_id: str):
        user = require_user(user_id)
        return {**user, "goal_pace": goal_pace(user["goal_minutes"]), "next_day": str(default_day(user_id)),
                "sessions": store.sessions(user_id), "interactions": store.interactions(user_id)}

    @app.post("/recommend")
    def recommend(request: RecommendRequest):
        user = require_user(request.user_id)
        target = request.date or default_day(request.user_id)
        started = time.perf_counter()
        result = recommender.recommend(store.sessions(request.user_id, before=target), target, user["race_date"], request.top_k)
        result["rec_id"] = store.save_recommendation(request.user_id, target, recommender.name,
                                                     [item["item"] for item in result["recommendations"]])
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        result["goal_pace"] = goal_pace(user["goal_minutes"])
        return result

    @app.post("/plan")
    def plan(request: PlanRequest):
        user = require_user(request.user_id)
        start = request.start_date or default_day(request.user_id)
        weekdays = set(request.run_weekdays)
        if not weekdays <= set(range(7)):
            raise HTTPException(422, "run_weekdays must be between 0 and 6")
        run_dates = [start + timedelta(days=offset) for offset in range(request.days)
                     if (start + timedelta(days=offset)).weekday() in weekdays]
        if user["race_date"]:
            run_dates = [day for day in run_dates if str(day) < user["race_date"]]
        started = time.perf_counter()
        days = recommender.plan(store.sessions(request.user_id, before=start), run_dates, user["race_date"], request.top_k)
        for day in days:
            day["rec_id"] = store.save_recommendation(request.user_id, day["date"], recommender.name,
                                                      [item["item"] for item in day["recommendations"]])
        return {"user_id": request.user_id, "start_date": str(start), "race_date": user["race_date"],
                "goal_pace": goal_pace(user["goal_minutes"]), "days": days,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1)}

    @app.post("/log_interaction")
    def log_interaction(request: Interaction):
        require_user(request.user_id)
        try:
            interaction_id = store.log_interaction(**request.model_dump(mode="json"))
        except InteractionError as error:
            raise HTTPException(422, str(error)) from error
        return {"interaction_id": interaction_id, "stored": True}

    @app.get("/demo/runners")
    def demo_runners():
        path = demo_dir / "runners.csv"
        if not path.exists():
            return []
        return pd.read_csv(path).to_dict(orient="records")

    @app.post("/demo/{runner}/load")
    def load_demo(runner: int):
        runners = pd.read_csv(demo_dir / "runners.csv")
        match = runners[runners["runner"] == runner]
        if match.empty:
            raise HTTPException(404, "Unknown demo runner")
        info = match.iloc[0]
        sessions = pd.read_csv(demo_dir / "sessions.csv")
        history = sessions[(sessions["runner"] == runner) & (sessions["date"] < info["today"])]
        user_id = f"demo-{runner}"
        # Loading a demo runner always restarts it from its demo date: earlier logged state is
        # removed so the stored history matches the returned `today`.
        store.reset_user(user_id)
        store.upsert_user(user_id, info["race_date"], None)
        store.add_sessions(user_id, history.to_dict(orient="records"), "bmclab")
        return {"user_id": user_id, "today": info["today"], "race_date": info["race_date"],
                "race": info["race"], "sessions": len(history)}

    return app


app = create_app() if os.environ.get("MARATHON_SKIP_APP") != "1" else None
