"""StridePlan recommendation API, available at /docs.

Serves the trained GRU when a bundle exists (STRIDEPLAN_BUNDLE, default artifacts/first_results).
Known Endomondo histories are read from STRIDEPLAN_SESSIONS (default data/processed/sessions.csv).
Without a bundle, the Week 2 mock catalogue is served and every response says so.
"""

import os
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


class Session(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_timestamp: float = Field(ge=0, allow_inf_nan=False, description="Unix seconds")
    duration_seconds: float = Field(gt=0, le=86400, allow_inf_nan=False)
    avg_speed_mps: Optional[float] = Field(default=None, allow_inf_nan=False,
                                           description="Sessions without a valid speed are ignored, as in training")
    avg_heart_rate_bpm: Optional[float] = Field(default=None, allow_inf_nan=False)
    workout_id: Optional[str] = Field(default=None, max_length=100)


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: Optional[str] = Field(default=None, min_length=1, max_length=100)
    sport: Literal["run"] = "run"
    max_duration_minutes: float = Field(gt=0, le=360, allow_inf_nan=False)
    top_k: int = Field(default=5, ge=1, le=20, strict=True)
    history: Optional[list[Session]] = Field(
        default=None, max_length=1000,
        description="Past sessions. When given, it replaces any stored history for user_id")


class Workout(BaseModel):
    template_id: str
    title: str
    duration_min_minutes: float
    duration_max_minutes: float
    speed_min_kmh: Optional[float] = None
    speed_max_kmh: Optional[float] = None
    score: Optional[float] = Field(default=None, description="Model probability; null for popularity and mock")
    sport: Literal["run"] = "run"


class RecommendationResponse(BaseModel):
    mode: Literal["model", "mock"]
    model_version: Optional[str]
    personalized: bool
    strategy: Literal["gru", "global_popularity", "mock"]
    history_sessions_used: int
    recommendations: list[Workout]


DEMO_CATALOGUE = tuple(
    Workout(template_id=f"demo-run-{minutes}", title=f"{minutes}-minute run",
            duration_min_minutes=minutes, duration_max_minutes=minutes)
    for minutes in (20, 30, 45))


def create_app(bundle_dir=None, sessions_path=None):
    bundle_dir = Path(bundle_dir or os.environ.get("STRIDEPLAN_BUNDLE", "artifacts/first_results"))
    sessions_path = Path(sessions_path or os.environ.get("STRIDEPLAN_SESSIONS", "data/processed/sessions.csv"))
    recommender = None
    histories = {}
    if (bundle_dir / "bundle.json").is_file():
        # Torch is imported only when a model is served, so the mock contract runs without it.
        from strideplan.bundle import load_bundle
        from strideplan.inference import Recommender, load_histories
        recommender = Recommender(load_bundle(bundle_dir))
        if sessions_path.is_file():
            histories = load_histories(sessions_path)
    mode = "model" if recommender else "mock"
    version = recommender.bundle.version if recommender else None
    app = FastAPI(title="StridePlan", version="0.2.0", description=f"Next-workout recommendations ({mode} mode)")

    @app.get("/health")
    def health():
        return {"status": "ok", "mode": mode, "model_version": version, "known_users": len(histories)}

    @app.get("/catalogue", response_model=list[Workout])
    def catalogue():
        if recommender is None:
            return list(DEMO_CATALOGUE)
        return [Workout(**{**item, "template_id": str(item["template_id"])})
                for item in recommender.templates.values()]

    @app.post("/recommend", response_model=RecommendationResponse)
    def recommend(request: RecommendationRequest):
        if recommender is None:
            eligible = [workout for workout in DEMO_CATALOGUE
                        if workout.duration_min_minutes <= request.max_duration_minutes]
            return RecommendationResponse(mode="mock", model_version=None, personalized=False, strategy="mock",
                                          history_sessions_used=0, recommendations=eligible[:request.top_k])
        if request.history is not None:
            records = [session.model_dump() for session in request.history]
        else:
            records = histories.get(request.user_id, []) if request.user_id else []
        try:
            result = recommender.recommend(records, request.max_duration_minutes, request.top_k)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        workouts = [Workout(**{**item, "template_id": str(item["template_id"])}) for item in result["recommendations"]]
        return RecommendationResponse(mode="model", model_version=version,
                                      personalized=result["strategy"] == "gru", strategy=result["strategy"],
                                      history_sessions_used=result["history_sessions_used"],
                                      recommendations=workouts)

    return app


app = create_app()
