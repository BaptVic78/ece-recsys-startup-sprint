"""StridePlan recommendation API with a feedback loop, available at /docs.

STRIDEPLAN_BUNDLE    bundle directory, or a model registry written by strideplan.retrain
                     (default: artifacts/models if it has a registry, else artifacts/first_results)
STRIDEPLAN_SESSIONS  known Endomondo histories (default data/processed/sessions.csv)
STRIDEPLAN_DB        SQLite feedback log (default artifacts/feedback/feedback.sqlite3)
STRIDEPLAN_EPSILON   share of responses with one exploration slot (default 0.1)

Without a bundle, the Week 2 mock catalogue is served, nothing is logged and every response says so.
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
    workout_id: Optional[str] = Field(default=None, min_length=1, max_length=100)


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: Optional[str] = Field(default=None, min_length=1, max_length=100)
    sport: Literal["run"] = "run"
    max_duration_minutes: float = Field(gt=0, le=360, allow_inf_nan=False)
    top_k: int = Field(default=5, ge=1, le=20, strict=True)
    history: Optional[list[Session]] = Field(
        default=None, max_length=1000,
        description="Past sessions. When given, they replace the stored history; feedback still applies")


class Workout(BaseModel):
    template_id: str
    title: str
    duration_min_minutes: float
    duration_max_minutes: float
    speed_min_kmh: Optional[float] = None
    speed_max_kmh: Optional[float] = None
    score: Optional[float] = Field(default=None, description="Model probability; null for popularity and mock")
    predicted_effort: Optional[float] = Field(default=None, description="0 too easy, 1 too hard; once learned")
    position: Optional[int] = None
    explored: bool = False
    reasons: list[str] = []
    sport: Literal["run"] = "run"


class RecommendationResponse(BaseModel):
    mode: Literal["model", "mock"]
    model_version: Optional[str]
    rec_id: Optional[str] = Field(description="Send it back with feedback on these recommendations")
    personalized: bool
    strategy: Literal["gru", "global_popularity", "mock"]
    history_sessions_used: int
    feedback_events_used: int = 0
    feedback_applied: bool = False
    recommendations: list[Workout]


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rec_id: str = Field(min_length=1, max_length=64)
    template_id: str = Field(min_length=1, max_length=20)
    action: Literal["accepted", "rejected", "completed"]
    effort: Optional[Literal["too_easy", "ok", "too_hard"]] = Field(
        default=None, description="The three effort buttons, for a completed workout")
    session: Optional[Session] = Field(default=None, description="What was actually done, for a completed workout")


class SessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(min_length=1, max_length=100)
    session: Session


DEMO_CATALOGUE = tuple(
    Workout(template_id=f"demo-run-{minutes}", title=f"{minutes}-minute run",
            duration_min_minutes=minutes, duration_max_minutes=minutes)
    for minutes in (20, 30, 45))


def default_bundle_dir():
    models = Path("artifacts/models")
    return models if (models / "registry.json").is_file() else Path("artifacts/first_results")


def to_workout(item):
    fields = {key: item[key] for key in Workout.model_fields if key in item}
    return Workout(**{**fields, "template_id": str(item["template_id"])})


def create_app(bundle_dir=None, sessions_path=None, db_path=None, epsilon=None, seed=None):
    bundle_dir = Path(bundle_dir or os.environ.get("STRIDEPLAN_BUNDLE") or default_bundle_dir())
    sessions_path = Path(sessions_path or os.environ.get("STRIDEPLAN_SESSIONS", "data/processed/sessions.csv"))
    db_path = db_path or os.environ.get("STRIDEPLAN_DB", "artifacts/feedback/feedback.sqlite3")
    epsilon = float(os.environ.get("STRIDEPLAN_EPSILON", 0.1)) if epsilon is None else epsilon
    service = None
    if (bundle_dir / "bundle.json").is_file() or (bundle_dir / "registry.json").is_file():
        # Torch is imported only when a model is served, so the mock contract runs without it.
        from strideplan.bundle import load_bundle, resolve_bundle_dir
        from strideplan.inference import Recommender, load_histories
        from strideplan.reranker import Policy
        from strideplan.service import StridePlanService
        from strideplan.store import Store
        histories = load_histories(sessions_path) if sessions_path.is_file() else {}
        service = StridePlanService(Recommender(load_bundle(resolve_bundle_dir(bundle_dir))), Store(db_path),
                                    histories, Policy(epsilon=epsilon), seed=seed)
    mode = "model" if service else "mock"
    version = service.model_version if service else None
    app = FastAPI(title="StridePlan", version="0.3.0", description=f"Next-workout recommendations ({mode} mode)")

    def require_service():
        if service is None:
            raise HTTPException(status_code=503, detail="Feedback needs a trained model; the API is in mock mode")
        return service

    def parse_template(template_id):
        if not template_id.isdigit():
            raise HTTPException(status_code=422, detail="template_id must be a catalogue category ID")
        return int(template_id)

    @app.get("/health")
    def health():
        return {"status": "ok", "mode": mode, "model_version": version,
                "known_users": len(service.base_histories) if service else 0,
                "logged": service.store.counts() if service else None}

    @app.get("/catalogue", response_model=list[Workout])
    def catalogue():
        if service is None:
            return list(DEMO_CATALOGUE)
        return [to_workout(item) for item in service.recommender.templates.values()]

    @app.post("/recommend", response_model=RecommendationResponse)
    def recommend(request: RecommendationRequest):
        if service is None:
            eligible = [workout for workout in DEMO_CATALOGUE
                        if workout.duration_min_minutes <= request.max_duration_minutes]
            return RecommendationResponse(mode="mock", model_version=None, rec_id=None, personalized=False,
                                          strategy="mock", history_sessions_used=0,
                                          recommendations=eligible[:request.top_k])
        history = [session.model_dump() for session in request.history] if request.history is not None else None
        try:
            result = service.recommend(request.user_id, request.max_duration_minutes, request.top_k, history)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return RecommendationResponse(mode="model", model_version=version, rec_id=result["rec_id"],
                                      personalized=result["strategy"] == "gru" or result["feedback_applied"],
                                      strategy=result["strategy"],
                                      history_sessions_used=result["history_sessions_used"],
                                      feedback_events_used=result["feedback_events_used"],
                                      feedback_applied=result["feedback_applied"],
                                      recommendations=[to_workout(item) for item in result["recommendations"]])

    @app.post("/feedback")
    def feedback(request: FeedbackRequest):
        from strideplan.store import FeedbackError
        current = require_service()
        try:
            return current.record_feedback(request.rec_id, parse_template(request.template_id), request.action,
                                           request.effort, request.session.model_dump() if request.session else None)
        except FeedbackError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/sessions")
    def sessions(request: SessionRequest):
        return {"session_logged": require_service().record_session(request.user_id, request.session.model_dump())}

    @app.get("/users/{user_id}/history")
    def history(user_id: str):
        current = require_service()
        return {"user_id": user_id, "known_sessions": len(current.base_histories.get(user_id, [])),
                "logged_sessions": current.store.sessions(user_id), "feedback": current.store.feedback(user_id)}

    return app


def __getattr__(name):
    # Build the served app on first access (uvicorn strideplan.api:app), not when tests import this module.
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
