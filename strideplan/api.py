"""Week 2 mock recommendation contract, available at /docs."""

from typing import Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field


app = FastAPI(title="StridePlan", version="0.1.0", description="Week 2 mock recommendations")


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: Optional[str] = Field(default=None, min_length=1, max_length=100)
    sport: Literal["run"] = "run"
    max_duration_minutes: float = Field(gt=0, le=360, allow_inf_nan=False)
    top_k: int = Field(default=5, ge=1, le=20, strict=True)


class Workout(BaseModel):
    template_id: str
    title: str
    duration_minutes: int
    sport: Literal["run"] = "run"


class RecommendationResponse(BaseModel):
    mode: Literal["mock"] = "mock"
    personalized: Literal[False] = False
    recommendations: list[Workout]


DEMO_CATALOGUE = (
    Workout(template_id="demo-run-20", title="20-minute run", duration_minutes=20),
    Workout(template_id="demo-run-30", title="30-minute run", duration_minutes=30),
    Workout(template_id="demo-run-45", title="45-minute run", duration_minutes=45),
)


@app.get("/health")
def health():
    return {"status": "ok", "mode": "mock"}


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest):
    eligible = [workout for workout in DEMO_CATALOGUE if workout.duration_minutes <= request.max_duration_minutes]
    return RecommendationResponse(recommendations=eligible[:request.top_k])
