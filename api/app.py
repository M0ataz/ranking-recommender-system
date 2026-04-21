"""
Ranking Recommender System - REST API

Provides:
  GET /recommend/{user_id}         - Top-K ranked recommendations for a user
  GET /recommend/{user_id}?top_k=N - Customizable K
  GET /health                      - Health check
  GET /model/info                  - Model metadata
  GET /docs                        - Auto-generated Swagger UI
"""

import os
import sys
import time
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is on the path regardless of working directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.recommend import RecommendationEngine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("recommender.api")

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Ranking Recommender System API",
    description=(
        "Production-level ranking recommender system using Alternating Least Squares (ALS) "
        "matrix factorization on implicit feedback data. Returns top-K ranked item "
        "recommendations with relevance scores. Handles cold-start users via popularity fallback."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model path (can be overridden via environment variable)
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(PROJECT_ROOT, "models", "als_model.pkl")
)

# Lazy-loaded engine (initialized on first request)
_engine: Optional[RecommendationEngine] = None


def get_engine() -> RecommendationEngine:
    global _engine
    if _engine is None:
        logger.info(f"Loading model from {MODEL_PATH}...")
        _engine = RecommendationEngine(model_path=MODEL_PATH)
        # Trigger model load immediately to surface errors early
        _engine._load_model()
        logger.info("Model loaded successfully.")
    return _engine


# ---------------------------------------------------------------------------
# Response Schemas
# ---------------------------------------------------------------------------
class RecommendedItem(BaseModel):
    rank: int = Field(..., description="1-indexed rank position in the recommendation list.")
    item_id: int = Field(..., description="Unique item identifier.")
    score: float = Field(..., description="Relevance score (higher = more relevant).")
    cold_start: bool = Field(
        ...,
        description="True if this recommendation was generated via cold-start fallback "
                    "(user was not seen during training)."
    )


class RecommendationResponse(BaseModel):
    user_id: int = Field(..., description="The queried user's ID.")
    top_k: int = Field(..., description="Number of recommendations returned.")
    recommendations: List[RecommendedItem] = Field(..., description="Ranked list of recommendations.")
    latency_ms: float = Field(..., description="Response latency in milliseconds.")
    cold_start: bool = Field(
        ...,
        description="True if the user was not seen during training (cold-start scenario)."
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    num_users: int
    num_items: int
    factors: int
    regularization: float
    iterations: int
    alpha: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    tags=["System"]
)
def health_check():
    """Returns the health status of the API and whether the model is loaded."""
    model_loaded = _engine is not None and _engine._model is not None
    return HealthResponse(status="ok", model_loaded=model_loaded)


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Model Metadata",
    tags=["System"]
)
def model_info():
    """Returns metadata about the currently loaded recommendation model."""
    try:
        engine = get_engine()
        info = engine.get_model_info()
        return ModelInfoResponse(**info)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get(
    "/recommend/{user_id}",
    response_model=RecommendationResponse,
    summary="Get Top-K Ranked Recommendations",
    tags=["Recommendations"]
)
def recommend(
    user_id: int,
    top_k: int = Query(default=10, ge=1, le=100, description="Number of top-K items to return (1–100)."),
):
    """
    Returns top-K ranked item recommendations for the given user.

    - **Known users**: Recommendations are generated using ALS latent factors,
      ranking all unseen items by predicted relevance score.
    - **Cold-start users** (not seen during training): Falls back to a
      popularity-based ranking of the most-interacted items globally.

    The `cold_start` field in each item (and at the response level) indicates
    which scenario was triggered.
    """
    try:
        engine = get_engine()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model not available: {str(e)}. Run `python scripts/train.py` first."
        )

    start_time = time.perf_counter()

    try:
        recs = engine.recommend(user_id=user_id, top_k=top_k)
    except Exception as e:
        logger.error(f"Error generating recommendations for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")

    latency_ms = (time.perf_counter() - start_time) * 1000
    is_cold_start = recs[0]["cold_start"] if recs else False

    logger.info(
        f"user_id={user_id} top_k={top_k} cold_start={is_cold_start} "
        f"latency={latency_ms:.2f}ms"
    )

    return RecommendationResponse(
        user_id=user_id,
        top_k=len(recs),
        recommendations=[RecommendedItem(**r) for r in recs],
        latency_ms=round(latency_ms, 3),
        cold_start=is_cold_start
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
