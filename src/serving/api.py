# src/serving/api.py
"""
FastAPI prediction service for the Loan Default model.
Supports single and batch predictions, health checks,
model metadata, and on-demand drift reporting.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
import pandas as pd
import sqlalchemy as sa
from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.training.features import engineer_features
from src.training.trainer import load_production_model
from src.utils.config import get_config
from src.utils.db import get_engine

logger = logging.getLogger(__name__)
cfg = get_config()

# ─────────────────────────────────────────────
# App state (model cache)
# ─────────────────────────────────────────────


class ModelState:
    model = None
    version: str = "unknown"
    loaded_at: datetime | None = None
    feature_names: list[str] = []
    feature_pipeline = None
    lock = asyncio.Lock()


state = ModelState()


async def _load_model():
    """(Re)load the production model from MLflow."""
    async with state.lock:
        logger.info("Loading production model from MLflow…")
        try:
            model, version = load_production_model()
            state.model = model
            state.version = version
            state.loaded_at = datetime.utcnow()
            logger.info("Model version %s loaded at %s", version, state.loaded_at)
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _load_model()

    # Background model refresh task
    async def _refresh_loop():
        ttl = cfg["serving"]["model_cache_ttl_seconds"]
        while True:
            await asyncio.sleep(ttl)
            try:
                await _load_model()
            except Exception as e:
                logger.warning("Model refresh failed: %s", e)

    task = asyncio.create_task(_refresh_loop())
    yield
    task.cancel()


# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────

app = FastAPI(
    title="Loan Default Prediction API",
    description="Real-time loan default risk scoring with automated retraining.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────


class LoanApplication(BaseModel):
    """Single loan application for prediction."""

    sk_id_curr: int | None = Field(None, description="Applicant ID (for logging)")
    amt_income_total: float = Field(..., gt=0, description="Annual income")
    amt_credit: float = Field(..., gt=0, description="Loan amount")
    amt_annuity: float | None = Field(None, gt=0, description="Monthly annuity")
    amt_goods_price: float | None = Field(None, description="Goods price")
    code_gender: str | None = Field(None, description="M / F")
    days_birth: int | None = Field(None, description="Negative days since birth")
    days_employed: int | None = Field(None, description="Negative days employed")
    name_income_type: str | None = None
    name_education_type: str | None = None
    name_family_status: str | None = None
    name_housing_type: str | None = None
    name_contract_type: str | None = None
    ext_source_1: float | None = Field(None, ge=0, le=1)
    ext_source_2: float | None = Field(None, ge=0, le=1)
    ext_source_3: float | None = Field(None, ge=0, le=1)
    cnt_children: int | None = Field(None, ge=0)
    cnt_fam_members: float | None = Field(None, ge=0)
    region_rating_client: int | None = None
    flag_own_car: str | None = None
    flag_own_realty: str | None = None
    occupation_type: str | None = None
    organization_type: str | None = None

    class Config:
        extra = "allow"  # Accept extra fields gracefully


class PredictionResponse(BaseModel):
    prediction_id: str
    sk_id_curr: int | None
    default_probability: float = Field(..., description="P(default) in [0,1]")
    prediction_label: int = Field(..., description="1=high risk, 0=low risk")
    risk_band: str = Field(..., description="LOW / MEDIUM / HIGH / VERY_HIGH")
    model_version: str
    predicted_at: str
    threshold_used: float


class BatchPredictionRequest(BaseModel):
    applications: list[LoanApplication] = Field(..., max_items=1000)
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    batch_id: str
    model_version: str
    total_applications: int
    predicted_defaults: int
    average_default_probability: float
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_loaded_at: str | None
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    loaded_at: str | None
    tracking_uri: str
    experiment_name: str


_START_TIME = time.time()


# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────


def _proba_to_risk_band(proba: float) -> str:
    if proba < 0.15:
        return "LOW"
    elif proba < 0.30:
        return "MEDIUM"
    elif proba < 0.55:
        return "HIGH"
    return "VERY_HIGH"


def _applications_to_df(applications: list[LoanApplication]) -> pd.DataFrame:
    return pd.DataFrame([app.dict() for app in applications])


def _predict_batch(
    df: pd.DataFrame,
    threshold: float = 0.5,
) -> list[dict]:
    """Run predictions on a DataFrame. Returns list of prediction dicts."""
    if state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    try:
        features, _, _ = engineer_features(df, pipeline=state.feature_pipeline, fit=False)
    except Exception as e:
        logger.error("Feature engineering failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Feature engineering error: {e}",
        ) from e

    probas = state.model.predict_proba(features)[:, 1]
    labels = (probas >= threshold).astype(int)
    now = datetime.utcnow().isoformat()

    results = []
    for i, (proba, label) in enumerate(zip(probas, labels, strict=False)):
        results.append(
            {
                "prediction_id": str(uuid.uuid4()),
                "sk_id_curr": (
                    df.get("sk_id_curr", [None] * len(df)).iloc[i]
                    if "sk_id_curr" in df.columns
                    else None
                ),
                "default_probability": round(float(proba), 6),
                "prediction_label": int(label),
                "risk_band": _proba_to_risk_band(float(proba)),
                "model_version": state.version,
                "predicted_at": now,
                "threshold_used": threshold,
            }
        )
    return results


async def _log_predictions(predictions: list[dict], engine=None) -> None:
    """Async background task to persist predictions to DB."""
    if not cfg["serving"]["log_predictions"]:
        return
    try:
        engine = engine or get_engine()
        df = pd.DataFrame(predictions)
        df = df.rename(columns={"default_probability": "prediction_proba"})
        with engine.begin() as conn:
            df.to_sql(
                cfg["serving"]["prediction_log_table"],
                conn,
                schema=cfg["database"]["schema"],
                if_exists="append",
                index=False,
            )
    except Exception as e:
        logger.warning("Failed to log predictions: %s", e)


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health_check():
    return {
        "status": "ok" if state.model is not None else "degraded",
        "model_version": state.version,
        "model_loaded_at": state.loaded_at.isoformat() if state.loaded_at else None,
        "uptime_seconds": round(time.time() - _START_TIME, 1),
    }


@app.get("/model/info", response_model=ModelInfoResponse, tags=["ops"])
async def model_info():
    return {
        "model_name": cfg["mlflow"]["model_name"],
        "model_version": state.version,
        "loaded_at": state.loaded_at.isoformat() if state.loaded_at else None,
        "tracking_uri": cfg["mlflow"]["tracking_uri"],
        "experiment_name": cfg["mlflow"]["experiment_name"],
    }


@app.post("/model/reload", tags=["ops"])
async def reload_model():
    """Force reload of production model from MLflow registry."""
    await _load_model()
    return {"status": "reloaded", "model_version": state.version}


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict_single(
    application: LoanApplication,
    background_tasks: BackgroundTasks,
    threshold: float = 0.5,
):
    """Score a single loan application."""
    df = _applications_to_df([application])
    results = _predict_batch(df, threshold=threshold)
    prediction = results[0]
    background_tasks.add_task(_log_predictions, [prediction])
    return prediction


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"])
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
):
    """Score up to 1 000 loan applications in one call."""
    t0 = time.time()
    df = _applications_to_df(request.applications)
    predictions = _predict_batch(df, threshold=request.threshold)
    elapsed_ms = (time.time() - t0) * 1000

    background_tasks.add_task(_log_predictions, predictions)

    probas = [p["default_probability"] for p in predictions]
    return {
        "predictions": predictions,
        "batch_id": str(uuid.uuid4()),
        "model_version": state.version,
        "total_applications": len(predictions),
        "predicted_defaults": sum(p["prediction_label"] for p in predictions),
        "average_default_probability": round(float(np.mean(probas)), 4),
        "processing_time_ms": round(elapsed_ms, 2),
    }


@app.post("/feedback/{prediction_id}", tags=["monitoring"])
async def submit_feedback(prediction_id: str, actual_label: int):
    """Submit ground-truth outcome to update prediction log."""
    if actual_label not in (0, 1):
        raise HTTPException(status_code=400, detail="actual_label must be 0 or 1")
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                f"""
                UPDATE {cfg["database"]["schema"]}.prediction_log
                SET actual_label = :label, feedback_at = NOW()
                WHERE prediction_id = :pid
                """
            ),
            {"label": actual_label, "pid": prediction_id},
        )
    return {"status": "updated", "prediction_id": prediction_id}


@app.get("/monitoring/summary", tags=["monitoring"])
async def monitoring_summary():
    """Return latest drift report summary."""
    engine = get_engine()
    query = f"""
        SELECT report_type, overall_drift_score, is_drift_detected,
               retrain_triggered, created_at
        FROM {cfg["database"]["schema"]}.drift_reports
        ORDER BY created_at DESC
        LIMIT 10
    """
    with engine.connect() as conn:
        rows = conn.execute(sa.text(query)).mappings().all()
    return {"reports": [dict(r) for r in rows]}


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.serving.api:app",
        host=cfg["serving"]["host"],
        port=cfg["serving"]["port"],
        workers=cfg["serving"]["workers"],
        log_level="info",
    )
