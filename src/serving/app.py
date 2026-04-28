"""FastAPI application for HDB resale price prediction.

Endpoints
---------
POST /predict       — single-flat resale price prediction
POST /explain       — SHAP feature contributions (stub, returns 501)
GET  /health        — service liveness and model load status
GET  /model-info    — metadata for the currently loaded model version
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from serving.config import ServingConfig
from serving.logging_config import configure_logging
from serving.model_loader import ModelLoader
from serving.schemas import (
    ExplainRequest,
    HDBFeatureInput,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)

logger = logging.getLogger(__name__)

config = ServingConfig()
loader = ModelLoader(config)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    loader.load_initial()
    loader.start_background_polling()
    yield


app = FastAPI(
    title="HDB Resale Price Predictor",
    description=(
        "Serves the @champion GradientBoostingRegressor from the MLflow registry. "
        "The model is swapped automatically when the @champion alias is reassigned."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


def _build_feature_dataframe(req: HDBFeatureInput) -> pd.DataFrame:
    """Convert a feature request into the DataFrame expected by the sklearn pipeline.

    Applies the same normalisation as load_and_prepare_data in the training
    script: town and storey_range uppercased, flat_info derived from
    flat_type + flat_model, float_time_series derived from month string.
    """
    year_str, month_str = req.month.split("-")
    float_time_series = int(year_str) + (int(month_str) - 1) / 12.0
    flat_info = f"{req.flat_type.strip().upper()} {req.flat_model.strip().title()}"

    return pd.DataFrame(
        [
            {
                "floor_area_sqm": req.floor_area_sqm,
                "lease_commence_date": req.lease_commence_date,
                "float_time_series": float_time_series,
                "town": req.town.strip().upper(),
                "storey_range": req.storey_range.strip().upper(),
                "flat_info": flat_info,
            }
        ]
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    """Return a resale price prediction for a single HDB flat.

    The serving model is the current @champion version. The background reload
    thread swaps it within 60 seconds of a promotion event, so the version
    field in the response may change between calls after a promotion.
    """
    model = loader.get_model()
    version = loader.get_version()
    df = _build_feature_dataframe(req)
    try:
        prediction = float(model.predict(df)[0])  # type: ignore[index]
    except Exception as exc:
        logger.exception("Prediction failed for request: %s", req.model_dump())
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc

    return PredictResponse(
        predicted_resale_price=round(prediction, 2),
        model_version=version or "unknown",
        model_alias=config.model_alias,
    )


@app.post("/explain")
async def explain(req: ExplainRequest) -> JSONResponse:
    """Return SHAP feature contributions for a single prediction.

    SHAP TreeExplainer integration is pending. The request schema is validated
    now so the contract is established; the response body will be replaced with
    per-feature contributions in the next serving task.
    """
    return JSONResponse(
        status_code=501,
        content={
            "detail": (
                "SHAP explainability is not yet implemented. "
                "TreeExplainer integration is tracked as a separate task and "
                "will replace this stub in the next iteration."
            )
        },
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return service liveness and current model load status."""
    version = loader.get_version()
    return HealthResponse(
        status="ok",
        model_loaded=version is not None,
        model_version=version,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    """Return provenance metadata for the currently loaded model version."""
    return ModelInfoResponse(
        model_name=config.model_name,
        model_alias=config.model_alias,
        model_version=loader.get_version(),
        run_id=loader.get_run_id(),
    )
