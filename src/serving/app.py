"""FastAPI application for HDB resale price prediction.

Endpoints
---------
POST /predict       — single-flat resale price prediction
POST /explain       — SHAP feature contributions (TreeExplainer)
GET  /health        — service liveness and model load status
GET  /model-info    — metadata for the currently loaded model version
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status

from serving.config import ServingConfig
from serving.logging_config import configure_logging
from serving.model_loader import ModelLoader
from serving.schemas import (
    ExplainRequest,
    ExplainResponse,
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
    if version is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded yet — try again shortly",
        )
    df = _build_feature_dataframe(req)
    try:
        prediction = float(model.predict(df)[0])  # type: ignore[index]
    except Exception as exc:
        logger.exception("Prediction failed for request: %s", req.model_dump())
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc

    return PredictResponse(
        predicted_resale_price=round(prediction, 2),
        model_version=version,
        model_alias=config.model_alias,
    )


@app.post("/explain", response_model=ExplainResponse)
async def explain(req: ExplainRequest) -> ExplainResponse:
    """Return SHAP feature contributions for a single prediction.

    Uses a TreeExplainer initialised at model load time — not per request.
    The explainer operates on post-preprocessing features, so feature_contributions
    keys are the transformed column names from the sklearn ColumnTransformer.

    By construction: sum(feature_contributions.values()) + base_value ≈ predicted_resale_price.
    """
    version = loader.get_version()
    if version is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded yet — try again shortly",
        )
    shap_bundle = loader.get_explainer()
    if shap_bundle is None:
        logger.error("SHAP explainer unavailable despite model being loaded (version %s)", version)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SHAP explainer not available — try again shortly",
        )
    model = loader.get_model()
    df = _build_feature_dataframe(req)
    try:
        prediction = float(model.predict(df)[0])  # type: ignore[index]
        X_transformed = shap_bundle.preprocessor.transform(df)  # type: ignore[attr-defined]
        shap_values = shap_bundle.explainer.shap_values(X_transformed)
        # expected_value is a scalar for some tree models but a 1-element array for
        # GradientBoostingRegressor — .item() handles both safely.
        base_value = float(np.asarray(shap_bundle.explainer.expected_value).item())
        contributions: dict[str, float] = dict(
            zip(shap_bundle.feature_names, shap_values[0].tolist(), strict=False)
        )
    except Exception as exc:
        logger.exception("SHAP explanation failed for request: %s", req.model_dump())
        raise HTTPException(status_code=500, detail="Explanation failed.") from exc

    top_contributor = max(contributions, key=lambda k: abs(contributions[k]))
    logger.info(
        "explain request: town=%s flat_type=%s floor_area_sqm=%.1f, "
        "prediction=%.2f, top_contributor=%s",
        req.town,
        req.flat_type,
        req.floor_area_sqm,
        prediction,
        top_contributor,
    )

    return ExplainResponse(
        predicted_resale_price=round(prediction, 2),
        base_value=base_value,
        feature_contributions=contributions,
        model_version=version,
        model_alias=config.model_alias,
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
