"""Prediction and SHAP explanation tools backed by the in-process model.

Both tools share a lazily-initialised module-level :class:`ModelLoader`
singleton. The MCP server runs as its own process, so it loads the
``@champion`` model the first time either tool is invoked and reuses that
in-memory model for the lifetime of the process — no per-call reload, no HTTP
round-trip to the FastAPI service.
"""

import threading
from typing import TypedDict

import numpy as np
import pandas as pd

from mcp_server.config import MCPConfig
from serving.config import ServingConfig
from serving.model_loader import ModelLoader

_config = MCPConfig()
_loader: ModelLoader | None = None
_loader_lock = threading.Lock()


def _get_loader() -> ModelLoader:
    """Return the shared ModelLoader, loading the champion model on first use.

    Thread-safe: the double-checked lock guards against two concurrent first
    invocations both triggering a load.
    """
    global _loader
    with _loader_lock:
        if _loader is None:
            loader = ModelLoader(
                ServingConfig(
                    mlflow_tracking_uri=_config.mlflow_tracking_uri,
                    model_name=_config.model_name,
                    model_alias=_config.model_alias,
                )
            )
            loader.load_initial()
            _loader = loader
        return _loader


def _build_input_frame(
    town: str,
    flat_type: str,
    floor_area_sqm: float,
    lease_commence_date: int,
    month: str,
) -> pd.DataFrame:
    """Build the single-row raw-feature DataFrame the pipeline expects.

    Mirrors the normalisation the FastAPI serving layer applies: town and
    flat_type are upper-cased so they match the training distribution.
    """
    return pd.DataFrame(
        [
            {
                "town": town.strip().upper(),
                "flat_type": flat_type.strip().upper(),
                "floor_area_sqm": floor_area_sqm,
                "lease_commence_date": lease_commence_date,
                "month": month,
            }
        ]
    )


class PredictionResult(TypedDict):
    """Structured prediction returned to the LLM client."""

    predicted_resale_price: float
    model_version: int
    model_alias: str


class FeatureContribution(TypedDict):
    """One feature's signed SHAP contribution to a prediction."""

    feature: str
    contribution: float


class ExplanationResult(TypedDict):
    """SHAP explanation with the top contributors by absolute value."""

    predicted_resale_price: float
    base_value: float
    top_contributors: list[FeatureContribution]
    model_version: int
    model_alias: str


def predict_price(
    town: str,
    flat_type: str,
    floor_area_sqm: float,
    lease_commence_date: int,
    month: str,
) -> PredictionResult:
    """Predict the resale price of a single HDB flat.

    Args:
        town: HDB town, e.g. "TAMPINES". Case-insensitive.
        flat_type: Flat type, e.g. "4 ROOM". Case-insensitive.
        floor_area_sqm: Floor area in square metres.
        lease_commence_date: Year the lease commenced, e.g. 1985.
        month: Transaction month in YYYY-MM format, e.g. "2024-06".

    Returns:
        The predicted resale price in SGD plus the serving model's version and alias.
    """
    loader = _get_loader()
    model = loader.get_model()
    df = _build_input_frame(town, flat_type, floor_area_sqm, lease_commence_date, month)
    prediction = float(model.predict(df)[0])  # type: ignore[index]
    version = loader.get_version()
    assert version is not None  # load_initial succeeded, so a version is set
    return PredictionResult(
        predicted_resale_price=round(prediction, 2),
        model_version=version,
        model_alias=_config.model_alias,
    )


def explain_prediction(
    town: str,
    flat_type: str,
    floor_area_sqm: float,
    lease_commence_date: int,
    month: str,
) -> ExplanationResult:
    """Explain a single resale price prediction with SHAP feature contributions.

    Returns the five contributors with the largest absolute SHAP value, sorted
    descending. By construction the full set of contributions plus base_value
    sums to the predicted price; the top five are the dominant drivers.

    Args:
        town: HDB town, e.g. "TAMPINES". Case-insensitive.
        flat_type: Flat type, e.g. "4 ROOM". Case-insensitive.
        floor_area_sqm: Floor area in square metres.
        lease_commence_date: Year the lease commenced, e.g. 1985.
        month: Transaction month in YYYY-MM format, e.g. "2024-06".

    Returns:
        The predicted price, the model's base (expected) value, the top five
        feature contributions, and the serving model's version and alias.
    """
    loader = _get_loader()
    model = loader.get_model()
    bundle = loader.get_explainer()
    assert bundle is not None  # load_initial builds the explainer alongside the model
    df = _build_input_frame(town, flat_type, floor_area_sqm, lease_commence_date, month)

    prediction = float(model.predict(df)[0])  # type: ignore[index]
    x_transformed = bundle.preprocessor.transform(df)  # type: ignore[attr-defined]
    shap_values = bundle.explainer.shap_values(x_transformed)
    # expected_value is a scalar for some tree models but a 1-element array for
    # GradientBoostingRegressor — .item() handles both safely.
    base_value = float(np.asarray(bundle.explainer.expected_value).item())
    contributions = dict(zip(bundle.feature_names, shap_values[0].tolist(), strict=False))

    top_contributors: list[FeatureContribution] = [
        {"feature": feature, "contribution": value}
        for feature, value in sorted(
            contributions.items(), key=lambda kv: abs(kv[1]), reverse=True
        )[:5]
    ]

    version = loader.get_version()
    assert version is not None
    return ExplanationResult(
        predicted_resale_price=round(prediction, 2),
        base_value=base_value,
        top_contributors=top_contributors,
        model_version=version,
        model_alias=_config.model_alias,
    )
