"""Pydantic v2 schemas for all API request and response types.

No raw dicts cross the API boundary — every endpoint uses one of these
models for both request parsing and response serialisation.
"""

from pydantic import BaseModel, Field


class HDBFeatureInput(BaseModel):
    """Feature fields shared by prediction and explanation requests.

    The serving layer derives flat_info and float_time_series from the natural
    fields supplied here, matching the preprocessing applied during training.
    """

    town: str = Field(..., examples=["ANG MO KIO"])
    flat_type: str = Field(..., examples=["3 ROOM"])
    flat_model: str = Field(..., examples=["Model A"])
    storey_range: str = Field(..., examples=["07 TO 09"])
    floor_area_sqm: float = Field(..., gt=0, examples=[67.0])
    lease_commence_date: int = Field(..., gt=1960, lt=2030, examples=[1986])
    month: str = Field(..., pattern=r"^\d{4}-\d{2}$", examples=["2024-01"])


class PredictRequest(HDBFeatureInput):
    """Input features for a single resale price prediction."""


class PredictResponse(BaseModel):
    """Prediction result with model provenance."""

    predicted_resale_price: float
    model_version: int
    model_alias: str


class ExplainRequest(HDBFeatureInput):
    """Input features for a SHAP explanation request.

    Schema matches PredictRequest; kept as a separate type so the response
    contract can evolve independently.
    """


class ExplainResponse(BaseModel):
    """SHAP explanation result with per-feature contributions and model provenance.

    The additivity property holds by construction for tree models:
    sum(feature_contributions.values()) + base_value ≈ predicted_resale_price.
    """

    predicted_resale_price: float
    base_value: float
    feature_contributions: dict[str, float]
    model_version: int
    model_alias: str


class HealthResponse(BaseModel):
    """Service health and model load status."""

    status: str
    model_loaded: bool
    model_version: int | None


class ModelInfoResponse(BaseModel):
    """Metadata about the currently loaded model version."""

    model_name: str
    model_alias: str
    model_version: int | None
    run_id: str | None
