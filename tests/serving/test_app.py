"""Tests for FastAPI endpoints: /predict, /explain, /health, /model-info.

Uses httpx.AsyncClient with ASGITransport (no lifespan triggered) and
monkeypatches serving.app.loader with a StubModelLoader. This keeps the
tests free of MLflow network calls while still exercising the full request
path including Pydantic validation and response serialisation.
"""

from unittest.mock import Mock

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

import serving.app as app_module
from tests.conftest import (
    StubModelLoader,
    build_fixture_explainer_bundle,
    build_fixture_pipeline,
    make_synthetic_features,
)

_TAMPINES_4ROOM = {
    "town": "TAMPINES",
    "flat_type": "4 ROOM",
    "flat_model": "Model A",
    "storey_range": "07 TO 09",
    "floor_area_sqm": 93.0,
    "lease_commence_date": 1990,
    "month": "2024-01",
}


def _mock_model(prediction: float = 450_000.0) -> Mock:
    m = Mock()
    m.predict.return_value = np.array([prediction])
    return m


@pytest.fixture
async def loaded_client(monkeypatch):
    """AsyncClient backed by a StubModelLoader with a loaded model."""
    stub = StubModelLoader(
        model=_mock_model(),
        version="42",
        run_id="test-run-id-cafe",
    )
    monkeypatch.setattr(app_module, "loader", stub)
    async with AsyncClient(
        transport=ASGITransport(app=app_module.app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture
async def unloaded_client(monkeypatch):
    """AsyncClient backed by a StubModelLoader with no model loaded.

    get_model() raises RuntimeError (as the real loader does), which FastAPI
    surfaces as HTTP 500. get_version() returns None, making /health report
    model_loaded=false and /predict return 503 when get_model() is bypassed.
    """
    stub = StubModelLoader(model=None, version=None, run_id=None)
    monkeypatch.setattr(app_module, "loader", stub)
    async with AsyncClient(
        transport=ASGITransport(app=app_module.app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture
async def version_unknown_client(monkeypatch):
    """AsyncClient where get_model() succeeds but get_version() returns None.

    This exercises the explicit 503 branch in /predict.
    """
    stub = StubModelLoader(model=_mock_model(), version=None, run_id=None)
    monkeypatch.setattr(app_module, "loader", stub)
    async with AsyncClient(
        transport=ASGITransport(app=app_module.app), base_url="http://test"
    ) as client:
        yield client


class TestHealthEndpoint:
    async def test_health_returns_200_with_model_loaded(self, loaded_client):
        resp = await loaded_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert isinstance(body["model_version"], int)
        assert body["model_version"] == 42

    async def test_health_returns_200_with_model_not_loaded(self, unloaded_client):
        resp = await unloaded_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is False
        assert body["model_version"] is None


class TestModelInfoEndpoint:
    async def test_model_info_returns_200_with_all_fields(self, loaded_client):
        resp = await loaded_client.get("/model-info")
        assert resp.status_code == 200
        body = resp.json()
        assert body["model_name"] == "hdb-predictor"
        assert body["model_alias"] == "champion"
        assert isinstance(body["model_version"], int)
        assert body["model_version"] == 42
        assert body["run_id"] == "test-run-id-cafe"

    async def test_model_info_returns_nulls_when_not_loaded(self, unloaded_client):
        resp = await unloaded_client.get("/model-info")
        assert resp.status_code == 200
        body = resp.json()
        assert body["model_version"] is None
        assert body["run_id"] is None


class TestPredictEndpoint:
    async def test_predict_happy_path_tampines_4room(self, loaded_client):
        resp = await loaded_client.post("/predict", json=_TAMPINES_4ROOM)
        assert resp.status_code == 200
        body = resp.json()
        assert 100_000 < body["predicted_resale_price"] < 2_000_000
        assert isinstance(body["model_version"], int)
        assert body["model_alias"] == "champion"

    async def test_predict_returns_503_when_version_not_loaded(self, version_unknown_client):
        resp = await version_unknown_client.post("/predict", json=_TAMPINES_4ROOM)
        assert resp.status_code == 503

    async def test_predict_returns_422_on_negative_floor_area(self, loaded_client):
        payload = {**_TAMPINES_4ROOM, "floor_area_sqm": -5.0}
        resp = await loaded_client.post("/predict", json=payload)
        assert resp.status_code == 422

    async def test_predict_returns_422_on_zero_floor_area(self, loaded_client):
        payload = {**_TAMPINES_4ROOM, "floor_area_sqm": 0.0}
        resp = await loaded_client.post("/predict", json=payload)
        assert resp.status_code == 422

    async def test_predict_returns_422_on_bad_month_format_single_digit(self, loaded_client):
        payload = {**_TAMPINES_4ROOM, "month": "2024-6"}
        resp = await loaded_client.post("/predict", json=payload)
        assert resp.status_code == 422

    async def test_predict_returns_422_on_bad_month_format_two_digit_year(self, loaded_client):
        payload = {**_TAMPINES_4ROOM, "month": "24-06"}
        resp = await loaded_client.post("/predict", json=payload)
        assert resp.status_code == 422

    async def test_predict_returns_422_on_invalid_lease_year(self, loaded_client):
        payload = {**_TAMPINES_4ROOM, "lease_commence_date": 1959}
        resp = await loaded_client.post("/predict", json=payload)
        assert resp.status_code == 422

    async def test_predict_model_version_is_int_in_response(self, loaded_client):
        resp = await loaded_client.post("/predict", json=_TAMPINES_4ROOM)
        assert resp.status_code == 200
        assert isinstance(resp.json()["model_version"], int)


@pytest.fixture
def _fitted_pipeline():
    """A fitted sklearn pipeline for SHAP tests — built once, shared within the module."""
    X, y = make_synthetic_features(n=120)
    pipeline = build_fixture_pipeline()
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture
async def explain_client(monkeypatch, _fitted_pipeline):
    """AsyncClient backed by a real sklearn pipeline and SHAP bundle.

    Both the model and the explainer are derived from the same fitted pipeline
    so the SHAP additivity property holds — predicted_resale_price from
    model.predict equals base_value + sum(feature_contributions).
    """
    bundle = build_fixture_explainer_bundle(_fitted_pipeline)
    stub = StubModelLoader(
        model=_fitted_pipeline,
        version="42",
        run_id="test-run-shap",
        explainer=bundle,
    )
    monkeypatch.setattr(app_module, "loader", stub)
    async with AsyncClient(
        transport=ASGITransport(app=app_module.app), base_url="http://test"
    ) as client:
        yield client


class TestExplainEndpoint:
    async def test_explain_happy_path_tampines_4room_returns_200(self, explain_client):
        resp = await explain_client.post("/explain", json=_TAMPINES_4ROOM)
        assert resp.status_code == 200
        body = resp.json()
        assert "predicted_resale_price" in body
        assert "base_value" in body
        assert "feature_contributions" in body
        assert "model_version" in body
        assert "model_alias" in body
        assert isinstance(body["feature_contributions"], dict)
        assert len(body["feature_contributions"]) > 0
        assert 100_000 < body["predicted_resale_price"] < 2_000_000

    async def test_explain_all_response_fields_are_populated(self, explain_client):
        resp = await explain_client.post("/explain", json=_TAMPINES_4ROOM)
        body = resp.json()
        assert body["model_version"] == 42
        assert body["model_alias"] == "champion"
        assert isinstance(body["base_value"], float)
        assert all(isinstance(v, float) for v in body["feature_contributions"].values())

    async def test_explain_additivity_holds_within_epsilon(self, explain_client):
        """sum(contributions) + base_value ≈ predicted_resale_price within 1.0 SGD."""
        resp = await explain_client.post("/explain", json=_TAMPINES_4ROOM)
        body = resp.json()
        shap_total = sum(body["feature_contributions"].values()) + body["base_value"]
        assert abs(shap_total - body["predicted_resale_price"]) < 1.0

    async def test_explain_returns_503_when_version_not_loaded(self, version_unknown_client):
        resp = await version_unknown_client.post("/explain", json=_TAMPINES_4ROOM)
        assert resp.status_code == 503

    async def test_explain_returns_422_on_negative_floor_area(self, explain_client):
        payload = {**_TAMPINES_4ROOM, "floor_area_sqm": -1.0}
        resp = await explain_client.post("/explain", json=payload)
        assert resp.status_code == 422

    async def test_explain_returns_422_on_bad_month_format(self, explain_client):
        payload = {**_TAMPINES_4ROOM, "month": "2024-6"}
        resp = await explain_client.post("/explain", json=payload)
        assert resp.status_code == 422

    async def test_explain_returns_422_on_invalid_lease_year(self, explain_client):
        payload = {**_TAMPINES_4ROOM, "lease_commence_date": 1959}
        resp = await explain_client.post("/explain", json=payload)
        assert resp.status_code == 422
