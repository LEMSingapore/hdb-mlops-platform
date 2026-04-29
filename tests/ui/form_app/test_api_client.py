"""Tests for APIClient using a mock httpx transport.

The API client is tested in isolation — no FastAPI process is started.
A _MockTransport intercepts requests and returns scripted responses,
exercising the happy paths and every error-mapping branch.
"""

import httpx
import pytest

from ui.form_app.api_client import (
    APIClient,
    APIConnectionError,
    ServerError,
    ServiceUnavailableError,
    ValidationError,
)
from ui.form_app.config import UIConfig

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_PAYLOAD: dict = {
    "town": "TAMPINES",
    "flat_type": "4 ROOM",
    "flat_model": "Model A",
    "storey_range": "07 TO 09",
    "floor_area_sqm": 90.0,
    "lease_commence_date": 1990,
    "month": "2024-06",
}

_PREDICT_BODY: dict = {
    "predicted_resale_price": 607471.0,
    "model_version": 1,
    "model_alias": "champion",
}

_EXPLAIN_BODY: dict = {
    "predicted_resale_price": 607471.0,
    "base_value": 450000.0,
    "feature_contributions": {
        "num__floor_area_sqm": 50000.0,
        "cat__town_TAMPINES": 107471.0,
    },
    "model_version": 1,
    "model_alias": "champion",
}


class _MockTransport(httpx.BaseTransport):
    """Synchronous transport that delegates to a handler function.

    The handler receives the outgoing Request and must return a Response.
    Raising an exception from the handler propagates it to the caller.
    """

    def __init__(self, handler):
        self._handler = handler

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self._handler(request)


def _make_client(handler) -> APIClient:
    config = UIConfig(api_base_url="http://test-api:8000", request_timeout_seconds=5)
    return APIClient(config, transport=_MockTransport(handler))


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_predict_returns_parsed_json():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/predict"
        return httpx.Response(200, json=_PREDICT_BODY)

    result = _make_client(handler).predict(_PAYLOAD)
    assert result["predicted_resale_price"] == pytest.approx(607471.0)
    assert result["model_alias"] == "champion"
    assert result["model_version"] == 1


def test_explain_returns_parsed_json():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/explain"
        return httpx.Response(200, json=_EXPLAIN_BODY)

    result = _make_client(handler).explain(_PAYLOAD)
    assert "feature_contributions" in result
    assert result["base_value"] == pytest.approx(450000.0)
    assert result["predicted_resale_price"] == pytest.approx(607471.0)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_503_raises_service_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "Model not loaded yet — try again shortly"})

    with pytest.raises(ServiceUnavailableError):
        _make_client(handler).predict(_PAYLOAD)


def test_422_raises_validation_error_with_field_map():
    body = {
        "detail": [
            {
                "loc": ["body", "floor_area_sqm"],
                "msg": "Input should be greater than 0",
                "type": "greater_than",
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json=body)

    with pytest.raises(ValidationError) as exc_info:
        _make_client(handler).predict(_PAYLOAD)

    assert "floor_area_sqm" in exc_info.value.errors
    assert exc_info.value.errors["floor_area_sqm"] == "Input should be greater than 0"


def test_500_raises_server_error_with_detail():
    body = {"detail": "Prediction failed."}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json=body)

    with pytest.raises(ServerError) as exc_info:
        _make_client(handler).predict(_PAYLOAD)

    assert "Prediction failed" in str(exc_info.value.detail)


def test_network_failure_raises_connection_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    with pytest.raises(APIConnectionError) as exc_info:
        _make_client(handler).predict(_PAYLOAD)

    assert "http://test-api:8000" in exc_info.value.url
