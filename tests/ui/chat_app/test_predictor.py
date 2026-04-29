"""Tests for ui.chat_app.predictor using mock HTTP transport and mock lookup.

HTTP calls are intercepted by a mock httpx transport injected via
``unittest.mock.patch``. The postal lookup is mocked with ``pytest.monkeypatch``
to avoid loading the lookup CSV in every test.
"""

import json
from unittest.mock import patch

import httpx
import pytest

from lookup.postal import AddressInfo
from ui.chat_app.predictor import (
    APIConnectionError,
    ServerError,
    ServiceUnavailableError,
    ValidationError,
    explain_price,
    lookup_postal,
    predict_price,
)
from ui.form_app.api_client import APIClient
from ui.form_app.config import UIConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FIELDS = dict(
    town="TAMPINES",
    flat_type="4 ROOM",
    floor_area_sqm=92.0,
    lease_commence_date=1990,
    month="2025-04",
)

_PREDICT_BODY = {
    "predicted_resale_price": 621000.0,
    "model_version": 4,
    "model_alias": "champion",
}

_EXPLAIN_BODY = {
    "predicted_resale_price": 621000.0,
    "base_value": 440000.0,
    "feature_contributions": {
        "num__floor_area_sqm": 55000.0,
        "cat__town_TAMPINES": 90000.0,
        "cat__flat_type_4 ROOM": 40000.0,
        "num__lease_commence_date": -8000.0,
        "num__month_float": -4000.0,
    },
    "model_version": 4,
    "model_alias": "champion",
}


class _MockTransport(httpx.BaseTransport):
    def __init__(self, handler):
        self._handler = handler

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self._handler(request)


def _client_with(handler) -> APIClient:
    config = UIConfig(api_base_url="http://test-api:8000", request_timeout_seconds=5)
    return APIClient(config, transport=_MockTransport(handler))


# ---------------------------------------------------------------------------
# predict_price
# ---------------------------------------------------------------------------


def test_predict_price_posts_correct_body_to_predict_endpoint():
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_PREDICT_BODY)

    with patch("ui.chat_app.predictor._client", _client_with(handler)):
        predict_price(**_FIELDS)

    assert len(captured) == 1
    assert captured[0].url.path == "/predict"
    body = json.loads(captured[0].content)
    assert body["town"] == "TAMPINES"
    assert body["flat_type"] == "4 ROOM"
    assert body["floor_area_sqm"] == pytest.approx(92.0)
    assert body["lease_commence_date"] == 1990
    assert body["month"] == "2025-04"


def test_predict_price_returns_full_response_dict():
    with patch(
        "ui.chat_app.predictor._client",
        _client_with(lambda _: httpx.Response(200, json=_PREDICT_BODY)),
    ):
        result = predict_price(**_FIELDS)

    assert result["predicted_resale_price"] == pytest.approx(621000.0)
    assert result["model_version"] == 4
    assert result["model_alias"] == "champion"


# ---------------------------------------------------------------------------
# explain_price
# ---------------------------------------------------------------------------


def test_explain_price_posts_to_explain_endpoint():
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_EXPLAIN_BODY)

    with patch("ui.chat_app.predictor._client", _client_with(handler)):
        explain_price(**_FIELDS)

    assert captured[0].url.path == "/explain"


def test_explain_price_returns_top_three_contributors():
    with patch(
        "ui.chat_app.predictor._client",
        _client_with(lambda _: httpx.Response(200, json=_EXPLAIN_BODY)),
    ):
        result = explain_price(**_FIELDS)

    assert len(result["top_contributors"]) == 3


def test_explain_price_contributors_sorted_by_absolute_value_descending():
    with patch(
        "ui.chat_app.predictor._client",
        _client_with(lambda _: httpx.Response(200, json=_EXPLAIN_BODY)),
    ):
        result = explain_price(**_FIELDS)

    contributions = [abs(c["contribution"]) for c in result["top_contributors"]]
    assert contributions == sorted(contributions, reverse=True)


def test_explain_price_top_contributor_is_highest_abs_value():
    """cat__town_TAMPINES (90000) should rank first, above floor_area (55000)."""
    with patch(
        "ui.chat_app.predictor._client",
        _client_with(lambda _: httpx.Response(200, json=_EXPLAIN_BODY)),
    ):
        result = explain_price(**_FIELDS)

    assert result["top_contributors"][0]["feature"] == "cat__town_TAMPINES"


def test_explain_price_result_includes_predicted_price_base_value_model_version():
    with patch(
        "ui.chat_app.predictor._client",
        _client_with(lambda _: httpx.Response(200, json=_EXPLAIN_BODY)),
    ):
        result = explain_price(**_FIELDS)

    assert result["predicted_resale_price"] == pytest.approx(621000.0)
    assert result["base_value"] == pytest.approx(440000.0)
    assert result["model_version"] == 4


# ---------------------------------------------------------------------------
# lookup_postal
# ---------------------------------------------------------------------------


def test_lookup_postal_returns_address_info_for_known_code(monkeypatch):
    expected = AddressInfo(
        postal_code="528003",
        block="137",
        street_abbreviated="TAMPINES ST 11",
        street_full="TAMPINES STREET 11",
        town="TAMPINES",
    )
    monkeypatch.setattr("ui.chat_app.predictor._lookup_postal", lambda _: expected)

    result = lookup_postal(528003)

    assert result is not None
    assert result.town == "TAMPINES"
    assert result.postal_code == "528003"


def test_lookup_postal_returns_none_for_unknown_code(monkeypatch):
    monkeypatch.setattr("ui.chat_app.predictor._lookup_postal", lambda _: None)

    result = lookup_postal(999999)

    assert result is None


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


def test_predict_price_propagates_service_unavailable():
    mock_client = _client_with(lambda _: httpx.Response(503, json={}))
    with (
        patch("ui.chat_app.predictor._client", mock_client),
        pytest.raises(ServiceUnavailableError),
    ):
        predict_price(**_FIELDS)


def test_predict_price_propagates_validation_error():
    body = {
        "detail": [{"loc": ["body", "floor_area_sqm"], "msg": "Input should be greater than 0"}]
    }
    with (
        patch(
            "ui.chat_app.predictor._client",
            _client_with(lambda _: httpx.Response(422, json=body)),
        ),
        pytest.raises(ValidationError),
    ):
        predict_price(**_FIELDS)


def test_predict_price_propagates_server_error():
    with (
        patch(
            "ui.chat_app.predictor._client",
            _client_with(lambda _: httpx.Response(500, json={"detail": "boom"})),
        ),
        pytest.raises(ServerError),
    ):
        predict_price(**_FIELDS)


def test_predict_price_propagates_connection_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with (
        patch("ui.chat_app.predictor._client", _client_with(handler)),
        pytest.raises(APIConnectionError),
    ):
        predict_price(**_FIELDS)
