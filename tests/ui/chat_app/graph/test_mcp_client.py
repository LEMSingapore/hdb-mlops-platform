"""Tests for the in-process MCP client wrapper.

predict and explain run against a StubModelLoader holding a fitted fixture
pipeline and a real SHAP bundle — same injection the MCP tool tests use — so the
wrapper exercises its full in-process Client path without an MLflow round-trip.
The postal lookup uses the committed lookup table, so it is deterministic
without a fixture.
"""

from typing import Any

import pytest

import mcp_server.tools.predict as predict_mod
from tests.conftest import (
    StubModelLoader,
    build_fixture_explainer_bundle,
    build_fixture_pipeline,
    make_synthetic_features,
)
from ui.chat_app.graph.mcp_client import MCPToolClient, get_client

_TAMPINES_4ROOM: dict[str, Any] = {
    "town": "TAMPINES",
    "flat_type": "4 ROOM",
    "floor_area_sqm": 95.0,
    "lease_commence_date": 1985,
    "month": "2024-06",
}


@pytest.fixture
def stub_loader(monkeypatch):
    """Inject a StubModelLoader with a fitted fixture pipeline and explainer."""
    X, y = make_synthetic_features(n=80)
    pipeline = build_fixture_pipeline()
    pipeline.fit(X, y)
    bundle = build_fixture_explainer_bundle(pipeline)
    stub = StubModelLoader(model=pipeline, version=7, explainer=bundle)
    monkeypatch.setattr(predict_mod, "_loader", stub)
    return stub


class TestPredictPrice:
    async def test_returns_structured_prediction(self, stub_loader) -> None:
        client = MCPToolClient()
        result = await client.predict_price(**_TAMPINES_4ROOM)
        assert result["model_version"] == 7
        assert result["model_alias"] == "champion"
        assert 100_000 < result["predicted_resale_price"] < 1_000_000


class TestExplainPrediction:
    async def test_returns_contributors_and_base_value(self, stub_loader) -> None:
        client = MCPToolClient()
        result = await client.explain_prediction(**_TAMPINES_4ROOM)
        assert isinstance(result["base_value"], float)
        # The fixture's nine raw dummies aggregate to five source features.
        assert len(result["top_contributors"]) == 5
        for entry in result["top_contributors"]:
            assert set(entry.keys()) == {"feature", "contribution"}


class TestLookupPostalCode:
    async def test_resolves_known_postal_to_town(self) -> None:
        client = MCPToolClient()
        result = await client.lookup_postal_code(522201)
        assert result is not None
        assert result["town"] == "TAMPINES"
        assert result["street_full"] == "TAMPINES STREET 21"

    async def test_unknown_postal_returns_none(self) -> None:
        client = MCPToolClient()
        assert await client.lookup_postal_code(999999) is None


class TestSingleton:
    def test_get_client_returns_same_instance(self) -> None:
        assert get_client() is get_client()
