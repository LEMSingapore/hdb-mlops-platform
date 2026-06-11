"""Happy-path tests for predict_price and explain_prediction.

These inject a StubModelLoader holding a small fitted fixture pipeline and a
real SHAP ExplainerBundle, so the tools exercise their full path (real
prediction, real SHAP values) without an MLflow round-trip or the production
v7 model.
"""

import pytest
from fastmcp import Client

import mcp_server.tools.predict as predict_mod
from mcp_server.server import mcp
from tests.conftest import (
    StubModelLoader,
    build_fixture_explainer_bundle,
    build_fixture_pipeline,
    make_synthetic_features,
)

_TAMPINES_4ROOM = {
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
    async def test_happy_path_returns_structured_prediction(self, stub_loader) -> None:
        async with Client(mcp) as client:
            result = await client.call_tool("predict_price", _TAMPINES_4ROOM)
        data = result.structured_content
        assert data["model_version"] == 7
        assert data["model_alias"] == "champion"
        # Synthetic targets span 350k-650k, so any sane prediction lands in range.
        assert 100_000 < data["predicted_resale_price"] < 1_000_000

    async def test_price_is_rounded_to_two_decimals(self, stub_loader) -> None:
        async with Client(mcp) as client:
            result = await client.call_tool("predict_price", _TAMPINES_4ROOM)
        price = result.structured_content["predicted_resale_price"]
        assert round(price, 2) == price


class TestExplainPrediction:
    async def test_happy_path_returns_top_five_contributors(self, stub_loader) -> None:
        async with Client(mcp) as client:
            result = await client.call_tool("explain_prediction", _TAMPINES_4ROOM)
        data = result.structured_content
        assert data["model_version"] == 7
        assert isinstance(data["base_value"], float)
        # The fixture pipeline's nine raw dummies aggregate to five features —
        # two active categoricals plus three non-categorical features — so the
        # top five returns one entry per source feature.
        assert len(data["top_contributors"]) == 5
        for entry in data["top_contributors"]:
            assert set(entry.keys()) == {"feature", "contribution"}

    async def test_top_contributors_have_one_entry_per_categorical(self, stub_loader) -> None:
        async with Client(mcp) as client:
            result = await client.call_tool("explain_prediction", _TAMPINES_4ROOM)
        labels = [entry["feature"] for entry in result.structured_content["top_contributors"]]
        # One-hot dummies are aggregated, so a categorical column never appears
        # more than once in the human-facing view.
        assert sum(label.startswith("Town:") for label in labels) <= 1
        assert sum(label.startswith("Flat type:") for label in labels) <= 1

    async def test_top_contributors_name_active_category_not_inactive(self, stub_loader) -> None:
        async with Client(mcp) as client:
            result = await client.call_tool("explain_prediction", _TAMPINES_4ROOM)
        labels = [entry["feature"] for entry in result.structured_content["top_contributors"]]
        # The active categories for this row are present; inactive dummies that
        # the raw top-N used to surface (e.g. a 3-room entry for a 4-room flat)
        # are not.
        assert "Town: Tampines" in labels
        assert "Flat type: 4-room" in labels
        town_labels = [label for label in labels if label.startswith("Town:")]
        flat_labels = [label for label in labels if label.startswith("Flat type:")]
        assert town_labels == ["Town: Tampines"]
        assert flat_labels == ["Flat type: 4-room"]

    async def test_top_contributors_carry_human_readable_labels(self, stub_loader) -> None:
        async with Client(mcp) as client:
            result = await client.call_tool("explain_prediction", _TAMPINES_4ROOM)
        top = result.structured_content["top_contributors"]
        for entry in top:
            label = entry["feature"]
            # A presentation label reads naturally: it never exposes a raw
            # transformer prefix, and every handled case carries a space or colon.
            assert not label.startswith(("cat__", "num__", "month__"))
            assert " " in label or ":" in label

    async def test_feature_contributions_dict_keeps_raw_keys(self, stub_loader) -> None:
        async with Client(mcp) as client:
            result = await client.call_tool("explain_prediction", _TAMPINES_4ROOM)
        contributions = result.structured_content["feature_contributions"]
        # The full per-dummy raw view stays available for programmatic
        # consumers — including inactive dummies the aggregation collapses out
        # of top_contributors (this row is a 4-room, yet the 3-room dummy
        # remains here).
        assert "num__floor_area_sqm" in contributions
        assert "cat__flat_type_3 ROOM" in contributions
        assert all(key.startswith(("cat__", "num__", "month__")) for key in contributions)
        assert all(isinstance(value, float) for value in contributions.values())

    async def test_contributors_sorted_by_absolute_value_descending(self, stub_loader) -> None:
        async with Client(mcp) as client:
            result = await client.call_tool("explain_prediction", _TAMPINES_4ROOM)
        contributions = [c["contribution"] for c in result.structured_content["top_contributors"]]
        magnitudes = [abs(c) for c in contributions]
        assert magnitudes == sorted(magnitudes, reverse=True)

    async def test_prediction_matches_predict_price(self, stub_loader) -> None:
        async with Client(mcp) as client:
            predict = await client.call_tool("predict_price", _TAMPINES_4ROOM)
            explain = await client.call_tool("explain_prediction", _TAMPINES_4ROOM)
        assert (
            explain.structured_content["predicted_resale_price"]
            == predict.structured_content["predicted_resale_price"]
        )
