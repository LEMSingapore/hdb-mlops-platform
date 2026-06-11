"""End-to-end graph traversal tests.

The happy path runs the compiled graph against the real MCP predict and explain
tools, with a StubModelLoader standing in for the @champion model so the test is
hermetic. The parse and narrate nodes make live Anthropic calls in production, so
both are stubbed here with deterministic doubles — these tests exercise the graph
wiring and the MCP tool layer, not the LLM phrasing, which is unit-tested in
``test_nodes`` with a mocked client. The missing-fields path swaps in a parse
stub that returns only a partial extraction and asserts the graph short-circuits
past prediction straight to narration.
"""

import pytest

import mcp_server.tools.predict as predict_mod
from tests.conftest import (
    StubModelLoader,
    build_fixture_explainer_bundle,
    build_fixture_pipeline,
    make_synthetic_features,
)
from ui.chat_app.graph import GraphState, build_graph
from ui.chat_app.graph.nodes import narrate as narrate_node
from ui.chat_app.graph.nodes import parse as parse_node


@pytest.fixture
def stub_loader(monkeypatch):
    """Inject a StubModelLoader so the MCP predict/explain tools need no MLflow."""
    X, y = make_synthetic_features(n=80)
    pipeline = build_fixture_pipeline()
    pipeline.fit(X, y)
    bundle = build_fixture_explainer_bundle(pipeline)
    stub = StubModelLoader(model=pipeline, version=7, explainer=bundle)
    monkeypatch.setattr(predict_mod, "_loader", stub)
    return stub


async def _canonical_parse(state: GraphState) -> dict:
    """Deterministic stand-in for the LLM parse node: the standing test input."""
    return {
        "town": "TAMPINES",
        "flat_type": "3 ROOM",
        "floor_area_sqm": 95.0,
        "lease_commence_date": 1985,
        "month": "2024-06",
        "postal_code": 528003,
        "status": "pending",
    }


async def _deterministic_narrate(state: GraphState) -> dict:
    """Deterministic stand-in for the LLM narrate node, echoing the prediction."""
    return {
        "response_text": (
            f"Predicted price: S${state.predicted_price:,.0f} (model v{state.model_version})"
        )
    }


class TestHappyPath:
    async def test_full_traversal_predicts_and_narrates(self, stub_loader, monkeypatch) -> None:
        # Patch the LLM nodes before build_graph so the compiled graph captures them.
        monkeypatch.setattr(parse_node, "run", _canonical_parse)
        monkeypatch.setattr(narrate_node, "run", _deterministic_narrate)
        graph = build_graph()
        result = await graph.ainvoke(
            GraphState(user_message="3 room flat in Tampines, postal 528003")
        )
        state = GraphState(**result)

        assert state.status == "ready_to_predict"
        assert state.predicted_price is not None
        assert 100_000 < state.predicted_price < 1_000_000
        assert state.model_version == 7
        assert state.base_value is not None
        assert state.top_contributors
        assert state.error is None
        assert "Predicted price" in state.response_text


class TestMissingFieldsPath:
    async def test_short_circuits_to_narrate(self, monkeypatch) -> None:
        async def partial_parse(state: GraphState) -> dict:
            return {"town": "TAMPINES", "flat_type": "4 ROOM"}

        async def deterministic_narrate(state: GraphState) -> dict:
            return {"response_text": "Please provide: " + ", ".join(state.missing_fields)}

        # Patch before build_graph so the compiled graph captures the stubs.
        monkeypatch.setattr(parse_node, "run", partial_parse)
        monkeypatch.setattr(narrate_node, "run", deterministic_narrate)
        graph = build_graph()
        result = await graph.ainvoke(GraphState(user_message="tampines 4 room"))
        state = GraphState(**result)

        assert state.status == "needs_follow_up"
        assert state.predicted_price is None
        assert set(state.missing_fields) == {
            "floor_area_sqm",
            "lease_commence_date",
            "month",
        }
        assert state.response_text.startswith("Please provide")
