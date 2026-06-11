"""End-to-end graph traversal tests.

The happy path runs the compiled graph against the real MCP predict and explain
tools, with a StubModelLoader standing in for the @champion model so the test is
hermetic. The postal lookup hits the real (committed) lookup table; the stub
parse supplies postal 528003, which is absent from that table, so the lookup
no-ops and the explicit town carries the prediction. The missing-fields path
swaps in a parse stub that returns only a partial extraction and asserts the
graph short-circuits past prediction straight to narration.
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


class TestHappyPath:
    async def test_full_traversal_predicts_and_narrates(self, stub_loader) -> None:
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

        # Patch before build_graph so the compiled graph captures the stub.
        monkeypatch.setattr(parse_node, "run", partial_parse)
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
