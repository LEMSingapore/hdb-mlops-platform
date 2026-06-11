"""Routing tests for the compiled graph's conditional paths.

These tests assert how the graph routes — which nodes run and what terminal
status results — for each branch off the parse and validate decisions. The LLM
nodes (parse, narrate) are replaced with deterministic doubles so the assertions
turn on routing, not LLM phrasing; the LLM behaviour itself is unit-tested in
``test_nodes``. The MCP-backed nodes get fake clients so no model or SQLite layer
is needed.

Patching ``parse.run`` / ``narrate.run`` and the per-node ``get_client`` must
happen before ``build_graph``, because the compiled graph captures whatever those
names point to at build time.
"""

from typing import Any

import pytest

from ui.chat_app.graph import GraphState, build_graph
from ui.chat_app.graph.nodes import explain as explain_node
from ui.chat_app.graph.nodes import lookup as lookup_node
from ui.chat_app.graph.nodes import narrate as narrate_node
from ui.chat_app.graph.nodes import parse as parse_node
from ui.chat_app.graph.nodes import predict as predict_node

_OTHER_FIELDS: dict[str, Any] = {
    "flat_type": "4 ROOM",
    "floor_area_sqm": 95.0,
    "lease_commence_date": 1985,
    "month": "2024-06",
}
_PREDICTION = {"predicted_resale_price": 586_888.0, "model_version": 7, "model_alias": "champion"}
_EXPLANATION = {
    "top_contributors": [{"feature": "Floor area (95 sqm)", "contribution": 40_000.0}],
    "base_value": 400_000.0,
}


class _FakeClient:
    """Async test double for MCPToolClient; records the predict kwargs it sees."""

    def __init__(self, *, postal=None, prediction=None, explanation=None) -> None:
        self._postal = postal
        self._prediction = prediction
        self._explanation = explanation
        self.predict_kwargs: dict[str, Any] | None = None

    async def lookup_postal_code(self, postal_code):
        return self._postal

    async def predict_price(self, **kwargs):
        self.predict_kwargs = kwargs
        return self._prediction

    async def explain_prediction(self, **kwargs):
        return self._explanation


async def _deterministic_narrate(state: GraphState) -> dict:
    """Stand-in narrate node: a fixed phrasing per terminal status."""
    if state.status == "needs_follow_up":
        return {"response_text": "Please provide: " + ", ".join(state.missing_fields)}
    if state.status == "out_of_scope":
        return {"response_text": f"Sorry, out of scope: {state.parse_reasoning}"}
    if state.status == "error":
        return {"response_text": "Sorry, something went wrong. Please try again."}
    return {"response_text": f"Predicted S${state.predicted_price:,.0f}"}


def _stub_parse(monkeypatch, fields: dict[str, Any]) -> None:
    async def parse(state: GraphState) -> dict:
        return fields

    monkeypatch.setattr(parse_node, "run", parse)


@pytest.fixture(autouse=True)
def _stub_narrate(monkeypatch):
    """Every branch test uses the deterministic narrate double."""
    monkeypatch.setattr(narrate_node, "run", _deterministic_narrate)


class TestMissingFieldsPath:
    async def test_partial_extraction_routes_to_follow_up(self, monkeypatch) -> None:
        _stub_parse(monkeypatch, {"town": "QUEENSTOWN", "flat_type": "4 ROOM", "status": "pending"})
        graph = build_graph()
        result = await graph.ainvoke(GraphState(user_message="4 room in queenstown"))
        state = GraphState(**result)

        assert state.status == "needs_follow_up"
        assert state.predicted_price is None
        assert set(state.missing_fields) == {"floor_area_sqm", "lease_commence_date", "month"}
        assert state.response_text
        assert "floor_area_sqm" in state.response_text


class TestOutOfScopePath:
    async def test_short_circuits_to_decline(self, monkeypatch) -> None:
        _stub_parse(
            monkeypatch,
            {"status": "out_of_scope", "parse_reasoning": "condominiums are not HDB flats"},
        )
        graph = build_graph()
        result = await graph.ainvoke(GraphState(user_message="3 bedroom condo in newton"))
        state = GraphState(**result)

        assert state.status == "out_of_scope"
        assert state.predicted_price is None
        assert state.missing_fields == []
        assert "out of scope" in state.response_text


class TestPredictToolFailurePath:
    async def test_prediction_fault_becomes_error(self, monkeypatch) -> None:
        _stub_parse(
            monkeypatch,
            {"town": "TAMPINES", "status": "pending", **_OTHER_FIELDS},
        )

        class _Boom(_FakeClient):
            async def predict_price(self, **kwargs):
                raise RuntimeError("predict tool down")

        monkeypatch.setattr(predict_node, "get_client", lambda: _Boom())
        graph = build_graph()
        result = await graph.ainvoke(GraphState(user_message="tampines 4 room ..."))
        state = GraphState(**result)

        assert state.status == "error"
        assert state.predicted_price is None
        assert state.error is not None and "predict tool down" in state.error
        assert state.response_text == "Sorry, something went wrong. Please try again."


class TestPostalLookupMissPath:
    async def test_unresolved_postal_with_no_town_needs_follow_up(self, monkeypatch) -> None:
        # Postal code given but no explicit town; the lookup returns None.
        _stub_parse(
            monkeypatch,
            {"postal_code": 999999, "status": "pending", **_OTHER_FIELDS},
        )
        monkeypatch.setattr(lookup_node, "get_client", lambda: _FakeClient(postal=None))
        graph = build_graph()
        result = await graph.ainvoke(GraphState(user_message="4 room, postal 999999"))
        state = GraphState(**result)

        assert state.status == "needs_follow_up"
        assert state.resolved_town is None
        assert state.missing_fields == ["town"]


class TestPostalLookupSuccessPath:
    async def test_resolved_town_drives_the_prediction(self, monkeypatch) -> None:
        # Postal code only, no explicit town; the lookup resolves it to Bedok.
        _stub_parse(
            monkeypatch,
            {"postal_code": 460001, "status": "pending", **_OTHER_FIELDS},
        )
        lookup_client = _FakeClient(postal={"town": "BEDOK"})
        predict_client = _FakeClient(prediction=_PREDICTION, explanation=_EXPLANATION)
        monkeypatch.setattr(lookup_node, "get_client", lambda: lookup_client)
        monkeypatch.setattr(predict_node, "get_client", lambda: predict_client)
        monkeypatch.setattr(explain_node, "get_client", lambda: predict_client)
        graph = build_graph()
        result = await graph.ainvoke(GraphState(user_message="4 room, postal 460001"))
        state = GraphState(**result)

        assert state.status == "ready_to_predict"
        assert state.resolved_town == "BEDOK"
        assert state.predicted_price == 586_888.0
        # The prediction used the looked-up town, not a null explicit town.
        assert predict_client.predict_kwargs is not None
        assert predict_client.predict_kwargs["town"] == "BEDOK"
