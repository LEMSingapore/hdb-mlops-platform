"""Happy-path and key-branch tests for the real nodes.

The MCP-backed nodes (postal_lookup, predict, explain) call ``get_client()``;
each test replaces it with a fake whose async methods return canned payloads or
raise, so the node's own logic is exercised without the tool layer. The validate
node is pure Python and needs no double.
"""

from typing import Any

from ui.chat_app.graph.nodes import explain, lookup, predict, validate
from ui.chat_app.graph.state import GraphState

_READY_FIELDS: dict[str, Any] = {
    "town": "TAMPINES",
    "flat_type": "4 ROOM",
    "floor_area_sqm": 95.0,
    "lease_commence_date": 1985,
    "month": "2024-06",
}


class _FakeClient:
    """Async test double for MCPToolClient returning preset payloads."""

    def __init__(self, *, postal=None, prediction=None, explanation=None) -> None:
        self._postal = postal
        self._prediction = prediction
        self._explanation = explanation

    async def lookup_postal_code(self, postal_code):
        return self._postal

    async def predict_price(self, **kwargs):
        return self._prediction

    async def explain_prediction(self, **kwargs):
        return self._explanation


class _BoomClient:
    """Async test double whose every method raises, to drive the error branch."""

    async def lookup_postal_code(self, postal_code):
        raise RuntimeError("tool down")

    async def predict_price(self, **kwargs):
        raise RuntimeError("tool down")

    async def explain_prediction(self, **kwargs):
        raise RuntimeError("tool down")


class TestPostalLookupNode:
    async def test_resolves_town_from_result(self, monkeypatch) -> None:
        monkeypatch.setattr(lookup, "get_client", lambda: _FakeClient(postal={"town": "TAMPINES"}))
        state = GraphState(user_message="x", postal_code=522201)
        assert await lookup.run(state) == {"resolved_town": "TAMPINES"}

    async def test_no_postal_code_is_noop(self) -> None:
        state = GraphState(user_message="x")
        assert await lookup.run(state) == {}

    async def test_not_found_is_noop(self, monkeypatch) -> None:
        monkeypatch.setattr(lookup, "get_client", lambda: _FakeClient(postal=None))
        state = GraphState(user_message="x", postal_code=999999)
        assert await lookup.run(state) == {}

    async def test_null_town_in_result_is_noop(self, monkeypatch) -> None:
        monkeypatch.setattr(lookup, "get_client", lambda: _FakeClient(postal={"town": None}))
        state = GraphState(user_message="x", postal_code=522201)
        assert await lookup.run(state) == {}

    async def test_tool_failure_sets_error_status(self, monkeypatch) -> None:
        monkeypatch.setattr(lookup, "get_client", lambda: _BoomClient())
        state = GraphState(user_message="x", postal_code=522201)
        result = await lookup.run(state)
        assert result["status"] == "error"
        assert "tool down" in result["error"]


class TestValidateNode:
    async def test_all_fields_present_is_ready(self) -> None:
        state = GraphState(user_message="x", **_READY_FIELDS)
        assert await validate.run(state) == {
            "missing_fields": [],
            "status": "ready_to_predict",
        }

    async def test_resolved_town_satisfies_town_requirement(self) -> None:
        fields = {**_READY_FIELDS}
        del fields["town"]
        state = GraphState(user_message="x", resolved_town="TAMPINES", **fields)
        assert (await validate.run(state))["status"] == "ready_to_predict"

    async def test_missing_fields_listed_for_follow_up(self) -> None:
        state = GraphState(user_message="x", town="TAMPINES")
        result = await validate.run(state)
        assert result["status"] == "needs_follow_up"
        assert set(result["missing_fields"]) == {
            "flat_type",
            "floor_area_sqm",
            "lease_commence_date",
            "month",
        }

    async def test_error_status_passes_through(self) -> None:
        state = GraphState(user_message="x", status="error", error="boom")
        assert await validate.run(state) == {}


class TestPredictNode:
    async def test_records_price_and_version(self, monkeypatch) -> None:
        monkeypatch.setattr(
            predict,
            "get_client",
            lambda: _FakeClient(
                prediction={
                    "predicted_resale_price": 500_000.0,
                    "model_version": 7,
                    "model_alias": "champion",
                }
            ),
        )
        state = GraphState(user_message="x", status="ready_to_predict", **_READY_FIELDS)
        assert await predict.run(state) == {
            "predicted_price": 500_000.0,
            "model_version": 7,
        }

    async def test_skips_when_not_ready(self) -> None:
        state = GraphState(user_message="x", status="needs_follow_up")
        assert await predict.run(state) == {}

    async def test_tool_failure_sets_error_status(self, monkeypatch) -> None:
        monkeypatch.setattr(predict, "get_client", lambda: _BoomClient())
        state = GraphState(user_message="x", status="ready_to_predict", **_READY_FIELDS)
        result = await predict.run(state)
        assert result["status"] == "error"
        assert "tool down" in result["error"]


class TestExplainNode:
    async def test_records_contributors_and_base_value(self, monkeypatch) -> None:
        monkeypatch.setattr(
            explain,
            "get_client",
            lambda: _FakeClient(
                explanation={
                    "top_contributors": [{"feature": "Floor area", "contribution": 1.0}],
                    "base_value": 400_000.0,
                }
            ),
        )
        state = GraphState(
            user_message="x",
            status="ready_to_predict",
            predicted_price=500_000.0,
            **_READY_FIELDS,
        )
        result = await explain.run(state)
        assert result["base_value"] == 400_000.0
        assert result["top_contributors"][0]["feature"] == "Floor area"

    async def test_skips_without_a_prediction(self) -> None:
        state = GraphState(user_message="x", status="ready_to_predict", predicted_price=None)
        assert await explain.run(state) == {}

    async def test_skips_when_not_ready(self) -> None:
        state = GraphState(user_message="x", status="needs_follow_up", predicted_price=500_000.0)
        assert await explain.run(state) == {}
