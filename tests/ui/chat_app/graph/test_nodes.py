"""Happy-path and key-branch tests for the real nodes.

The MCP-backed nodes (postal_lookup, predict, explain) call ``get_client()``;
each test replaces it with a fake whose async methods return canned payloads or
raise, so the node's own logic is exercised without the tool layer. The validate
node is pure Python and needs no double. The LLM-backed nodes (parse, narrate)
have their module-level ``_client`` patched with a mock returning canned message
objects, so no live Anthropic call is made.
"""

import json
from typing import Any
from unittest.mock import MagicMock

from ui.chat_app.graph.nodes import explain, lookup, narrate, parse, predict, validate
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


# ---------------------------------------------------------------------------
# LLM node mock helpers
# ---------------------------------------------------------------------------


def _llm_response(text: str) -> MagicMock:
    """A mock Anthropic message whose single text block carries ``text``."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    msg = MagicMock()
    msg.content = [block]
    return msg


def _mock_client(text: str) -> MagicMock:
    """A mock Anthropic client whose ``messages.create`` returns ``text``."""
    client = MagicMock()
    client.messages.create.return_value = _llm_response(text)
    return client


def _parse_continuation(status: str, fields: dict[str, Any], reasoning: str = "ok") -> str:
    """The text the model returns after the prefilled opening brace.

    The parse node prepends ``{`` to the model's reply, so the mock must omit it.
    """
    payload = json.dumps({"status": status, "reasoning": reasoning, "fields": fields})
    assert payload.startswith("{")
    return payload[1:]


_FULL_FIELDS = {
    "town": "tampines",
    "flat_type": "4 room",
    "floor_area_sqm": 95.0,
    "lease_commence_date": 1985,
    "month": "2024-06",
    "postal_code": None,
}


class TestParseNode:
    async def test_extracts_and_normalises_fields(self, monkeypatch) -> None:
        monkeypatch.setattr(
            parse, "_client", _mock_client(_parse_continuation("extracted", _FULL_FIELDS))
        )
        result = await parse.run(GraphState(user_message="4 room flat in tampines, 95 sqm"))
        assert result["status"] == "pending"
        # Town and flat_type are uppercased to match the encoder vocabulary.
        assert result["town"] == "TAMPINES"
        assert result["flat_type"] == "4 ROOM"
        assert result["floor_area_sqm"] == 95.0
        assert result["lease_commence_date"] == 1985
        assert result["month"] == "2024-06"
        assert result["postal_code"] is None

    async def test_out_of_scope_sets_status_and_reasoning(self, monkeypatch) -> None:
        empty = dict.fromkeys(_FULL_FIELDS, None)
        monkeypatch.setattr(
            parse,
            "_client",
            _mock_client(_parse_continuation("out_of_scope", empty, reasoning="that is a condo")),
        )
        result = await parse.run(GraphState(user_message="how much for a condo in newton"))
        assert result == {"status": "out_of_scope", "parse_reasoning": "that is a condo"}

    async def test_malformed_json_sets_error_status(self, monkeypatch) -> None:
        monkeypatch.setattr(parse, "_client", _mock_client("not valid json at all"))
        result = await parse.run(GraphState(user_message="anything"))
        assert result["status"] == "error"
        assert result["parse_reasoning"]

    async def test_api_failure_sets_error_status(self, monkeypatch) -> None:
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("api down")
        monkeypatch.setattr(parse, "_client", client)
        result = await parse.run(GraphState(user_message="anything"))
        assert result["status"] == "error"
        assert "api down" in result["parse_reasoning"]

    async def test_null_town_is_left_as_none(self, monkeypatch) -> None:
        fields = {**_FULL_FIELDS, "town": None}
        monkeypatch.setattr(
            parse, "_client", _mock_client(_parse_continuation("extracted", fields))
        )
        result = await parse.run(GraphState(user_message="a 4 room flat, 95 sqm"))
        assert result["town"] is None


_READY_STATE_KW: dict[str, Any] = {
    "status": "ready_to_predict",
    "predicted_price": 586_888.0,
    "model_version": 7,
    "top_contributors": [
        {"feature": "Floor area (95 sqm)", "contribution": 40_000.0},
        {"feature": "Town: Tampines", "contribution": 12_000.0},
        {"feature": "Lease commence (1985)", "contribution": -8_000.0},
    ],
}


class TestNarrateNode:
    async def test_ready_branch_narrates_price(self, monkeypatch) -> None:
        client = _mock_client("Your flat is worth about S$586,888.")
        monkeypatch.setattr(narrate, "_client", client)
        state = GraphState(user_message="x", **_READY_STATE_KW)
        result = await narrate.run(state)
        assert result == {"response_text": "Your flat is worth about S$586,888."}
        # The contributors and price are passed to the model as context.
        context = client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "S$586,888" in context
        assert "Floor area (95 sqm)" in context

    async def test_follow_up_branch_lists_missing_fields(self, monkeypatch) -> None:
        client = _mock_client("What's the floor area and lease year?")
        monkeypatch.setattr(narrate, "_client", client)
        state = GraphState(
            user_message="x",
            status="needs_follow_up",
            missing_fields=["floor_area_sqm", "lease_commence_date"],
        )
        result = await narrate.run(state)
        assert result == {"response_text": "What's the floor area and lease year?"}
        context = client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "floor area" in context

    async def test_out_of_scope_branch_declines(self, monkeypatch) -> None:
        client = _mock_client("I only cover HDB resale flats.")
        monkeypatch.setattr(narrate, "_client", client)
        state = GraphState(
            user_message="x", status="out_of_scope", parse_reasoning="that is a condo"
        )
        result = await narrate.run(state)
        assert result == {"response_text": "I only cover HDB resale flats."}

    async def test_error_branch_apologises(self, monkeypatch) -> None:
        client = _mock_client("Sorry, something went wrong. Please try again.")
        monkeypatch.setattr(narrate, "_client", client)
        state = GraphState(user_message="x", status="error", error="boom: internal detail")
        result = await narrate.run(state)
        assert result["response_text"] == "Sorry, something went wrong. Please try again."
        # The raw internal error is never sent to the model.
        context = client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "internal detail" not in context

    async def test_llm_failure_falls_back_to_fixed_text(self, monkeypatch) -> None:
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("api down")
        monkeypatch.setattr(narrate, "_client", client)
        state = GraphState(user_message="x", **_READY_STATE_KW)
        result = await narrate.run(state)
        assert result["response_text"] == narrate._FALLBACK_ERROR
