"""Tests for ui.chat_app.chat_agent.

All Anthropic API calls are mocked — no live LLM calls are made. The
Anthropic client is replaced with a mock via ``unittest.mock.patch`` so
tests only verify orchestration logic: tool dispatch order, response
formatting, and the missing-fields follow-up path.
"""

from unittest.mock import MagicMock, patch

from ui.chat_app.chat_agent import FLAT_TYPES, TOOLS, TOWNS, chat_turn

# ---------------------------------------------------------------------------
# Tool schema assertions
# ---------------------------------------------------------------------------


def _tool(name: str) -> dict:
    match = [t for t in TOOLS if t["name"] == name]
    assert match, f"Tool {name!r} not found in TOOLS"
    return match[0]


def test_tools_list_contains_three_tools():
    assert len(TOOLS) == 3
    names = {t["name"] for t in TOOLS}
    assert names == {"lookup_postal_code", "predict_hdb_price", "explain_hdb_price"}


def test_predict_hdb_price_schema_has_five_required_fields():
    schema = _tool("predict_hdb_price")["input_schema"]
    assert set(schema["required"]) == {
        "town",
        "flat_type",
        "floor_area_sqm",
        "lease_commence_date",
        "month",
    }


def test_predict_hdb_price_schema_town_enum_matches_towns_constant():
    schema = _tool("predict_hdb_price")["input_schema"]
    assert schema["properties"]["town"]["enum"] == TOWNS


def test_predict_hdb_price_schema_flat_type_enum_matches_flat_types_constant():
    schema = _tool("predict_hdb_price")["input_schema"]
    assert schema["properties"]["flat_type"]["enum"] == FLAT_TYPES


def test_explain_hdb_price_schema_required_fields_match_predict():
    predict_req = set(_tool("predict_hdb_price")["input_schema"]["required"])
    explain_req = set(_tool("explain_hdb_price")["input_schema"]["required"])
    assert predict_req == explain_req


def test_lookup_postal_code_schema_requires_postal_code_int():
    schema = _tool("lookup_postal_code")["input_schema"]
    assert schema["required"] == ["postal_code"]
    assert schema["properties"]["postal_code"]["type"] == "integer"


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_message(stop_reason: str, content: list) -> MagicMock:
    msg = MagicMock()
    msg.stop_reason = stop_reason
    msg.content = content
    return msg


def _text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _tool_use_block(name: str, tool_id: str, input_data: dict) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.id = tool_id
    block.input = input_data
    return block


# ---------------------------------------------------------------------------
# Postal code triggers lookup first, then predict
# ---------------------------------------------------------------------------


def test_postal_code_triggers_lookup_then_predict():
    """A query with a postal code should call lookup_postal_code before predict."""
    lookup_block = _tool_use_block("lookup_postal_code", "tu_001", {"postal_code": 528003})
    predict_block = _tool_use_block(
        "predict_hdb_price",
        "tu_002",
        {
            "town": "TAMPINES",
            "flat_type": "4 ROOM",
            "floor_area_sqm": 92.0,
            "lease_commence_date": 1990,
            "month": "2025-04",
        },
    )
    final_text = _text_block("The estimated price is S$621,000.")

    responses = [
        _make_message("tool_use", [lookup_block]),
        _make_message("tool_use", [predict_block]),
        _make_message("end_turn", [final_text]),
    ]
    call_count = 0

    def fake_create(**kwargs):
        nonlocal call_count
        resp = responses[call_count]
        call_count += 1
        return resp

    with (
        patch("ui.chat_app.chat_agent._client") as mock_anthropic,
        patch("ui.chat_app.chat_agent.lookup_postal", return_value=None) as mock_lookup,
        patch(
            "ui.chat_app.chat_agent.predict_price",
            return_value={
                "predicted_resale_price": 621000.0,
                "model_version": 4,
                "model_alias": "champion",
            },
        ),
        patch(
            "ui.chat_app.chat_agent.explain_price",
            return_value={
                "predicted_resale_price": 621000.0,
                "base_value": 440000.0,
                "top_contributors": [],
                "model_version": 4,
            },
        ),
    ):
        # Ensure lookup_postal returns an object with .town/.block/.street_full
        from lookup.postal import AddressInfo

        mock_lookup.return_value = AddressInfo(
            postal_code="528003",
            block="137",
            street_abbreviated="TAMPINES ST 11",
            street_full="TAMPINES STREET 11",
            town="TAMPINES",
        )

        mock_anthropic.messages.create.side_effect = fake_create

        history = [{"role": "user", "content": "4-room Tampines 92sqm postal 528003"}]
        _, reply = chat_turn(history)

    assert call_count == 3

    # First tool result in history should carry the lookup_postal_code use_id
    tool_result_msgs = [
        m
        for m in history
        if m.get("role") == "user"
        and isinstance(m.get("content"), list)
        and any(
            isinstance(item, dict) and item.get("type") == "tool_result" for item in m["content"]
        )
    ]
    assert tool_result_msgs[0]["content"][0]["tool_use_id"] == "tu_001"
    assert tool_result_msgs[1]["content"][0]["tool_use_id"] == "tu_002"
    assert "621,000" in reply


# ---------------------------------------------------------------------------
# Known price appears in formatted reply
# ---------------------------------------------------------------------------


def test_predict_result_price_appears_in_sgd_in_reply():
    predict_block = _tool_use_block(
        "predict_hdb_price",
        "tu_100",
        {
            "town": "BISHAN",
            "flat_type": "5 ROOM",
            "floor_area_sqm": 120.0,
            "lease_commence_date": 1988,
            "month": "2025-04",
        },
    )
    text_with_price = _text_block("Estimated price: S$780,000 for your 5-room Bishan flat.")

    responses = [
        _make_message("tool_use", [predict_block]),
        _make_message("end_turn", [text_with_price]),
    ]
    idx = 0

    def fake_create(**kwargs):
        nonlocal idx
        r = responses[idx]
        idx += 1
        return r

    with (
        patch("ui.chat_app.chat_agent._client") as mock_anthropic,
        patch(
            "ui.chat_app.chat_agent.predict_price",
            return_value={
                "predicted_resale_price": 780000.0,
                "model_version": 4,
                "model_alias": "champion",
            },
        ),
    ):
        mock_anthropic.messages.create.side_effect = fake_create
        history = [{"role": "user", "content": "5-room Bishan, 120sqm, lease 1988"}]
        _, reply = chat_turn(history)

    assert "780,000" in reply


# ---------------------------------------------------------------------------
# Missing fields path — no tool call
# ---------------------------------------------------------------------------


def test_incomplete_description_returns_follow_up_question_not_tool_call():
    """If all fields are not yet known, the LLM should ask a follow-up."""
    follow_up = _text_block("Could you tell me the floor area of the flat?")
    response = _make_message("end_turn", [follow_up])

    with patch("ui.chat_app.chat_agent._client") as mock_anthropic:
        mock_anthropic.messages.create.return_value = response
        history = [{"role": "user", "content": "I want to know the price of a flat in Bishan"}]
        _, reply = chat_turn(history)

    mock_anthropic.messages.create.assert_called_once()
    assert reply == "Could you tell me the floor area of the flat?"

    # No tool results should appear in history (only the assistant message)
    tool_result_msgs = [
        m
        for m in history
        if m.get("role") == "user"
        and isinstance(m.get("content"), list)
        and any(
            isinstance(item, dict) and item.get("type") == "tool_result" for item in m["content"]
        )
    ]
    assert len(tool_result_msgs) == 0


# ---------------------------------------------------------------------------
# System prompt uses cache_control
# ---------------------------------------------------------------------------


def test_system_prompt_has_cache_control_ephemeral():
    response = _make_message("end_turn", [_text_block("Hi")])

    with patch("ui.chat_app.chat_agent._client") as mock_anthropic:
        mock_anthropic.messages.create.return_value = response
        chat_turn([{"role": "user", "content": "Hello"}])

    call_kwargs = mock_anthropic.messages.create.call_args.kwargs
    system = call_kwargs["system"]
    assert isinstance(system, list)
    assert system[0]["cache_control"] == {"type": "ephemeral"}


# ---------------------------------------------------------------------------
# Multiple tool_use blocks in one response are all answered
# ---------------------------------------------------------------------------


def test_multiple_tool_uses_in_one_response_each_get_a_tool_result():
    """If the model returns predict + explain in one response, both get tool_results."""
    predict_block = _tool_use_block(
        "predict_hdb_price",
        "tu_p",
        {
            "town": "BISHAN",
            "flat_type": "5 ROOM",
            "floor_area_sqm": 120.0,
            "lease_commence_date": 1988,
            "month": "2025-04",
        },
    )
    explain_block = _tool_use_block(
        "explain_hdb_price",
        "tu_e",
        {
            "town": "BISHAN",
            "flat_type": "5 ROOM",
            "floor_area_sqm": 120.0,
            "lease_commence_date": 1988,
            "month": "2025-04",
        },
    )
    final_text = _text_block("Estimated S$780,000. Floor area drives the price.")

    multi_tool_response = _make_message("tool_use", [predict_block, explain_block])
    end_response = _make_message("end_turn", [final_text])

    responses = [multi_tool_response, end_response]
    idx = 0

    def fake_create(**kwargs):
        nonlocal idx
        r = responses[idx]
        idx += 1
        return r

    with (
        patch("ui.chat_app.chat_agent._client") as mock_anthropic,
        patch(
            "ui.chat_app.chat_agent.predict_price",
            return_value={
                "predicted_resale_price": 780000.0,
                "model_version": 4,
                "model_alias": "champion",
            },
        ),
        patch(
            "ui.chat_app.chat_agent.explain_price",
            return_value={
                "predicted_resale_price": 780000.0,
                "base_value": 440000.0,
                "top_contributors": [],
                "model_version": 4,
            },
        ),
    ):
        mock_anthropic.messages.create.side_effect = fake_create
        history = [{"role": "user", "content": "5-room Bishan, 120sqm, lease 1988"}]
        _, reply = chat_turn(history)

    # The user message after the multi-tool response must contain both tool_results
    tool_result_msgs = [
        m
        for m in history
        if m.get("role") == "user"
        and isinstance(m.get("content"), list)
        and any(
            isinstance(item, dict) and item.get("type") == "tool_result" for item in m["content"]
        )
    ]
    assert len(tool_result_msgs) == 1, "both results go in one user message"
    result_ids = {item["tool_use_id"] for item in tool_result_msgs[0]["content"]}
    assert result_ids == {"tu_p", "tu_e"}
    assert "780,000" in reply
