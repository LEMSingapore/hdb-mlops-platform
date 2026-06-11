"""Tests for tool registration and input-schema correctness on the MCP server."""

from fastmcp import Client

from mcp_server.server import mcp

_EXPECTED_TOOLS = {
    "predict_price",
    "explain_prediction",
    "lookup_postal_code",
    "get_model_info",
    "find_similar_transactions",
}

# Required input fields per tool. Tools with optional-only or no inputs map to
# an empty set; presence of the tool is asserted separately.
_REQUIRED_INPUTS = {
    "predict_price": {"town", "flat_type", "floor_area_sqm", "lease_commence_date", "month"},
    "explain_prediction": {"town", "flat_type", "floor_area_sqm", "lease_commence_date", "month"},
    "lookup_postal_code": {"postal_code"},
    "get_model_info": set(),
    "find_similar_transactions": {
        "town",
        "flat_type",
        "floor_area_sqm",
        "lease_commence_date",
    },
}


async def test_all_five_tools_are_registered() -> None:
    async with Client(mcp) as client:
        tools = await client.list_tools()
    assert {t.name for t in tools} == _EXPECTED_TOOLS


async def test_each_tool_declares_expected_required_inputs() -> None:
    async with Client(mcp) as client:
        tools = await client.list_tools()
    by_name = {t.name: t for t in tools}
    for name, expected_required in _REQUIRED_INPUTS.items():
        schema = by_name[name].inputSchema
        required = set(schema.get("required", []))
        assert required == expected_required, name


async def test_find_similar_k_is_optional_with_default() -> None:
    async with Client(mcp) as client:
        tools = await client.list_tools()
    schema = next(t for t in tools if t.name == "find_similar_transactions").inputSchema
    assert "k" in schema["properties"]
    assert "k" not in schema.get("required", [])
    assert schema["properties"]["k"]["default"] == 10


async def test_server_exposes_instructions() -> None:
    # The instructions guide an LLM client's tool selection; they must be present.
    assert mcp.instructions is not None
    assert "lookup_postal_code" in mcp.instructions
