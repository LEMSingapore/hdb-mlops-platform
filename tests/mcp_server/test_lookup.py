"""Happy-path and not-found tests for lookup_postal_code.

Uses postal codes from the committed lookup table (data/lookups/postal_codes.csv),
so these tests are deterministic without a fixture override.
"""

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from mcp_server.server import mcp


class TestLookupPostalCode:
    async def test_happy_path_resolves_to_town(self) -> None:
        # 522201 -> TAMPINES STREET 21 in the committed lookup table.
        async with Client(mcp) as client:
            result = await client.call_tool("lookup_postal_code", {"postal_code": 522201})
        data = result.structured_content
        assert data["postal_code"] == "522201"
        assert data["town"] == "TAMPINES"
        assert data["block"]
        assert data["street_full"] == "TAMPINES STREET 21"

    async def test_accepts_string_postal_code(self) -> None:
        async with Client(mcp) as client:
            result = await client.call_tool("lookup_postal_code", {"postal_code": "522201"})
        assert result.structured_content["postal_code"] == "522201"

    async def test_unknown_postal_raises_tool_error(self) -> None:
        async with Client(mcp) as client:
            with pytest.raises(ToolError, match="not found"):
                await client.call_tool("lookup_postal_code", {"postal_code": "000000"})
