"""Happy-path and no-results tests for find_similar_transactions.

Points MCPConfig at the tiny SQLite fixture via the HDB_DB_PATH env var.
"""

from pathlib import Path

import pytest
from fastmcp import Client

from mcp_server.server import mcp

_MATCH_FIELDS = {
    "month",
    "town",
    "flat_type",
    "block",
    "street_name",
    "storey_range",
    "floor_area_sqm",
    "flat_model",
    "lease_commence_date",
    "resale_price",
    "distance",
}


@pytest.fixture
def db_env(tiny_sqlite_db: Path, monkeypatch):
    """Point the data layer at the tiny fixture DB for the duration of a test."""
    monkeypatch.setenv("HDB_DB_PATH", str(tiny_sqlite_db))
    return tiny_sqlite_db


class TestFindSimilarTransactions:
    async def test_happy_path_exact_match(self, db_env) -> None:
        # TAMPINES 4 ROOM has 8 rows in the fixture; k=5 needs no fallback.
        async with Client(mcp) as client:
            result = await client.call_tool(
                "find_similar_transactions",
                {
                    "town": "TAMPINES",
                    "flat_type": "4 ROOM",
                    "floor_area_sqm": 92.0,
                    "lease_commence_date": 1990,
                    "k": 5,
                },
            )
        data = result.structured_content
        assert data["match_count"] == 5
        assert len(data["matches"]) == 5
        assert data["used_town_only_fallback"] is False
        assert all(m["town"] == "TAMPINES" for m in data["matches"])
        assert all(m["flat_type"] == "4 ROOM" for m in data["matches"])

    async def test_each_match_carries_distance_and_full_fields(self, db_env) -> None:
        async with Client(mcp) as client:
            result = await client.call_tool(
                "find_similar_transactions",
                {
                    "town": "TAMPINES",
                    "flat_type": "4 ROOM",
                    "floor_area_sqm": 92.0,
                    "lease_commence_date": 1990,
                    "k": 3,
                },
            )
        matches = result.structured_content["matches"]
        for m in matches:
            assert set(m.keys()) == _MATCH_FIELDS
        distances = [m["distance"] for m in matches]
        assert distances == sorted(distances)

    async def test_fallback_flag_set_when_widened(self, db_env) -> None:
        # TAMPINES 5 ROOM has only 2 rows; k=5 widens to town-only.
        async with Client(mcp) as client:
            result = await client.call_tool(
                "find_similar_transactions",
                {
                    "town": "TAMPINES",
                    "flat_type": "5 ROOM",
                    "floor_area_sqm": 110.0,
                    "lease_commence_date": 1990,
                    "k": 5,
                },
            )
        data = result.structured_content
        assert data["used_town_only_fallback"] is True
        assert data["match_count"] == 5

    async def test_no_results_returns_empty_matches(self, db_env) -> None:
        # No WOODLANDS rows in the fixture.
        async with Client(mcp) as client:
            result = await client.call_tool(
                "find_similar_transactions",
                {
                    "town": "WOODLANDS",
                    "flat_type": "4 ROOM",
                    "floor_area_sqm": 90.0,
                    "lease_commence_date": 1990,
                    "k": 5,
                },
            )
        data = result.structured_content
        assert data["match_count"] == 0
        assert data["matches"] == []
