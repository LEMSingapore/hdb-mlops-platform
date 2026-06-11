"""Comparable-transactions tool.

Returns the k historical resale transactions nearest to a target flat by
feature distance, querying the SQLite data layer built in Phase 1.6a. Surfaces
both the per-row distance and whether the search widened to town-only matching
so the LLM client can convey how closely the comparables match.
"""

from typing import TypedDict

from data.queries import find_similar_with_fallback
from mcp_server.config import MCPConfig


class SimilarTransaction(TypedDict):
    """One historical resale transaction with its distance to the target."""

    month: str
    town: str
    flat_type: str
    block: str
    street_name: str
    storey_range: str
    floor_area_sqm: float
    flat_model: str
    lease_commence_date: int
    resale_price: float
    distance: float


class SimilarTransactionsResult(TypedDict):
    """Comparable transactions plus match metadata."""

    matches: list[SimilarTransaction]
    match_count: int
    used_town_only_fallback: bool


def find_similar_transactions(
    town: str,
    flat_type: str,
    floor_area_sqm: float,
    lease_commence_date: int,
    k: int = 10,
) -> SimilarTransactionsResult:
    """Return the k nearest historical transactions to a target flat.

    Ranks by ``abs(floor_area_sqm - target) + abs(lease_commence_date - target)``
    within an exact town+flat_type match, widening to town-only if fewer than k
    exact matches exist.

    Args:
        town: HDB town, e.g. "TAMPINES". Case-insensitive.
        flat_type: Flat type, e.g. "4 ROOM". Case-insensitive.
        floor_area_sqm: Target floor area in square metres.
        lease_commence_date: Target lease commencement year.
        k: Maximum number of comparables to return. Defaults to 10.

    Returns:
        The matched transactions (may be fewer than k), the match count, and
        ``used_town_only_fallback`` indicating whether the search was widened.
    """
    config = MCPConfig()
    result = find_similar_with_fallback(
        town=town,
        flat_type=flat_type,
        floor_area_sqm=floor_area_sqm,
        lease_commence_date=lease_commence_date,
        k=k,
        db_path=config.db_path,
    )

    matches: list[SimilarTransaction] = [
        {
            "month": str(row.month),
            "town": str(row.town),
            "flat_type": str(row.flat_type),
            "block": str(row.block),
            "street_name": str(row.street_name),
            "storey_range": str(row.storey_range),
            "floor_area_sqm": float(row.floor_area_sqm),
            "flat_model": str(row.flat_model),
            "lease_commence_date": int(row.lease_commence_date),
            "resale_price": float(row.resale_price),
            "distance": float(row.distance),
        }
        for row in result.matches.itertuples(index=False)
    ]

    return SimilarTransactionsResult(
        matches=matches,
        match_count=len(matches),
        used_town_only_fallback=result.used_town_only_fallback,
    )
