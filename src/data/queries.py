"""Query functions returning DataFrames from the HDB transactions database."""

import sqlite3
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from data.connection import get_connection


class SimilarResult(NamedTuple):
    """Result of a nearest-transaction search.

    Attributes:
        matches: DataFrame of matched transactions, sorted by ascending distance,
            with an additional distance column.
        used_town_only_fallback: True if the exact town+flat_type filter yielded
            fewer than k rows and the search widened to town-only. Consumers use
            this to communicate how loosely the comparables were matched.
    """

    matches: pd.DataFrame
    used_town_only_fallback: bool


def _cursor_to_df(cursor: sqlite3.Cursor) -> pd.DataFrame:
    """Convert a sqlite3 cursor result to a DataFrame."""
    if cursor.description is None:
        return pd.DataFrame()
    columns = [d[0] for d in cursor.description]
    rows = cursor.fetchall()
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame([tuple(row) for row in rows], columns=columns)


def load_all_transactions(db_path: Path | None = None) -> pd.DataFrame:
    """Return all rows from the transactions table as a DataFrame for training.

    Args:
        db_path: Override database path. Uses DataConfig default if None.

    Returns:
        DataFrame with columns matching the transactions schema.
    """
    with get_connection(db_path) as conn:
        cursor = conn.execute("SELECT * FROM transactions")
        return _cursor_to_df(cursor)


def count_transactions(db_path: Path | None = None) -> int:
    """Return the total number of rows in the transactions table.

    Args:
        db_path: Override database path. Uses DataConfig default if None.

    Returns:
        Row count as an integer.
    """
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()
        return int(row[0])


def find_similar_with_fallback(
    town: str,
    flat_type: str,
    floor_area_sqm: float,
    lease_commence_date: int,
    k: int = 10,
    db_path: Path | None = None,
) -> SimilarResult:
    """Return the k nearest transactions plus whether the town-only fallback was used.

    Filters on exact town AND flat_type, then ranks by:
        abs(floor_area_sqm - target) + abs(lease_commence_date - target)

    Falls back to town-only matching if fewer than k exact matches are found.
    Returns all available rows without error when k exceeds available results.

    The distance column is included in the returned DataFrame. Unlike
    :func:`find_similar`, this variant also reports whether the widening fallback
    fired — the find_similar_transactions MCP tool needs both the per-row distance
    and the fallback flag so the LLM client can convey match confidence.

    Args:
        town: Target town (compared after uppercasing).
        flat_type: Target flat type (compared after uppercasing).
        floor_area_sqm: Target floor area in square metres.
        lease_commence_date: Target lease commencement year.
        k: Maximum number of results to return.
        db_path: Override database path. Uses DataConfig default if None.

    Returns:
        A :class:`SimilarResult` pairing the matches DataFrame with the
        used_town_only_fallback flag.
    """
    _town = town.strip().upper()
    _flat_type = flat_type.strip().upper()

    exact_sql = """\
        SELECT *,
               ABS(floor_area_sqm - :area) + ABS(lease_commence_date - :lcd) AS distance
        FROM transactions
        WHERE town = :town AND flat_type = :flat_type
        ORDER BY distance
        LIMIT :k\
    """
    fallback_sql = """\
        SELECT *,
               ABS(floor_area_sqm - :area) + ABS(lease_commence_date - :lcd) AS distance
        FROM transactions
        WHERE town = :town
        ORDER BY distance
        LIMIT :k\
    """
    exact_params: dict = {
        "town": _town,
        "flat_type": _flat_type,
        "area": floor_area_sqm,
        "lcd": lease_commence_date,
        "k": k,
    }
    fallback_params: dict = {
        "town": _town,
        "area": floor_area_sqm,
        "lcd": lease_commence_date,
        "k": k,
    }

    with get_connection(db_path) as conn:
        exact = _cursor_to_df(conn.execute(exact_sql, exact_params))
        if len(exact) >= k:
            return SimilarResult(matches=exact, used_town_only_fallback=False)
        fallback = _cursor_to_df(conn.execute(fallback_sql, fallback_params))
        return SimilarResult(matches=fallback, used_town_only_fallback=True)


def find_similar(
    town: str,
    flat_type: str,
    floor_area_sqm: float,
    lease_commence_date: int,
    k: int = 10,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Return the k nearest historical transactions ranked by feature distance.

    Thin wrapper over :func:`find_similar_with_fallback` that discards the
    fallback flag. Retained for callers (e.g. training-adjacent analysis) that
    only need the matches DataFrame.

    Args:
        town: Target town (compared after uppercasing).
        flat_type: Target flat type (compared after uppercasing).
        floor_area_sqm: Target floor area in square metres.
        lease_commence_date: Target lease commencement year.
        k: Maximum number of results to return.
        db_path: Override database path. Uses DataConfig default if None.

    Returns:
        DataFrame of up to k transactions, sorted by ascending feature distance,
        with an additional distance column.
    """
    return find_similar_with_fallback(
        town, flat_type, floor_area_sqm, lease_commence_date, k, db_path
    ).matches
