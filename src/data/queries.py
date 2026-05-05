"""Query functions returning DataFrames from the HDB transactions database."""

import sqlite3
from pathlib import Path

import pandas as pd

from data.connection import get_connection


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


def find_similar(
    town: str,
    flat_type: str,
    floor_area_sqm: float,
    lease_commence_date: int,
    k: int = 10,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Return the k nearest historical transactions ranked by feature distance.

    Filters on exact town AND flat_type, then ranks by:
        abs(floor_area_sqm - target) + abs(lease_commence_date - target)

    Falls back to town-only matching if fewer than k exact matches are found.
    Returns all available rows without error when k exceeds available results.

    The distance column is included in the returned DataFrame — it is useful
    for Phase 1.6b's find_similar_transactions MCP tool response.

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
            return exact
        return _cursor_to_df(conn.execute(fallback_sql, fallback_params))
