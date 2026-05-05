"""SQLite connection management for the HDB data layer."""

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from data.config import DataConfig


@contextmanager
def get_connection(db_path: Path | None = None) -> Generator[sqlite3.Connection, None, None]:
    """Yield a configured sqlite3 connection, closing it on exit.

    Single source of truth for DB connections — every module that needs a
    connection should call this rather than sqlite3.connect() directly.

    Args:
        db_path: Path to the database file. Uses DataConfig.db_path if None.

    Yields:
        A sqlite3.Connection with row_factory set to sqlite3.Row.

    Raises:
        FileNotFoundError: If the database file does not exist at the
            resolved path. Message includes the remediation step.
    """
    resolved = db_path if db_path is not None else DataConfig().db_path
    if not resolved.exists():
        raise FileNotFoundError(
            f"Database not found at {resolved}. "
            "Run python scripts/csv_to_sqlite.py first to populate the "
            "database from the raw CSVs."
        )
    conn = sqlite3.connect(str(resolved))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()
