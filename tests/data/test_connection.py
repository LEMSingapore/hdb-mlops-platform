"""Tests for src/data/connection.py."""

import sqlite3
from pathlib import Path

import pytest

from data.connection import get_connection


class TestGetConnection:
    def test_yields_usable_sqlite_connection(self, tiny_sqlite_db: Path) -> None:
        with get_connection(tiny_sqlite_db) as conn:
            row = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()
            assert row[0] == 20

    def test_row_factory_is_sqlite_row(self, tiny_sqlite_db: Path) -> None:
        with get_connection(tiny_sqlite_db) as conn:
            row = conn.execute("SELECT town FROM transactions LIMIT 1").fetchone()
        assert isinstance(row, sqlite3.Row)

    def test_connection_closes_cleanly_on_context_exit(self, tiny_sqlite_db: Path) -> None:
        with get_connection(tiny_sqlite_db) as conn:
            pass
        # A closed connection raises ProgrammingError on use.
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")

    def test_honours_custom_db_path_env_var(self, tiny_sqlite_db: Path, monkeypatch) -> None:
        monkeypatch.setenv("HDB_DB_PATH", str(tiny_sqlite_db))
        # Call with db_path=None so DataConfig reads the env var.
        with get_connection(None) as conn:
            row = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()
            assert row[0] == 20

    def test_raises_file_not_found_for_missing_db(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.db"
        with pytest.raises(FileNotFoundError, match="csv_to_sqlite"), get_connection(missing):
            pass
