"""Tests for src/data/queries.py."""

from pathlib import Path

import pytest

from data.queries import (
    count_transactions,
    find_similar,
    find_similar_with_fallback,
    load_all_transactions,
)

_EXPECTED_COLUMNS = {
    "id",
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
}


class TestLoadAllTransactions:
    def test_returns_all_rows(self, tiny_sqlite_db: Path) -> None:
        df = load_all_transactions(tiny_sqlite_db)
        assert len(df) == 20

    def test_returns_expected_columns(self, tiny_sqlite_db: Path) -> None:
        df = load_all_transactions(tiny_sqlite_db)
        assert _EXPECTED_COLUMNS.issubset(set(df.columns))

    def test_resale_price_is_numeric(self, tiny_sqlite_db: Path) -> None:
        df = load_all_transactions(tiny_sqlite_db)
        assert (df["resale_price"] > 0).all()

    def test_empty_database_returns_empty_dataframe(self, tmp_path: Path) -> None:
        import sqlite3

        db_path = tmp_path / "empty.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE transactions ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, month TEXT NOT NULL, "
            "town TEXT NOT NULL, flat_type TEXT NOT NULL, block TEXT NOT NULL, "
            "street_name TEXT NOT NULL, storey_range TEXT NOT NULL, "
            "floor_area_sqm REAL NOT NULL, flat_model TEXT NOT NULL, "
            "lease_commence_date INTEGER NOT NULL, resale_price REAL NOT NULL)"
        )
        conn.commit()
        conn.close()

        df = load_all_transactions(db_path)
        assert len(df) == 0
        assert "resale_price" in df.columns


class TestCountTransactions:
    def test_returns_correct_count(self, tiny_sqlite_db: Path) -> None:
        assert count_transactions(tiny_sqlite_db) == 20

    def test_returns_zero_for_empty_table(self, tmp_path: Path) -> None:
        import sqlite3

        db_path = tmp_path / "empty.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE transactions ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, month TEXT NOT NULL, "
            "town TEXT NOT NULL, flat_type TEXT NOT NULL, block TEXT NOT NULL, "
            "street_name TEXT NOT NULL, storey_range TEXT NOT NULL, "
            "floor_area_sqm REAL NOT NULL, flat_model TEXT NOT NULL, "
            "lease_commence_date INTEGER NOT NULL, resale_price REAL NOT NULL)"
        )
        conn.commit()
        conn.close()

        assert count_transactions(db_path) == 0


class TestFindSimilar:
    def test_exact_match_returns_rows_for_town_and_flat_type(self, tiny_sqlite_db: Path) -> None:
        df = find_similar("TAMPINES", "4 ROOM", 92.0, 1990, k=3, db_path=tiny_sqlite_db)
        assert len(df) == 3
        assert (df["town"] == "TAMPINES").all()
        assert (df["flat_type"] == "4 ROOM").all()

    def test_exact_match_results_are_ordered_by_ascending_distance(
        self, tiny_sqlite_db: Path
    ) -> None:
        df = find_similar("TAMPINES", "4 ROOM", 92.0, 1990, k=5, db_path=tiny_sqlite_db)
        assert df["distance"].is_monotonic_increasing

    def test_town_case_is_normalised(self, tiny_sqlite_db: Path) -> None:
        df = find_similar("tampines", "4 room", 92.0, 1990, k=3, db_path=tiny_sqlite_db)
        assert len(df) == 3
        assert (df["town"] == "TAMPINES").all()

    def test_insufficient_exact_matches_falls_back_to_town_only(self, tiny_sqlite_db: Path) -> None:
        # TAMPINES 5 ROOM has only 2 rows; k=5 triggers fallback to TAMPINES-only.
        df = find_similar("TAMPINES", "5 ROOM", 110.0, 1990, k=5, db_path=tiny_sqlite_db)
        assert len(df) == 5
        assert (df["town"] == "TAMPINES").all()
        # Fallback result must include flat types other than 5 ROOM.
        assert df["flat_type"].nunique() > 1

    def test_k_larger_than_available_returns_all_without_error(self, tiny_sqlite_db: Path) -> None:
        # The fixture has 20 total rows; requesting k=1000 returns everything available.
        df = find_similar("TAMPINES", "4 ROOM", 90.0, 1990, k=1000, db_path=tiny_sqlite_db)
        assert len(df) > 0
        assert len(df) <= 1000

    def test_distance_column_is_present_in_result(self, tiny_sqlite_db: Path) -> None:
        df = find_similar("TAMPINES", "4 ROOM", 90.0, 1990, k=3, db_path=tiny_sqlite_db)
        assert "distance" in df.columns

    def test_empty_database_returns_empty_dataframe(self, tmp_path: Path) -> None:
        import sqlite3

        db_path = tmp_path / "empty.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE transactions ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, month TEXT NOT NULL, "
            "town TEXT NOT NULL, flat_type TEXT NOT NULL, block TEXT NOT NULL, "
            "street_name TEXT NOT NULL, storey_range TEXT NOT NULL, "
            "floor_area_sqm REAL NOT NULL, flat_model TEXT NOT NULL, "
            "lease_commence_date INTEGER NOT NULL, resale_price REAL NOT NULL)"
        )
        conn.commit()
        conn.close()

        df = find_similar("TAMPINES", "4 ROOM", 90.0, 1990, k=5, db_path=db_path)
        assert len(df) == 0

    def test_missing_db_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            find_similar("TAMPINES", "4 ROOM", 90.0, 1990, db_path=tmp_path / "none.db")


class TestFindSimilarWithFallback:
    def test_exact_match_reports_no_fallback(self, tiny_sqlite_db: Path) -> None:
        # TAMPINES 4 ROOM has 8 rows; k=3 is satisfied without widening.
        result = find_similar_with_fallback(
            "TAMPINES", "4 ROOM", 92.0, 1990, k=3, db_path=tiny_sqlite_db
        )
        assert result.used_town_only_fallback is False
        assert len(result.matches) == 3
        assert (result.matches["flat_type"] == "4 ROOM").all()

    def test_insufficient_exact_matches_reports_fallback(self, tiny_sqlite_db: Path) -> None:
        # TAMPINES 5 ROOM has only 2 rows; k=5 widens to town-only.
        result = find_similar_with_fallback(
            "TAMPINES", "5 ROOM", 110.0, 1990, k=5, db_path=tiny_sqlite_db
        )
        assert result.used_town_only_fallback is True
        assert len(result.matches) == 5
        assert result.matches["flat_type"].nunique() > 1

    def test_distance_column_present(self, tiny_sqlite_db: Path) -> None:
        result = find_similar_with_fallback(
            "TAMPINES", "4 ROOM", 90.0, 1990, k=3, db_path=tiny_sqlite_db
        )
        assert "distance" in result.matches.columns
        assert result.matches["distance"].is_monotonic_increasing

    def test_no_matches_returns_empty_with_fallback_flag(self, tiny_sqlite_db: Path) -> None:
        # No transactions for this town; both queries return empty.
        result = find_similar_with_fallback(
            "WOODLANDS", "4 ROOM", 90.0, 1990, k=5, db_path=tiny_sqlite_db
        )
        assert result.used_town_only_fallback is True
        assert len(result.matches) == 0
