"""Migrate HDB resale CSVs into a single queryable SQLite database.

Run once after cloning the repo. Re-run if data/raw/ CSVs change.

    python scripts/csv_to_sqlite.py

The script is idempotent: it drops and recreates the transactions table on
every run. The resulting data/hdb.db is gitignored — the CSVs in data/raw/
remain the source of truth.
"""

import logging
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent
_DATA_RAW = _REPO_ROOT / "data" / "raw"
_DB_PATH = _REPO_ROOT / "data" / "hdb.db"

_RAW_FILES = [
    "Resale Flat Prices (Based on Approval Date), 1990 - 1999.csv",
    "Resale Flat Prices (Based on Approval Date), 2000 - Feb 2012.csv",
    "Resale Flat Prices (Based on Registration Date), From Mar 2012 to Dec 2014.csv",
    "Resale Flat Prices (Based on Registration Date), From Jan 2015 to Dec 2016.csv",
    "Resale flat prices based on registration date from Jan-2017 onwards.csv",
]

_DDL = """\
DROP TABLE IF EXISTS transactions;
CREATE TABLE transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    month TEXT NOT NULL,
    town TEXT NOT NULL,
    flat_type TEXT NOT NULL,
    block TEXT NOT NULL,
    street_name TEXT NOT NULL,
    storey_range TEXT NOT NULL,
    floor_area_sqm REAL NOT NULL,
    flat_model TEXT NOT NULL,
    lease_commence_date INTEGER NOT NULL,
    resale_price REAL NOT NULL
);
CREATE INDEX idx_town ON transactions(town);
CREATE INDEX idx_flat_type ON transactions(flat_type);
CREATE INDEX idx_month ON transactions(month);
CREATE INDEX idx_town_flat_type ON transactions(town, flat_type);\
"""

_INSERT_COLUMNS = [
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
]


def load_csvs(data_dir: Path) -> pd.DataFrame:
    """Load and concatenate all five raw HDB CSVs into a single DataFrame.

    Normalises town and flat_type to uppercase. Drops remaining_lease (present
    in the 2017+ file only) — the value is computable as
    99 - (transaction_year - lease_commence_date) and is not needed for training
    or the find_similar query.

    Args:
        data_dir: Directory containing the raw CSV files.

    Returns:
        Concatenated DataFrame with exactly the columns in _INSERT_COLUMNS.

    Raises:
        FileNotFoundError: If no CSV files are found in data_dir.
    """
    frames: list[pd.DataFrame] = []
    for filename in _RAW_FILES:
        path = data_dir / filename
        if not path.exists():
            logger.warning("CSV not found, skipping: %s", path)
            continue
        df = pd.read_csv(path)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No CSV files found in {data_dir}.")

    merged = pd.concat(frames, ignore_index=True)

    # remaining_lease is present in the 2017+ CSV only — drop before writing.
    if "remaining_lease" in merged.columns:
        merged = merged.drop(columns=["remaining_lease"])

    # Normalise string fields.
    merged["town"] = merged["town"].str.strip().str.upper()
    merged["flat_type"] = merged["flat_type"].str.strip().str.upper()
    merged["flat_model"] = merged["flat_model"].str.strip()
    merged["street_name"] = merged["street_name"].str.strip()
    merged["block"] = merged["block"].fillna("").astype(str).str.strip()

    # Ensure correct numeric types before insertion.
    merged["floor_area_sqm"] = merged["floor_area_sqm"].astype(float)
    merged["lease_commence_date"] = merged["lease_commence_date"].astype(int)
    merged["resale_price"] = merged["resale_price"].astype(float)

    return merged[_INSERT_COLUMNS]


def build_db(merged: pd.DataFrame, db_path: Path) -> None:
    """Create the SQLite database and insert all rows.

    Idempotent: drops and recreates the transactions table on every call.

    Args:
        merged: DataFrame with columns matching _INSERT_COLUMNS.
        db_path: Destination path for the SQLite file.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_DDL)
        merged.to_sql("transactions", conn, if_exists="append", index=False)
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    t0 = time.perf_counter()
    logger.info("Loading CSVs from %s", _DATA_RAW)
    merged = load_csvs(_DATA_RAW)
    row_count = len(merged)
    logger.info("Loaded %d rows from CSVs", row_count)

    logger.info("Building SQLite database at %s", _DB_PATH)
    build_db(merged, _DB_PATH)

    elapsed = time.perf_counter() - t0
    size_mb = _DB_PATH.stat().st_size / 1_048_576
    logger.info("Done: %d rows | %.1f MB | %.1fs", row_count, size_mb, elapsed)


if __name__ == "__main__":
    main()
