"""Generate src/lookup/street_to_town.py from the raw HDB resale price CSVs.

Reads all five resale flat price CSV files from data/raw/, derives a
deterministic street_key → town mapping (modal town per street), and writes
the result as a static Python dict into src/lookup/street_to_town.py.

Re-run this script if the resale CSVs change (e.g. new towns are added or
existing streets are reassigned). The generated file is committed to the repo
so the lookup module has no runtime CSV dependency.

Usage:
    python scripts/build_street_town_mapping.py

The key transformation applied to each street name before storing:
    1. Apply expand_street_abbreviations() to resolve abbreviated tokens.
    2. Uppercase.
    3. Strip leading/trailing whitespace and collapse internal runs of spaces.

This matches the transformation applied in src/lookup/postal.py at query time,
so keys always align.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lookup.abbreviations import expand_street_abbreviations  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_RAW_FILES = [
    "Resale Flat Prices (Based on Approval Date), 1990 - 1999.csv",
    "Resale Flat Prices (Based on Approval Date), 2000 - Feb 2012.csv",
    "Resale Flat Prices (Based on Registration Date), From Mar 2012 to Dec 2014.csv",
    "Resale Flat Prices (Based on Registration Date), From Jan 2015 to Dec 2016.csv",
    "Resale flat prices based on registration date from Jan-2017 onwards.csv",
]

_OUTPUT_PATH = REPO_ROOT / "src" / "lookup" / "street_to_town.py"

_MODULE_HEADER = '''\
"""Street name to HDB planning town mapping.

Derived from the five HDB resale flat price CSV files in data/raw/. Each
street_name was normalised (uppercase, strip, collapse spaces) and expanded
using expand_street_abbreviations(), then grouped to find the modal town.
Re-run scripts/build_street_town_mapping.py if the resale CSVs change.

Source files:
  Resale Flat Prices (Based on Approval Date), 1990 - 1999.csv
  Resale Flat Prices (Based on Approval Date), 2000 - Feb 2012.csv
  Resale Flat Prices (Based on Registration Date), From Mar 2012 to Dec 2014.csv
  Resale Flat Prices (Based on Registration Date), From Jan 2015 to Dec 2016.csv
  Resale flat prices based on registration date from Jan-2017 onwards.csv
"""

STREET_TO_TOWN: dict[str, str] = {
'''

_MODULE_FOOTER = "}\n"


def _make_key(street_name: str) -> str:
    """Normalise a street name to the canonical lookup key."""
    return " ".join(expand_street_abbreviations(street_name.strip().upper()).split())


def build_mapping(data_dir: Path) -> dict[str, str]:
    """Return street_key → town mapping from all raw CSV files.

    For each street key the modal town is chosen. A warning is emitted at
    build time for any street that appears under more than one town — this has
    not occurred in practice with the current dataset.
    """
    frames: list[pd.DataFrame] = []
    for filename in _RAW_FILES:
        path = data_dir / filename
        df = pd.read_csv(path, usecols=["town", "street_name"])
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["town"] = combined["town"].str.strip().str.upper()
    combined["street_name"] = combined["street_name"].str.strip().str.upper()
    combined["street_key"] = combined["street_name"].apply(_make_key)

    multi_town = combined.groupby("street_key")["town"].nunique().pipe(lambda s: s[s > 1])
    if not multi_town.empty:
        for key in multi_town.index:
            town_counts = combined[combined["street_key"] == key]["town"].value_counts()
            logger.warning(
                "Street '%s' maps to multiple towns — picking mode '%s'. Full distribution: %s",
                key,
                town_counts.index[0],
                dict(town_counts),
            )

    mapping: dict[str, str] = (
        combined.groupby("street_key")["town"].agg(lambda s: s.mode().iloc[0]).to_dict()
    )
    logger.info("Built mapping: %d street keys", len(mapping))
    return mapping


def write_module(mapping: dict[str, str], output_path: Path) -> None:
    """Write the mapping as a static Python module."""
    lines = [_MODULE_HEADER]
    for key in sorted(mapping):
        town = mapping[key]
        lines.append(f"    {key!r}: {town!r},\n")
    lines.append(_MODULE_FOOTER)
    output_path.write_text("".join(lines), encoding="utf-8")
    logger.info("Written %s", output_path)


def main() -> None:
    data_dir = REPO_ROOT / "data" / "raw"
    mapping = build_mapping(data_dir)
    write_module(mapping, _OUTPUT_PATH)


if __name__ == "__main__":
    main()
