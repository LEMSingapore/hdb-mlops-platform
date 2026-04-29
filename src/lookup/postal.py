"""Postal code lookup: resolves Singapore 6-digit postal codes to addresses.

The lookup table is loaded once at module import from a CSV with columns
``postal`` and ``blk_no_RD_name``. The CSV ships with the repository at
``data/lookups/postal_codes.csv``; the path is overridable via the
``HDB_POSTAL_CSV_PATH`` environment variable for testing.

The ``town`` field of :class:`AddressInfo` is left as ``None`` until Session B,
which builds a ``street_full → town`` mapping from the resale training data
(that dataset carries ``town`` as a direct column).
"""

import csv
import logging
import os
from pathlib import Path

from pydantic import BaseModel

from lookup.abbreviations import expand_street_abbreviations

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CSV = _REPO_ROOT / "data" / "lookups" / "postal_codes.csv"

LOOKUP_CSV_PATH: Path = Path(os.environ.get("HDB_POSTAL_CSV_PATH", str(_DEFAULT_CSV)))


class AddressInfo(BaseModel):
    """Resolved address for a Singapore postal code.

    Attributes:
        postal_code: Always 6-digit zero-padded string, e.g. "018907".
        block: Block number including any letter suffix, e.g. "323C" or "11A".
        street_abbreviated: Street name as stored in the HDB data, e.g. "BT BATOK ST 22".
        street_full: Street name with abbreviations expanded, e.g. "BUKIT BATOK STREET 22".
        town: HDB planning town, e.g. "BUKIT BATOK". Populated in Session B once
            the resale training data (which carries ``town`` directly) is used to
            build a ``street_full → town`` mapping. ``None`` until then.
    """

    postal_code: str
    block: str
    street_abbreviated: str
    street_full: str
    town: str | None


def _load_lookup(csv_path: Path) -> dict[str, tuple[str, str]]:
    """Load the postal CSV into a dict keyed by zero-padded 6-digit postal string.

    Returns:
        Mapping of ``postal_code_str → (block, street_abbreviated)``.
    """
    lookup: dict[str, tuple[str, str]] = {}
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            postal_str = row["postal"].strip().zfill(6)
            blk_rd = row["blk_no_RD_name"].strip()
            block, _, street = blk_rd.partition(" ")
            lookup[postal_str] = (block, street)
    logger.debug("Loaded %d postal codes from %s", len(lookup), csv_path)
    return lookup


_LOOKUP: dict[str, tuple[str, str]] = _load_lookup(LOOKUP_CSV_PATH)


def lookup_postal(postal_code: int | str) -> AddressInfo | None:
    """Resolve a postal code to an :class:`AddressInfo`.

    Accepts 5- or 6-digit input; always returns a zero-padded 6-digit
    ``postal_code`` in the result. Returns ``None`` if the postal code is not
    found in the lookup table.

    Args:
        postal_code: Postal code as an integer (e.g. ``398614``) or string
            (e.g. ``"018907"`` or ``"18907"``).

    Returns:
        :class:`AddressInfo` on success, ``None`` if not found.
    """
    padded = str(postal_code).strip().zfill(6)
    entry = _LOOKUP.get(padded)
    if entry is None:
        return None
    block, street_abbreviated = entry
    return AddressInfo(
        postal_code=padded,
        block=block,
        street_abbreviated=street_abbreviated,
        street_full=expand_street_abbreviations(street_abbreviated),
        town=None,
    )
