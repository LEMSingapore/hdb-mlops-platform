"""Postal code lookup tool.

Resolves a Singapore postal code to its block, street, and HDB town by
delegating to :func:`lookup.postal.lookup_postal`. The resolved town is what
the prediction tools expect, so an agent typically calls this first when the
user supplies a postal code rather than a town name.
"""

from typing import TypedDict

from fastmcp.exceptions import ToolError

from lookup.postal import lookup_postal as _lookup_postal


class PostalLookupResult(TypedDict):
    """Resolved address for a Singapore postal code."""

    postal_code: str
    block: str
    street_abbreviated: str
    street_full: str
    town: str | None


def lookup_postal_code(postal_code: int | str) -> PostalLookupResult:
    """Resolve a Singapore postal code to its block, street, and HDB town.

    Args:
        postal_code: A 5- or 6-digit postal code, as an integer or string.

    Returns:
        The resolved address. ``town`` is null only when the street is absent
        from the HDB street-to-town mapping (e.g. a non-HDB address).

    Raises:
        ToolError: If the postal code is not present in the lookup table.
    """
    info = _lookup_postal(postal_code)
    if info is None:
        raise ToolError(f"Postal code {postal_code} not found in the lookup table.")
    return PostalLookupResult(
        postal_code=info.postal_code,
        block=info.block,
        street_abbreviated=info.street_abbreviated,
        street_full=info.street_full,
        town=info.town,
    )
