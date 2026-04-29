"""Singapore street name abbreviations as used in HDB postal data.

Sourced from HDB address data conventions. Tokens are matched whole-word only
(split on whitespace), so "ST" matches the standalone token "ST" but not the
substring within "STREET". This guarantees idempotency: running expand twice
produces the same result as running it once.
"""

STREET_ABBREVIATIONS: dict[str, str] = {
    "AVE": "AVENUE",
    "BLVD": "BOULEVARD",
    "BT": "BUKIT",
    "CL": "CLOSE",
    "CRES": "CRESCENT",
    "CTR": "CENTRE",
    "CTRL": "CENTRAL",
    "DR": "DRIVE",
    "GDNS": "GARDENS",
    "HTS": "HEIGHTS",
    "JLN": "JALAN",
    "KG": "KAMPONG",
    "LOR": "LORONG",
    "NTH": "NORTH",
    "PK": "PARK",
    "PL": "PLACE",
    "RD": "ROAD",
    "ST": "STREET",
    "STH": "SOUTH",
    "TER": "TERRACE",
    "UPP": "UPPER",
}


def expand_street_abbreviations(addr: str) -> str:
    """Expand known HDB street name abbreviations to their full forms.

    Replacement is token-level (split on whitespace), so partial substrings
    are never touched. The function is idempotent: no expanded value appears
    as a key in STREET_ABBREVIATIONS, so a second pass is a no-op.

    Args:
        addr: Street name or full address string, e.g. "BT BATOK ST 22".

    Returns:
        Address with abbreviations expanded, e.g. "BUKIT BATOK STREET 22".
    """
    return " ".join(STREET_ABBREVIATIONS.get(token, token) for token in addr.split())
