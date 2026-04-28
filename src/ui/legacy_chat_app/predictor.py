# mypy: ignore-errors
# ruff: noqa
"""Pure prediction layer — no Streamlit, no LLM. Loads the trained XGBoost
model once at import and exposes predict_price() as a plain function."""

import os
from datetime import date

import joblib

TOWNS = [
    "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT BATOK", "BUKIT MERAH",
    "BUKIT PANJANG", "BUKIT TIMAH", "CENTRAL AREA", "CHOA CHU KANG",
    "CLEMENTI", "GEYLANG", "HOUGANG", "JURONG EAST", "JURONG WEST",
    "KALLANG/WHAMPOA", "MARINE PARADE", "PASIR RIS", "PUNGGOL",
    "QUEENSTOWN", "SEMBAWANG", "SENGKANG", "SERANGOON", "TAMPINES",
    "TOA PAYOH", "WOODLANDS", "YISHUN",
]

FLAT_TYPES = [
    "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE", "MULTI-GENERATION",
]

_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "models", "XBR_trained_hdb_resale_modelV4a.pkl"
)

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = joblib.load(_MODEL_PATH)
    return _model


def predict_price(
    *,
    floor_area: float,
    lease_commence_date: int,
    postal_code: int,
    town: str,
    flat_type: str,
    current_year: int | None = None,
) -> int:
    """Predict HDB resale price. Returns price rounded to nearest $1,000.

    Raises ValueError on unknown town or flat_type.
    """
    if current_year is None:
        current_year = date.today().year

    town = town.upper().strip()
    flat_type = flat_type.upper().strip()

    if town not in TOWNS:
        raise ValueError(f"Unknown town: {town!r}. Must be one of {TOWNS}.")
    if flat_type not in FLAT_TYPES:
        raise ValueError(
            f"Unknown flat type: {flat_type!r}. Must be one of {FLAT_TYPES}."
        )

    features = [floor_area, lease_commence_date, postal_code, current_year]
    features += [1 if t == town else 0 for t in TOWNS]
    features += [1 if f == flat_type else 0 for f in FLAT_TYPES]

    price = float(_get_model().predict([features])[0])
    return int(round(price / 1000) * 1000)
