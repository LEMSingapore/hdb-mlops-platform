"""Streamlit front end for HDB resale price prediction.

Form-based UI that calls the FastAPI service for /predict and /explain.
No model code, SHAP computation, or serialised model loading runs in this
process — the API handles all of that.
"""

import logging
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import shap
import streamlit as st

from ui.api_client import (
    APIClient,
    APIConnectionError,
    ServerError,
    ServiceUnavailableError,
    ValidationError,
)
from ui.config import UIConfig

logger = logging.getLogger(__name__)

TOWNS: list[str] = [
    "ANG MO KIO",
    "BEDOK",
    "BISHAN",
    "BUKIT BATOK",
    "BUKIT MERAH",
    "BUKIT PANJANG",
    "BUKIT TIMAH",
    "CENTRAL AREA",
    "CHOA CHU KANG",
    "CLEMENTI",
    "GEYLANG",
    "HOUGANG",
    "JURONG EAST",
    "JURONG WEST",
    "KALLANG/WHAMPOA",
    "MARINE PARADE",
    "PASIR RIS",
    "PUNGGOL",
    "QUEENSTOWN",
    "SEMBAWANG",
    "SENGKANG",
    "SERANGOON",
    "TAMPINES",
    "TOA PAYOH",
    "WOODLANDS",
    "YISHUN",
]

FLAT_TYPES: list[str] = [
    "1 ROOM",
    "2 ROOM",
    "3 ROOM",
    "4 ROOM",
    "5 ROOM",
    "EXECUTIVE",
    "MULTI-GENERATION",
]

FLAT_MODELS: list[str] = [
    "Improved",
    "New Generation",
    "Model A",
    "Standard",
    "Simplified",
    "Premium Apartment",
    "Maisonette",
    "Apartment",
    "DBSS",
    "Type S1",
    "Type S2",
    "Adjoined flat",
    "Multi Generation",
    "Premium Apartment Loft",
    "2-room",
    "3Gen",
    "Improved-Maisonette",
    "Premium Maisonette",
    "Model A-Maisonette",
    "Model A2",
    "Terrace",
]

STOREY_RANGES: list[str] = [
    "01 TO 03",
    "04 TO 06",
    "07 TO 09",
    "10 TO 12",
    "13 TO 15",
    "16 TO 18",
    "19 TO 21",
    "22 TO 24",
    "25 TO 27",
    "28 TO 30",
    "31 TO 33",
    "34 TO 36",
    "37 TO 39",
    "40 TO 42",
    "43 TO 45",
    "46 TO 48",
    "49 TO 51",
]


@st.cache_resource
def _get_client() -> APIClient:
    return APIClient(UIConfig())


def _waterfall_figure(
    feature_contributions: dict[str, float],
    base_value: float,
) -> plt.Figure:
    """Construct a SHAP Explanation from API response data and render a waterfall figure."""
    values = np.array(list(feature_contributions.values()))
    explanation = shap.Explanation(
        values=values,
        base_values=base_value,
        data=values,
        feature_names=list(feature_contributions.keys()),
    )
    shap.plots.waterfall(explanation, show=False)
    fig = plt.gcf()
    plt.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="HDB Resale Price Predictor", layout="centered")
    st.title("HDB Resale Price Predictor")

    client = _get_client()

    with st.form("prediction_form"):
        town = st.selectbox("Town", TOWNS)
        flat_type = st.selectbox("Flat Type", FLAT_TYPES)
        flat_model = st.selectbox("Flat Model", FLAT_MODELS)
        storey_range = st.selectbox("Storey Range", STOREY_RANGES)
        floor_area_sqm = st.number_input(
            "Floor Area (sqm)", min_value=28, max_value=200, value=90, step=1
        )
        lease_commence_date = st.number_input(
            "Lease Commence Date",
            min_value=1961,
            max_value=date.today().year - 1,
            value=1990,
            step=1,
        )
        transaction_month = st.date_input(
            "Transaction Month",
            value=date.today().replace(day=1),
            help="Pick any day — only the year and month are sent to the API.",
        )
        submitted = st.form_submit_button("Predict")

    if not submitted:
        return

    payload: dict = {
        "town": town,
        "flat_type": flat_type,
        "flat_model": flat_model,
        "storey_range": storey_range,
        "floor_area_sqm": float(floor_area_sqm),
        "lease_commence_date": int(lease_commence_date),
        "month": transaction_month.strftime("%Y-%m"),
    }

    try:
        predict_resp = client.predict(payload)
        explain_resp = client.explain(payload)
    except ServiceUnavailableError:
        st.error("Model not yet loaded — please try again in a moment")
        return
    except ValidationError as exc:
        for field, message in exc.errors.items():
            st.error(f"**{field}:** {message}")
        return
    except APIConnectionError as exc:
        st.error(f"Could not reach the API at {exc.url}. Is the FastAPI service running?")
        return
    except ServerError as exc:
        st.error("The API returned an internal server error:")
        st.json(exc.detail)
        return

    price: float = predict_resp["predicted_resale_price"]
    rounded = round(price / 1000) * 1000
    logger.info("Prediction: S$%,.0f (raw=%.2f)", rounded, price)

    st.subheader(f"S${rounded:,.0f}")
    st.caption("Predicted resale price, rounded to nearest S$1,000")

    contributions: dict[str, float] = explain_resp["feature_contributions"]
    base_value: float = explain_resp["base_value"]

    fig = _waterfall_figure(contributions, base_value)
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Show raw response"):
        st.json({"predict": predict_resp, "explain": explain_resp})


if __name__ == "__main__":
    main()
