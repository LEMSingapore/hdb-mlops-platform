"""HTTP-backed prediction and lookup helpers for the chat agent.

All heavy computation runs in the FastAPI service; these functions are thin
wrappers that translate function calls into HTTP requests and surface typed
errors the chat agent can handle.

``_client`` is a module-level singleton initialised from ``ChatConfig``.
Tests can replace it via ``unittest.mock.patch("ui.chat_app.predictor._client", ...)``.
"""

import logging
from typing import Any, TypedDict

from lookup.postal import AddressInfo
from lookup.postal import lookup_postal as _lookup_postal
from ui.chat_app.config import ChatConfig
from ui.form_app.api_client import (
    APIClient,
    APIConnectionError,  # noqa: F401 — re-exported for callers
    ServerError,  # noqa: F401
    ServiceUnavailableError,  # noqa: F401
    ValidationError,  # noqa: F401
)

logger = logging.getLogger(__name__)

_config = ChatConfig()
_client = APIClient(_config)


class _ContributionEntry(TypedDict):
    feature: str
    contribution: float


def predict_price(
    *,
    town: str,
    flat_type: str,
    floor_area_sqm: float,
    lease_commence_date: int,
    month: str,
) -> dict[str, Any]:
    """POST to /predict and return the parsed response.

    Args:
        town: HDB town name, e.g. "TAMPINES".
        flat_type: HDB flat type, e.g. "4 ROOM".
        floor_area_sqm: Floor area in square metres.
        lease_commence_date: Year the lease began.
        month: Transaction month in YYYY-MM format.

    Returns:
        Dict with keys ``predicted_resale_price``, ``model_version``, ``model_alias``.

    Raises:
        ServiceUnavailableError: on 503.
        ValidationError: on 422.
        ServerError: on 500.
        APIConnectionError: on network-level failures.
    """
    payload = {
        "town": town,
        "flat_type": flat_type,
        "floor_area_sqm": floor_area_sqm,
        "lease_commence_date": lease_commence_date,
        "month": month,
    }
    return _client.predict(payload)


def explain_price(
    *,
    town: str,
    flat_type: str,
    floor_area_sqm: float,
    lease_commence_date: int,
    month: str,
) -> dict[str, Any]:
    """POST to /explain and return the top-3 SHAP contributors by absolute value.

    The API returns contributions for all features; this function slices to the
    three largest by absolute value (descending) to keep the LLM context small.

    Args:
        town: HDB town name.
        flat_type: HDB flat type.
        floor_area_sqm: Floor area in square metres.
        lease_commence_date: Year the lease began.
        month: Transaction month in YYYY-MM format.

    Returns:
        Dict with keys ``predicted_resale_price``, ``base_value``,
        ``top_contributors`` (list of {feature, contribution} dicts),
        and ``model_version``.

    Raises:
        ServiceUnavailableError: on 503.
        ValidationError: on 422.
        ServerError: on 500.
        APIConnectionError: on network-level failures.
    """
    payload = {
        "town": town,
        "flat_type": flat_type,
        "floor_area_sqm": floor_area_sqm,
        "lease_commence_date": lease_commence_date,
        "month": month,
    }
    result = _client.explain(payload)

    all_contributions: dict[str, float] = result["feature_contributions"]
    items: list[_ContributionEntry] = [
        {"feature": k, "contribution": v} for k, v in all_contributions.items()
    ]
    top_contributors = sorted(items, key=lambda x: abs(x["contribution"]), reverse=True)[:3]

    return {
        "predicted_resale_price": result["predicted_resale_price"],
        "base_value": result["base_value"],
        "top_contributors": top_contributors,
        "model_version": result["model_version"],
    }


def lookup_postal(postal_code: int | str) -> AddressInfo | None:
    """Resolve a Singapore postal code to an address with town.

    Wraps :func:`lookup.postal.lookup_postal`. Returns ``None`` if the postal
    code is not in the lookup table.

    Args:
        postal_code: 5- or 6-digit postal code as int or string.

    Returns:
        :class:`~lookup.postal.AddressInfo` on success, ``None`` if not found.
    """
    return _lookup_postal(postal_code)
