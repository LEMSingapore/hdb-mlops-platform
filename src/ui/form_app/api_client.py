"""Thin HTTP client for the HDB predictor FastAPI service.

All model logic lives in the API. This module is responsible only for
issuing HTTP requests, mapping error status codes to typed exceptions, and
returning parsed JSON dicts to the caller.
"""

import logging
from typing import Protocol

import httpx

logger = logging.getLogger(__name__)


class _APIConfig(Protocol):
    """Structural interface satisfied by any settings object that provides
    the two fields needed to construct the HTTP client."""

    api_base_url: str
    request_timeout_seconds: int


class APIError(Exception):
    """Base class for all API client errors."""


class ServiceUnavailableError(APIError):
    """Raised on 503 — model not yet loaded."""


class ValidationError(APIError):
    """Raised on 422 — field-level schema validation failure.

    ``errors`` maps field name to the first error message for that field,
    matching the structure FastAPI returns in its 422 response body.
    """

    def __init__(self, errors: dict[str, str]) -> None:
        self.errors = errors
        super().__init__(str(errors))


class ServerError(APIError):
    """Raised on 500 — the API returned an internal server error.

    ``detail`` is the raw parsed JSON from the response body.
    """

    def __init__(self, detail: object) -> None:
        self.detail = detail
        super().__init__(str(detail))


class APIConnectionError(APIError):
    """Raised on network-level failures — connection refused or timeout."""

    def __init__(self, url: str) -> None:
        self.url = url
        super().__init__(f"Could not reach {url}")


class APIClient:
    """Reusable HTTP client wrapping /predict and /explain.

    A single instance should be created at startup and reused — the
    underlying httpx.Client maintains a connection pool across calls.
    Pass a custom ``transport`` to inject a mock in tests.
    """

    def __init__(
        self,
        config: _APIConfig,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._config = config
        self._client = httpx.Client(
            base_url=config.api_base_url,
            timeout=config.request_timeout_seconds,
            transport=transport,
        )

    def predict(self, payload: dict) -> dict:
        """POST to /predict and return the parsed JSON response.

        Args:
            payload: Feature dict matching the HDBFeatureInput schema.

        Returns:
            Parsed PredictResponse JSON as a plain dict.

        Raises:
            ServiceUnavailableError: on 503.
            ValidationError: on 422.
            ServerError: on 500.
            APIConnectionError: on network-level failures.
        """
        return self._post("/predict", payload)

    def explain(self, payload: dict) -> dict:
        """POST to /explain and return the parsed JSON response.

        Args:
            payload: Feature dict matching the HDBFeatureInput schema.

        Returns:
            Parsed ExplainResponse JSON as a plain dict.

        Raises:
            ServiceUnavailableError: on 503.
            ValidationError: on 422.
            ServerError: on 500.
            APIConnectionError: on network-level failures.
        """
        return self._post("/explain", payload)

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def _post(self, path: str, payload: dict) -> dict:
        try:
            response = self._client.post(path, json=payload)
        except httpx.TransportError as exc:
            logger.info("Network failure calling %s%s: %s", self._config.api_base_url, path, exc)
            raise APIConnectionError(self._config.api_base_url) from exc

        if response.status_code == 503:
            raise ServiceUnavailableError()
        if response.status_code == 422:
            raise ValidationError(self._parse_422(response.json()))
        if response.status_code == 500:
            raise ServerError(response.json())

        response.raise_for_status()
        return response.json()

    @staticmethod
    def _parse_422(body: dict) -> dict[str, str]:
        """Extract a field-name → message map from a FastAPI 422 response body."""
        result: dict[str, str] = {}
        for error in body.get("detail", []):
            loc = error.get("loc", [])
            # loc is ["body", "field_name"] for top-level fields; take the last element.
            field = str(loc[-1]) if len(loc) > 1 else ".".join(str(p) for p in loc)
            result[field] = error.get("msg", "Validation error")
        return result
