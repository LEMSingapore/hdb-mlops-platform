"""Unit tests for Pydantic request/response schemas."""

import pytest
from pydantic import ValidationError

from serving.schemas import (
    HDBFeatureInput,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)

_VALID_PAYLOAD = {
    "town": "TAMPINES",
    "flat_type": "4 ROOM",
    "floor_area_sqm": 93.0,
    "lease_commence_date": 1990,
    "month": "2024-01",
}


class TestHDBFeatureInput:
    def test_valid_payload_parses(self):
        feature = HDBFeatureInput(**_VALID_PAYLOAD)
        assert feature.town == "TAMPINES"
        assert feature.flat_type == "4 ROOM"
        assert feature.floor_area_sqm == 93.0
        assert feature.lease_commence_date == 1990
        assert feature.month == "2024-01"

    def test_five_fields_only(self):
        assert set(HDBFeatureInput.model_fields) == {
            "town",
            "flat_type",
            "floor_area_sqm",
            "lease_commence_date",
            "month",
        }

    def test_zero_floor_area_rejected(self):
        with pytest.raises(ValidationError):
            HDBFeatureInput(**{**_VALID_PAYLOAD, "floor_area_sqm": 0.0})

    def test_negative_floor_area_rejected(self):
        with pytest.raises(ValidationError):
            HDBFeatureInput(**{**_VALID_PAYLOAD, "floor_area_sqm": -10.0})

    def test_lease_year_too_early_rejected(self):
        with pytest.raises(ValidationError):
            HDBFeatureInput(**{**_VALID_PAYLOAD, "lease_commence_date": 1959})

    def test_lease_year_boundary_1960_rejected(self):
        with pytest.raises(ValidationError):
            HDBFeatureInput(**{**_VALID_PAYLOAD, "lease_commence_date": 1960})

    def test_lease_year_too_late_rejected(self):
        with pytest.raises(ValidationError):
            HDBFeatureInput(**{**_VALID_PAYLOAD, "lease_commence_date": 2031})

    def test_lease_year_boundary_2030_rejected(self):
        with pytest.raises(ValidationError):
            HDBFeatureInput(**{**_VALID_PAYLOAD, "lease_commence_date": 2030})

    def test_month_invalid_format_accepted_by_regex(self):
        # The pattern ^\d{4}-\d{2}$ accepts 2024-13 since it only checks digit count.
        feature = HDBFeatureInput(**{**_VALID_PAYLOAD, "month": "2024-13"})
        assert feature.month == "2024-13"

    def test_month_wrong_format_two_digit_year_rejected(self):
        with pytest.raises(ValidationError):
            HDBFeatureInput(**{**_VALID_PAYLOAD, "month": "24-06"})

    def test_month_wrong_format_single_digit_month_rejected(self):
        with pytest.raises(ValidationError):
            HDBFeatureInput(**{**_VALID_PAYLOAD, "month": "2024-6"})

    def test_month_no_separator_rejected(self):
        with pytest.raises(ValidationError):
            HDBFeatureInput(**{**_VALID_PAYLOAD, "month": "202406"})


class TestPredictRequest:
    def test_valid_payload_parses(self):
        req = PredictRequest(**_VALID_PAYLOAD)
        assert isinstance(req, HDBFeatureInput)


class TestPredictResponse:
    def test_model_version_is_int(self):
        resp = PredictResponse(
            predicted_resale_price=450_000.0,
            model_version=1,
            model_alias="champion",
        )
        assert isinstance(resp.model_version, int)

    def test_model_version_coerced_from_string(self):
        resp = PredictResponse(
            predicted_resale_price=450_000.0,
            model_version="3",  # type: ignore[arg-type]
            model_alias="champion",
        )
        assert resp.model_version == 3
        assert isinstance(resp.model_version, int)


class TestHealthResponse:
    def test_model_version_none_when_not_loaded(self):
        resp = HealthResponse(status="ok", model_loaded=False, model_version=None)
        assert resp.model_loaded is False
        assert resp.model_version is None

    def test_model_version_is_int_when_loaded(self):
        resp = HealthResponse(status="ok", model_loaded=True, model_version=7)
        assert isinstance(resp.model_version, int)
        assert resp.model_version == 7


class TestModelInfoResponse:
    def test_model_version_is_int_or_none(self):
        loaded = ModelInfoResponse(
            model_name="hdb-predictor",
            model_alias="champion",
            model_version=2,
            run_id="abc123",
        )
        assert isinstance(loaded.model_version, int)

        unloaded = ModelInfoResponse(
            model_name="hdb-predictor",
            model_alias="champion",
            model_version=None,
            run_id=None,
        )
        assert unloaded.model_version is None
