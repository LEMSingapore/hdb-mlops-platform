"""Tests for human-readable SHAP feature labels.

The formatter translates raw ``get_feature_names_out`` names into labels an LLM
client can present directly. These tests pin the three handled cases — numeric
pass-throughs and the month transformer, one-hot categoricals, and unknown
fall-through — plus the flat-type values whose normalisation is fiddly. They
also cover aggregating one-hot dummies back to their source categorical.
"""

import logging

from mcp_server.formatting import (
    aggregate_one_hot_contributions,
    format_feature_label,
)

_INPUT_VALUES = {
    "town": "TAMPINES",
    "flat_type": "4 ROOM",
    "floor_area_sqm": 95.0,
    "lease_commence_date": 1985,
    "month": "2024-06",
}


class TestNumericAndMonthLabels:
    def test_floor_area_interpolates_the_input_value(self) -> None:
        assert format_feature_label("num__floor_area_sqm", _INPUT_VALUES) == "Floor area (95.0 sqm)"

    def test_lease_commence_date_interpolates_the_input_value(self) -> None:
        assert (
            format_feature_label("num__lease_commence_date", _INPUT_VALUES)
            == "Lease start year (1985)"
        )

    def test_month_transformer_interpolates_the_input_value(self) -> None:
        assert (
            format_feature_label("month__float_time_series", _INPUT_VALUES)
            == "Transaction date (2024-06)"
        )


class TestOneHotTownLabels:
    def test_single_word_town_is_title_cased(self) -> None:
        assert format_feature_label("cat__town_TAMPINES", _INPUT_VALUES) == "Town: Tampines"

    def test_multi_word_town_is_title_cased(self) -> None:
        assert format_feature_label("cat__town_BUKIT BATOK", _INPUT_VALUES) == "Town: Bukit Batok"


class TestOneHotFlatTypeLabels:
    def test_room_count_becomes_hyphenated_lowercase(self) -> None:
        assert format_feature_label("cat__flat_type_4 ROOM", _INPUT_VALUES) == "Flat type: 4-room"

    def test_single_word_flat_type_is_capitalised(self) -> None:
        assert (
            format_feature_label("cat__flat_type_EXECUTIVE", _INPUT_VALUES)
            == "Flat type: Executive"
        )

    def test_hyphenated_flat_type_is_normalised(self) -> None:
        assert (
            format_feature_label("cat__flat_type_MULTI-GENERATION", _INPUT_VALUES)
            == "Flat type: Multi-generation"
        )

    def test_space_separated_flat_type_normalises_to_hyphen(self) -> None:
        assert (
            format_feature_label("cat__flat_type_MULTI GENERATION", _INPUT_VALUES)
            == "Flat type: Multi-generation"
        )

    def test_all_real_flat_types_format_sensibly(self) -> None:
        cases = {
            "cat__flat_type_2 ROOM": "Flat type: 2-room",
            "cat__flat_type_3 ROOM": "Flat type: 3-room",
            "cat__flat_type_4 ROOM": "Flat type: 4-room",
            "cat__flat_type_5 ROOM": "Flat type: 5-room",
            "cat__flat_type_EXECUTIVE": "Flat type: Executive",
            "cat__flat_type_MULTI GENERATION": "Flat type: Multi-generation",
            "cat__flat_type_MULTI-GENERATION": "Flat type: Multi-generation",
        }
        for raw_name, expected in cases.items():
            assert format_feature_label(raw_name, _INPUT_VALUES) == expected


class TestUnknownFeatures:
    def test_unknown_raw_name_falls_through_unchanged(self) -> None:
        assert (
            format_feature_label("some__unmapped_feature", _INPUT_VALUES)
            == "some__unmapped_feature"
        )

    def test_unknown_categorical_column_falls_through_unchanged(self) -> None:
        # flat_model is not a one-hot column the formatter knows about.
        assert (
            format_feature_label("cat__flat_model_Improved", _INPUT_VALUES)
            == "cat__flat_model_Improved"
        )


# Raw per-dummy SHAP contributions for a Tampines 4-room row: three town
# dummies, three flat-type dummies, two numerics, and the month feature.
_RAW_TAMPINES_4ROOM = {
    "cat__town_TAMPINES": 34666.0,
    "cat__town_BEDOK": -1030.0,
    "cat__town_BISHAN": -3320.0,
    "cat__flat_type_3 ROOM": 15312.0,
    "cat__flat_type_4 ROOM": -9634.0,
    "cat__flat_type_5 ROOM": -12900.0,
    "num__floor_area_sqm": 15807.0,
    "num__lease_commence_date": -20353.0,
    "month__float_time_series": 246378.0,
}
_INPUT_TAMPINES_4ROOM = {"town": "TAMPINES", "flat_type": "4 ROOM"}


class TestAggregateOneHotContributions:
    def test_town_dummies_collapse_to_active_category_net_sum(self) -> None:
        aggregated = aggregate_one_hot_contributions(_RAW_TAMPINES_4ROOM, _INPUT_TAMPINES_4ROOM)
        town_keys = [k for k in aggregated if k.startswith("cat__town_")]
        assert town_keys == ["cat__town_TAMPINES"]
        assert aggregated["cat__town_TAMPINES"] == 34666.0 - 1030.0 - 3320.0

    def test_flat_type_dummies_collapse_to_active_category_net_sum(self) -> None:
        aggregated = aggregate_one_hot_contributions(_RAW_TAMPINES_4ROOM, _INPUT_TAMPINES_4ROOM)
        flat_keys = [k for k in aggregated if k.startswith("cat__flat_type_")]
        assert flat_keys == ["cat__flat_type_4 ROOM"]
        assert aggregated["cat__flat_type_4 ROOM"] == 15312.0 - 9634.0 - 12900.0

    def test_numeric_and_month_features_pass_through_unchanged(self) -> None:
        aggregated = aggregate_one_hot_contributions(_RAW_TAMPINES_4ROOM, _INPUT_TAMPINES_4ROOM)
        assert aggregated["num__floor_area_sqm"] == 15807.0
        assert aggregated["num__lease_commence_date"] == -20353.0
        assert aggregated["month__float_time_series"] == 246378.0

    def test_aggregated_structure_is_one_entry_per_active_categorical(self) -> None:
        aggregated = aggregate_one_hot_contributions(_RAW_TAMPINES_4ROOM, _INPUT_TAMPINES_4ROOM)
        # Two active categoricals + two numerics + one month feature.
        assert set(aggregated) == {
            "cat__town_TAMPINES",
            "cat__flat_type_4 ROOM",
            "num__floor_area_sqm",
            "num__lease_commence_date",
            "month__float_time_series",
        }

    def test_unknown_active_value_falls_back_to_per_dummy_entries(self, caplog) -> None:
        raw = {
            "cat__town_TAMPINES": 34666.0,
            "cat__town_BEDOK": -1030.0,
            "num__floor_area_sqm": 15807.0,
        }
        # No cat__town_PUNGGOL dummy exists for an input outside the group.
        with caplog.at_level(logging.DEBUG, logger="mcp_server.formatting"):
            aggregated = aggregate_one_hot_contributions(raw, {"town": "PUNGGOL"})
        # Both town dummies are preserved rather than silently dropped.
        assert aggregated["cat__town_TAMPINES"] == 34666.0
        assert aggregated["cat__town_BEDOK"] == -1030.0
        assert aggregated["num__floor_area_sqm"] == 15807.0
        assert any(record.levelno == logging.DEBUG for record in caplog.records)
