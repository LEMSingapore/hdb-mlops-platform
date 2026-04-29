"""Tests for src/lookup/postal.py."""

from lookup.postal import AddressInfo, lookup_postal


class TestLookupPostalKnownCodes:
    def test_geylang_by_integer(self) -> None:
        result = lookup_postal(398614)
        assert result is not None
        assert result.postal_code == "398614"
        assert result.block == "1"
        assert result.street_abbreviated == "LOR 24 GEYLANG"

    def test_geylang_street_full_expands_lor(self) -> None:
        result = lookup_postal(398614)
        assert result is not None
        assert result.street_full == "LORONG 24 GEYLANG"

    def test_lowest_postal_by_string(self) -> None:
        result = lookup_postal("018907")
        assert result is not None
        assert result.postal_code == "018907"
        assert result.block == "11A"

    def test_lowest_postal_no_abbreviations_abbreviated_equals_full(self) -> None:
        result = lookup_postal("018907")
        assert result is not None
        assert result.street_abbreviated == result.street_full

    def test_address_with_abbreviation_abbreviated_differs_from_full(self) -> None:
        result = lookup_postal(398614)
        assert result is not None
        assert result.street_abbreviated != result.street_full

    def test_town_is_populated_for_known_hdb_street(self) -> None:
        # 659592 is block 1 at BT BATOK ST 22 (BUKIT BATOK STREET 22).
        # Both abbreviations expand and the street maps to BUKIT BATOK in the
        # resale training data, so town should be populated.
        result = lookup_postal(659592)
        assert result is not None
        assert result.town == "BUKIT BATOK"

    def test_town_is_none_when_street_not_in_mapping(self) -> None:
        # 398614 is LOR 24 GEYLANG — present in the postal CSV but the block at
        # that address is not an HDB resale flat, so it does not appear in the
        # resale CSVs and therefore has no entry in the street→town mapping.
        result = lookup_postal(398614)
        assert result is not None
        assert result.town is None


class TestLookupPostalFivedigit:
    def test_five_digit_integer(self) -> None:
        result = lookup_postal(18907)
        assert result is not None
        assert result.postal_code == "018907"

    def test_five_digit_string(self) -> None:
        result = lookup_postal("18907")
        assert result is not None
        assert result.postal_code == "018907"

    def test_five_digit_and_six_digit_return_same_result(self) -> None:
        assert lookup_postal(18907) == lookup_postal("018907")


class TestLookupPostalNotFound:
    def test_nonexistent_postal_returns_none(self) -> None:
        assert lookup_postal(99999999) is None

    def test_zero_returns_none(self) -> None:
        assert lookup_postal(0) is None

    def test_string_not_in_data_returns_none(self) -> None:
        assert lookup_postal("000000") is None


class TestAddressInfoModel:
    def test_result_is_address_info_instance(self) -> None:
        result = lookup_postal(398614)
        assert isinstance(result, AddressInfo)

    def test_postal_code_is_zero_padded_six_digits(self) -> None:
        result = lookup_postal("18907")
        assert result is not None
        assert len(result.postal_code) == 6
        assert result.postal_code == "018907"
