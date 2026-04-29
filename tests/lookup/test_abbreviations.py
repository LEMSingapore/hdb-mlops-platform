"""Tests for src/lookup/abbreviations.py."""

from lookup.abbreviations import expand_street_abbreviations


def test_lor_expands_to_lorong() -> None:
    assert expand_street_abbreviations("1 LOR 24 GEYLANG") == "1 LORONG 24 GEYLANG"


def test_jln_expands_to_jalan() -> None:
    assert expand_street_abbreviations("6A JLN GRISEK") == "6A JALAN GRISEK"


def test_rd_expands_to_road() -> None:
    assert expand_street_abbreviations("184 JELEBU RD") == "184 JELEBU ROAD"


def test_bt_expands_to_bukit() -> None:
    assert expand_street_abbreviations("1006 BT MERAH LANE 2") == "1006 BUKIT MERAH LANE 2"


def test_no_abbreviations_unchanged() -> None:
    addr = "1 STRAITS BOULEVARD"
    assert expand_street_abbreviations(addr) == addr


def test_whole_word_only_street_not_corrupted() -> None:
    """ST → STREET must not corrupt a token that already contains 'STREET'."""
    addr = "1 ORCHARD STREET"
    assert expand_street_abbreviations(addr) == "1 ORCHARD STREET"


def test_idempotent_with_abbreviations() -> None:
    addr = "BT BATOK ST 22"
    once = expand_street_abbreviations(addr)
    twice = expand_street_abbreviations(once)
    assert once == twice


def test_idempotent_no_abbreviations() -> None:
    addr = "BUKIT BATOK STREET 22"
    assert expand_street_abbreviations(addr) == expand_street_abbreviations(
        expand_street_abbreviations(addr)
    )


def test_multiple_abbreviations_in_one_string() -> None:
    assert expand_street_abbreviations("JLN BT MERAH") == "JALAN BUKIT MERAH"


def test_ave_expands_to_avenue() -> None:
    assert (
        expand_street_abbreviations("1 CHANGI BUSINESS PK AVE 1")
        == "1 CHANGI BUSINESS PARK AVENUE 1"
    )


def test_dr_expands_to_drive() -> None:
    assert expand_street_abbreviations("100B TOH YI DR") == "100B TOH YI DRIVE"


def test_cres_expands_to_crescent() -> None:
    assert expand_street_abbreviations("10 TEBAN GDNS CRES") == "10 TEBAN GARDENS CRESCENT"


def test_upp_expands_to_upper() -> None:
    assert expand_street_abbreviations("1 UPP CIRCULAR RD") == "1 UPPER CIRCULAR ROAD"


def test_ctrl_expands_to_central() -> None:
    assert expand_street_abbreviations("10 BT BATOK CTRL") == "10 BUKIT BATOK CENTRAL"


def test_empty_string_returns_empty() -> None:
    assert expand_street_abbreviations("") == ""
