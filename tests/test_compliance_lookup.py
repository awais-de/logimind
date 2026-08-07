"""Unit tests for agents/tools/compliance_lookup.py."""

from agents.tools.compliance_lookup import compliance_lookup


def test_returns_matched_rule() -> None:
    result = compliance_lookup("lithium_batteries", "DE")

    assert result is not None
    assert result["category"] == "lithium_batteries"
    assert result["destination"] == "DE"
    assert result["restrictions"]
    assert result["documentation_required"]


def test_returns_none_for_unmatched_combination() -> None:
    result = compliance_lookup("lithium_batteries", "NL")

    assert result is None


def test_returns_none_for_unknown_category() -> None:
    result = compliance_lookup("used_furniture", "DE")

    assert result is None


def test_matches_case_and_whitespace_insensitively() -> None:
    result = compliance_lookup("Lithium Batteries", "de")

    assert result is not None
    assert result["category"] == "lithium_batteries"


def test_matches_destination_from_city_country_string() -> None:
    result = compliance_lookup("alcohol", "Leipzig, DE")

    assert result is not None
    assert result["destination"] == "DE"
