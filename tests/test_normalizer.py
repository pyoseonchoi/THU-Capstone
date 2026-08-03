"""Tests for the value normalizer."""

from __future__ import annotations

import pytest

from app.reduction.normalizer import (
    normalize_date,
    normalize_entity_name,
    normalize_extracted_value,
    normalize_unit,
    parse_number,
)


class TestParseNumber:
    def test_integer(self):
        assert parse_number("1234") == 1234.0

    def test_float(self):
        assert parse_number("1234.56") == 1234.56

    def test_thousands_separator(self):
        assert parse_number("1,234,567") == 1234567.0

    def test_european_decimal(self):
        # "1234,56" with no dots -> European decimal
        assert parse_number("1234,56") == 1234.56

    def test_mixed_separators_english(self):
        # Period is decimal
        assert parse_number("1,234.56") == 1234.56

    def test_mixed_separators_european(self):
        # Comma is decimal
        assert parse_number("1.234,56") == 1234.56

    def test_percentage(self):
        assert parse_number("45.3%") == 45.3

    def test_negative_minus(self):
        assert parse_number("-123") == -123.0

    def test_negative_parens(self):
        assert parse_number("(456)") == -456.0

    def test_meter_suffix_is_not_treated_as_million(self):
        assert parse_number("4102m") is None

    def test_million_suffix(self):
        assert parse_number("1.5 million") == 1500000.0

    def test_billion_suffix(self):
        assert parse_number("2.3 billion") == 2300000000.0

    def test_currency(self):
        assert parse_number("$1,234.56") == 1234.56
        assert parse_number("€500") == 500.0

    def test_empty(self):
        assert parse_number("") is None
        assert parse_number("   ") is None

    def test_invalid(self):
        assert parse_number("not a number") is None

    def test_thousand_suffix(self):
        assert parse_number("5k") == 5000.0


class TestNormalizeUnit:
    def test_same_unit(self):
        assert normalize_unit(100.0, "km", "km") == 100.0

    def test_km_to_meters(self):
        result = normalize_unit(1.0, "km", "m")
        assert result == pytest.approx(1000.0)

    def test_miles_to_km(self):
        result = normalize_unit(1.0, "miles", "km")
        assert result == pytest.approx(1.609344)

    def test_hectares_to_km2(self):
        result = normalize_unit(100.0, "hectares", "km2")
        assert result == pytest.approx(1.0)

    def test_sq_miles_to_km2(self):
        result = normalize_unit(1.0, "square miles", "km2")
        assert result == pytest.approx(2.58999)

    def test_kg_to_grams(self):
        result = normalize_unit(1.0, "kg", "g")
        assert result == pytest.approx(1000.0)

    def test_unsupported_conversion(self):
        assert normalize_unit(1.0, "kg", "km") is None


class TestNormalizeDate:
    def test_iso(self):
        assert normalize_date("2023-01-15") == "2023-01-15"

    def test_slash_ymd(self):
        assert normalize_date("2023/1/15") == "2023-01-15"

    def test_slash_dmy(self):
        # 15/01/2023 -> DD/MM/YYYY
        assert normalize_date("15/01/2023") == "2023-01-15"

    def test_month_name_mdy(self):
        assert normalize_date("January 15, 2023") == "2023-01-15"

    def test_month_name_dmy(self):
        assert normalize_date("15 January 2023") == "2023-01-15"

    def test_year_only(self):
        assert normalize_date("2023") == "2023-01-01"

    def test_empty(self):
        assert normalize_date("") is None

    def test_invalid(self):
        assert normalize_date("not a date") is None

    def test_bc_year_preserves_era(self):
        assert normalize_date("first recorded in 475 BC") == "475 BC"


class TestNormalizeExtractedValue:
    def test_recomputes_scaled_number_from_raw_text(self):
        assert normalize_extracted_value(
            "15 million visitors",
            15,
            expected_type="number",
        ) == 15_000_000

    def test_bc_year_overrides_bad_model_normalization(self):
        assert normalize_extracted_value(
            "BC",
            0,
            expected_type="number",
            supporting_text="The first recorded eruption was in 475 BC.",
        ) == "475 BC"

    def test_embedded_metric_number_is_extracted(self):
        assert normalize_extracted_value(
            "4102m",
            None,
            expected_type="number",
        ) == 4102


class TestNormalizeEntityName:
    def test_strips_whitespace(self):
        assert normalize_entity_name("  Yellowstone Park  ") == "Yellowstone Park"

    def test_collapses_internal_spaces(self):
        assert normalize_entity_name("Grand   Canyon") == "Grand Canyon"

    def test_strips_trailing_punctuation(self):
        assert normalize_entity_name("Yosemite.") == "Yosemite"

    def test_preserves_case(self):
        assert normalize_entity_name("Mount Rainier") == "Mount Rainier"
