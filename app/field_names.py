"""Canonical field identifiers shared by planning, mapping, and reduction."""

from __future__ import annotations

import re
import unicodedata

_ALIASES = {
    "area_units": "area_unit",
    "country_name": "country",
    "location_country": "country",
    "number_of_annual_visitors": "annual_visitor_count",
    "visitors_per_year": "annual_visitor_count",
    "year_of_establishment": "establishment_year",
}


def canonical_field_name(value: str) -> str:
    """Convert an LLM-produced field label to a stable snake_case key."""
    text = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("_")
    return _ALIASES.get(text, text)
