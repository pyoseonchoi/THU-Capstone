"""Deterministic value and unit normalization.

Supports: numbers with various separators, percentages, dates,
distance, area, mass, currency, and scale suffixes.

Both raw_value and normalized_value are always preserved.
The raw source value is never overwritten.
"""

from __future__ import annotations

import re
from typing import Optional

from app.logging_config import get_logger

logger = get_logger("reduction.normalizer")

# ---------------------------------------------------------------------------
# Number parsing
# ---------------------------------------------------------------------------

_SCALE_SUFFIXES = {
    "million": 1_000_000,
    "mln": 1_000_000,
    "mil": 1_000_000,
    "m": 1_000_000,
    "billion": 1_000_000_000,
    "bln": 1_000_000_000,
    "bil": 1_000_000_000,
    "b": 1_000_000_000,
    "trillion": 1_000_000_000_000,
    "thousand": 1_000,
    "k": 1_000,
}


def parse_number(raw: str) -> Optional[float]:
    """Parse a numeric string with various formats.

    Handles:
    - Commas as thousands separators: "1,234,567" → 1234567
    - Periods as decimals: "1234.56" → 1234.56
    - European comma-as-decimal: "1234,56" (when no dots present) → 1234.56
    - Percentage signs: "45.3%" → 45.3
    - Scale suffixes: "1.5 million" → 1500000
    - Negative numbers with minus or parentheses

    Returns:
        Parsed float or None if unparseable.
    """
    if not raw or not raw.strip():
        return None

    text = raw.strip().lower()

    # Remove currency symbols and whitespace variants
    text = re.sub(r"[$€£¥₩]", "", text)
    text = text.replace("\u00a0", " ").replace("\u2009", " ")

    # Handle parentheses for negatives: (123) → -123
    negative = False
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
        negative = True
    if text.startswith("-"):
        text = text[1:]
        negative = True

    # Strip percentage sign (caller should note the unit)
    text = text.replace("%", "").strip()

    # Detect and apply scale suffix
    scale = 1.0
    for suffix, factor in sorted(_SCALE_SUFFIXES.items(), key=lambda x: -len(x[0])):
        if text.endswith(suffix):
            candidate = text[: -len(suffix)].strip()
            if candidate:
                text = candidate
                scale = factor
                break
        # Also handle "1.5 million" with space
        if f" {suffix}" in text:
            text = text.replace(f" {suffix}", "").strip()
            scale = factor
            break

    # Determine decimal separator
    # If both comma and period exist, the later one is the decimal
    has_comma = "," in text
    has_period = "." in text

    if has_comma and has_period:
        # Determine which is the decimal: the one that appears last
        last_comma = text.rfind(",")
        last_period = text.rfind(".")
        if last_period > last_comma:
            # Period is decimal (English): remove commas
            text = text.replace(",", "")
        else:
            # Comma is decimal (European): remove periods, replace comma
            text = text.replace(".", "").replace(",", ".")
    elif has_comma:
        # Check if comma could be decimal (e.g., "1234,56")
        parts = text.split(",")
        if len(parts) == 2 and len(parts[1]) <= 2 and parts[1].isdigit():
            # Likely European decimal
            text = text.replace(",", ".")
        else:
            # Likely thousands separator
            text = text.replace(",", "")

    # Remove remaining whitespace
    text = text.replace(" ", "")

    try:
        value = float(text) * scale
        if negative:
            value = -value
        return value
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Unit normalization
# ---------------------------------------------------------------------------

# Conversion factors to base units
_DISTANCE_TO_METERS = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "metre": 1.0,
    "metres": 1.0,
    "km": 1000.0,
    "kilometer": 1000.0,
    "kilometers": 1000.0,
    "kilometre": 1000.0,
    "kilometres": 1000.0,
    "mi": 1609.344,
    "mile": 1609.344,
    "miles": 1609.344,
    "ft": 0.3048,
    "foot": 0.3048,
    "feet": 0.3048,
}

_AREA_TO_KM2 = {
    "km2": 1.0,
    "km²": 1.0,
    "sq km": 1.0,
    "square kilometer": 1.0,
    "square kilometers": 1.0,
    "square kilometre": 1.0,
    "square kilometres": 1.0,
    "mi2": 2.58999,
    "sq mi": 2.58999,
    "square mile": 2.58999,
    "square miles": 2.58999,
    "ha": 0.01,
    "hectare": 0.01,
    "hectares": 0.01,
    "acre": 0.00404686,
    "acres": 0.00404686,
}

_MASS_TO_KG = {
    "kg": 1.0,
    "kilogram": 1.0,
    "kilograms": 1.0,
    "g": 0.001,
    "gram": 0.001,
    "grams": 0.001,
    "lb": 0.453592,
    "lbs": 0.453592,
    "pound": 0.453592,
    "pounds": 0.453592,
    "ton": 1000.0,
    "tons": 1000.0,
    "tonne": 1000.0,
    "tonnes": 1000.0,
}


def normalize_unit(
    value: float,
    from_unit: str,
    to_unit: str,
) -> Optional[float]:
    """Convert a value from one unit to another.

    Supports distance (to meters), area (to km²), and mass (to kg).

    Returns:
        Converted value or None if conversion is unsupported.
    """
    from_lower = from_unit.lower().strip()
    to_lower = to_unit.lower().strip()

    if from_lower == to_lower:
        return value

    # Try distance
    if from_lower in _DISTANCE_TO_METERS and to_lower in _DISTANCE_TO_METERS:
        meters = value * _DISTANCE_TO_METERS[from_lower]
        return meters / _DISTANCE_TO_METERS[to_lower]

    # Try area
    if from_lower in _AREA_TO_KM2 and to_lower in _AREA_TO_KM2:
        km2 = value * _AREA_TO_KM2[from_lower]
        return km2 / _AREA_TO_KM2[to_lower]

    # Try mass
    if from_lower in _MASS_TO_KG and to_lower in _MASS_TO_KG:
        kg = value * _MASS_TO_KG[from_lower]
        return kg / _MASS_TO_KG[to_lower]

    logger.warning("Cannot convert from '%s' to '%s'", from_unit, to_unit)
    return None


# ---------------------------------------------------------------------------
# Date normalization
# ---------------------------------------------------------------------------

_MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "september": 9, "sept": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def normalize_date(raw: str) -> Optional[str]:
    """Normalize a date string to ISO format YYYY-MM-DD.

    Handles:
    - YYYY-MM-DD, YYYY/MM/DD
    - DD/MM/YYYY, MM/DD/YYYY (assumes DD/MM/YYYY for ambiguous)
    - "January 15, 2023", "15 January 2023"
    - Year only: "2023"

    Returns:
        ISO date string or None.
    """
    text = raw.strip()
    if not text:
        return None

    # Already ISO
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", text)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    # YYYY/MM/DD
    m = re.match(r"^(\d{4})[/.](\d{1,2})[/.](\d{1,2})$", text)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    # DD/MM/YYYY or MM/DD/YYYY
    m = re.match(r"^(\d{1,2})[/.](\d{1,2})[/.](\d{4})$", text)
    if m:
        a, b, year = int(m.group(1)), int(m.group(2)), m.group(3)
        if a > 12:
            day, month = a, b
        elif b > 12:
            month, day = a, b
        else:
            # Ambiguous: assume DD/MM/YYYY
            day, month = a, b
        return f"{year}-{month:02d}-{day:02d}"

    # "Month DD, YYYY" or "DD Month YYYY"
    for month_name, month_num in _MONTH_MAP.items():
        # "January 15, 2023"
        pattern = rf"\b{month_name}\b\.?\s+(\d{{1,2}}),?\s+(\d{{4}})"
        m2 = re.search(pattern, text.lower())
        if m2:
            return f"{m2.group(2)}-{month_num:02d}-{int(m2.group(1)):02d}"
        # "15 January 2023"
        pattern = rf"(\d{{1,2}})\s+{month_name}\b\.?\s+(\d{{4}})"
        m2 = re.search(pattern, text.lower())
        if m2:
            return f"{m2.group(2)}-{month_num:02d}-{int(m2.group(1)):02d}"

    # Year only
    m = re.match(r"^(\d{4})$", text)
    if m:
        return f"{m.group(1)}-01-01"

    return None


# ---------------------------------------------------------------------------
# Entity name normalization
# ---------------------------------------------------------------------------

def normalize_entity_name(name: str) -> str:
    """Conservatively normalize an entity name.

    Strips whitespace, normalizes internal spacing, preserves case.
    Does NOT aggressively merge similar names.
    """
    # Collapse whitespace
    text = re.sub(r"\s+", " ", name.strip())
    # Remove trailing punctuation
    text = text.rstrip(".,;:")
    return text
