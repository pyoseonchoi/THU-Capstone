"""Source-literal matching helpers shared by the V3 LLM stages.

Every LLM stage in V3 must prove its output against the document text rather
than trusting the model, so normalization and JSON recovery live here instead
of being duplicated per stage.
"""

from __future__ import annotations

import html
import json
import re
import unicodedata
from typing import Any

_TRANSLATIONS = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        " ": " ",
    }
)


def normalize_source(text: str) -> str:
    """Fold typography and whitespace so quotes compare against the source."""
    normalized = unicodedata.normalize("NFKC", html.unescape(text)).translate(_TRANSLATIONS)
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def json_payload(raw: str) -> Any:
    """Parse a JSON response, tolerating code fences and surrounding prose."""
    text = raw.strip()
    if "```" in text:
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(1))
