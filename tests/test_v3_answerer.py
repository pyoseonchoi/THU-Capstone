"""Regression tests for final-answer JSON parsing."""

from __future__ import annotations

import json

from app.v3.answerer import _parse_answer


def test_parse_answer_flattens_a_nested_object_instead_of_reprinting_it():
    """A model that ignores the "plain string" instruction must not leak
    Python dict syntax (braces, single quotes) into the final answer text.
    """
    raw = json.dumps({
        "answer": {
            "thesis": "Fire is both a natural process and a management tool.",
            "representative_examples": [
                {"station": "Kestrel Steppe Field Centre", "detail": "Patch burning by herders."},
            ],
        }
    })

    result = _parse_answer(raw)

    assert "{" not in result and "}" not in result
    assert "'" not in result
    assert "Fire is both a natural process and a management tool." in result
    assert "Kestrel Steppe Field Centre" in result
    assert "Patch burning by herders." in result


def test_parse_answer_passes_a_plain_string_through_unchanged():
    raw = json.dumps({"answer": "Mossbank Wetland Barrage, 7,360 sq km."})

    result = _parse_answer(raw)

    assert result == "Mossbank Wetland Barrage, 7,360 sq km."
