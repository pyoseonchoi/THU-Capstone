"""Tests for resilient verifier parsing."""

import json

from app.llm.base import LLMResponse
from app.llm.usage_tracker import UsageTracker
from app.verification.claim_verifier import ClaimVerifier


class _Router:
    async def chat(self, *args, **kwargs):
        return LLMResponse(content=json.dumps({
            "claims": [{
                "claim": "A claim",
                "status": "SUPPORTED",
                "supporting_evidence_ids": None,
                "operation_id": None,
                "correction": None,
            }],
            "overall_status": "PASS",
        }))


def test_verifier_accepts_nullable_optional_fields():
    verifier = ClaimVerifier(_Router(), UsageTracker())

    claims, needs_repair = verifier._parse_verification(json.dumps({
        "claims": [{
            "claim": "A claim",
            "status": "SUPPORTED",
            "supporting_evidence_ids": None,
            "operation_id": None,
            "correction": None,
        }],
        "overall_status": "PASS",
    }))

    assert needs_repair is False
    assert claims[0].operation_id == ""
    assert claims[0].correction == ""
