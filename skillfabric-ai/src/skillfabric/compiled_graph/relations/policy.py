"""Deterministic relation validation policy."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

from skillfabric.compiled_graph.relations.models import CandidatePair

RelationDecision = Literal["accept", "reject", "llm"]
RELATION_POLICY_VERSION = "relation_validation_policy_v2"
RELATION_POLICY_CONFIG = {
    "reject_similarity_only_prior_below": 0.72,
}
RELATION_POLICY_DIGEST = hashlib.sha256(
    json.dumps(
        {
            "version": RELATION_POLICY_VERSION,
            "config": RELATION_POLICY_CONFIG,
        },
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()


@dataclass(frozen=True, slots=True)
class RelationValidationDecision:
    """Deterministic action for a relation candidate before LLM validation."""

    action: RelationDecision
    reason: str
    edge_type: str = "none"
    direction: str = "none"
    confidence: float = 0.0


def classify_relation_candidate(pair: CandidatePair) -> RelationValidationDecision:
    """Classify a candidate into deterministic accept/reject or LLM validation."""

    sources = set(pair.sources)
    if (
        sources == {"similar_neighbor"}
        and pair.prior < RELATION_POLICY_CONFIG["reject_similarity_only_prior_below"]
        and not pair.evidence
    ):
        return RelationValidationDecision(
            action="reject",
            reason="deterministic low-confidence similarity-only relation candidate",
        )
    return RelationValidationDecision(action="llm", reason="candidate requires LLM validation")
