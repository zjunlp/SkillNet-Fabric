"""Deterministic relation validation policy."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

from skillfabric.compiled_graph.relations.models import CandidatePair

RelationDecision = Literal["accept", "reject", "llm"]
RELATION_POLICY_VERSION = "relation_validation_policy_v1"
RELATION_POLICY_CONFIG = {
    "reject_similarity_only_prior_below": 0.72,
    "accept_execution_flow_prior_at_least": 0.95,
    "accept_execution_flow_min_evidence": 2,
    "accept_execution_flow_confidence": 0.92,
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
    if (
        "execution_flow" in sources
        and pair.prior >= RELATION_POLICY_CONFIG["accept_execution_flow_prior_at_least"]
        and pair.direction_hint in {"A->B", "B->A"}
        and len(pair.evidence) >= RELATION_POLICY_CONFIG["accept_execution_flow_min_evidence"]
    ):
        return RelationValidationDecision(
            action="accept",
            reason="Deterministic high-confidence execution flow relation.",
            edge_type="depend_on",
            direction=pair.direction_hint,
            confidence=RELATION_POLICY_CONFIG["accept_execution_flow_confidence"],
        )
    return RelationValidationDecision(action="llm", reason="candidate requires LLM validation")
