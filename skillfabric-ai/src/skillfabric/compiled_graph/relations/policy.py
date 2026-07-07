"""Deterministic relation validation policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from skillfabric.compiled_graph.relations.models import CandidatePair

RelationDecision = Literal["accept", "reject", "llm"]
RELATION_POLICY_VERSION = "relation_validation_policy_v2"
RELATION_POLICY_DIGEST = RELATION_POLICY_VERSION


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

    del pair
    return RelationValidationDecision(action="llm", reason="candidate requires LLM validation")
