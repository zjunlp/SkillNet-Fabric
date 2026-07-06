"""Deterministic execution validation policy."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

from skillfabric.compiled_graph.execution.models import ExecutionFlowCandidate

ExecutionDecision = Literal["accept", "reject", "llm"]
EXECUTION_POLICY_VERSION = "execution_validation_policy_v1"
EXECUTION_POLICY_CONFIG = {
    "generic_handoff_names": sorted(
        {
            "data",
            "file",
            "object",
            "observation",
            "output",
            "result",
            "text",
        }
    ),
}
EXECUTION_POLICY_DIGEST = hashlib.sha256(
    json.dumps(
        {
            "version": EXECUTION_POLICY_VERSION,
            "config": EXECUTION_POLICY_CONFIG,
        },
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()

_GENERIC_HANDOFF_NAMES = {
    *EXECUTION_POLICY_CONFIG["generic_handoff_names"],
}


@dataclass(frozen=True, slots=True)
class ExecutionValidationDecision:
    """Deterministic action for an execution candidate before LLM validation."""

    action: ExecutionDecision
    reason: str
    accepted: bool = False
    flow_type: str = "none"
    projected_edge_type: str = "none"
    confidence: float = 0.0


def classify_execution_candidate(candidate: ExecutionFlowCandidate) -> ExecutionValidationDecision:
    """Classify an execution candidate into deterministic accept/reject or LLM validation."""

    matched_name = " ".join(candidate.matched_name.lower().replace("_", " ").split())
    if matched_name in _GENERIC_HANDOFF_NAMES:
        return ExecutionValidationDecision(
            action="reject",
            reason="deterministic generic execution handoff candidate",
        )
    return ExecutionValidationDecision(action="llm", reason="candidate requires LLM validation")
