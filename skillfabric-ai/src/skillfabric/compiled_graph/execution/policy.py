"""Deterministic execution validation policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from skillfabric.compiled_graph.execution.models import ExecutionFlowCandidate

ExecutionDecision = Literal["accept", "reject", "llm"]
EXECUTION_POLICY_VERSION = "execution_validation_policy_v2"
EXECUTION_POLICY_DIGEST = EXECUTION_POLICY_VERSION


@dataclass(frozen=True, slots=True)
class ExecutionValidationDecision:
    """Deterministic action for an execution candidate before LLM validation."""

    action: ExecutionDecision
    accepted: bool = False
    flow_type: str = "none"
    projected_edge_type: str = "none"
    confidence: float = 0.0


def classify_execution_candidate(candidate: ExecutionFlowCandidate) -> ExecutionValidationDecision:
    """Classify an execution candidate into deterministic accept/reject or LLM validation."""

    del candidate
    return ExecutionValidationDecision(action="llm")
