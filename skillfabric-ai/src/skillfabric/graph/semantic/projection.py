"""Project validated decisions into one acyclic semantic skill graph."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol

from skillfabric.graph.models import Edge
from skillfabric.graph.semantic.models import (
    GraphProjectionResult,
    RelationDecision,
)
from skillfabric.graph.semantic.prompts import build_cycle_adjudication_messages
from skillfabric.registry.models import SkillNode
from skillfabric.runtime.json_utils import parse_json_response
from skillfabric.runtime.llm import LLMConfig, litellm_completion

_CYCLE_ACTION_KEYS = frozenset({"pair_index", "action", "confidence", "reason"})
_CYCLE_ACTIONS = frozenset({"keep", "downgrade_to_compose", "remove"})


class DependencyCycleError(RuntimeError):
    """Raised when accepted dependency semantics remain cyclic."""


class CycleAdjudicator(Protocol):
    """Provider protocol for reviewing every decision in one dependency cycle."""

    def adjudicate(
        self,
        decisions: tuple[RelationDecision, ...],
        skills: dict[str, SkillNode],
    ) -> list[RelationDecision]:
        """Return one replacement decision for every cycle pair."""


@dataclass(slots=True)
class LiteLLMCycleAdjudicator:
    """Full-source LLM reviewer for dependency cycles."""

    config: LLMConfig

    @classmethod
    def from_env(cls, *, env_path: str | Path | None = None) -> LiteLLMCycleAdjudicator:
        return cls(LLMConfig.from_env(env_path=env_path))

    def adjudicate(
        self,
        decisions: tuple[RelationDecision, ...],
        skills: dict[str, SkillNode],
    ) -> list[RelationDecision]:
        response = litellm_completion(
            messages=build_cycle_adjudication_messages(decisions, skills),
            config=self.config,
        )
        payload = parse_json_response(response)
        if set(payload) != {"decisions"} or not isinstance(payload["decisions"], list):
            raise DependencyCycleError("cycle adjudication must return a decisions list")
        replacements: dict[int, RelationDecision] = {}
        for raw in payload["decisions"]:
            if not isinstance(raw, dict):
                raise DependencyCycleError("cycle replacement decisions must be objects")
            pair_index = raw.get("pair_index")
            if (
                isinstance(pair_index, bool)
                or not isinstance(pair_index, int)
                or pair_index < 0
                or pair_index >= len(decisions)
            ):
                raise DependencyCycleError("cycle adjudication returned an unknown pair_index")
            if pair_index in replacements:
                raise DependencyCycleError("cycle adjudication returned a duplicate pair_index")
            try:
                replacements[pair_index] = _decision_from_cycle_action(
                    decisions[pair_index],
                    raw,
                    pair_index=pair_index,
                )
            except ValueError as exc:
                raise DependencyCycleError(f"invalid cycle replacement: {exc}") from exc
        if set(replacements) != set(range(len(decisions))):
            raise DependencyCycleError(
                "cycle adjudication must replace every cycle pair exactly once"
            )
        return [replacements[index] for index in range(len(decisions))]


def _decision_from_cycle_action(
    original: RelationDecision,
    payload: dict[str, Any],
    *,
    pair_index: int,
) -> RelationDecision:
    actual_keys = set(payload)
    if actual_keys != _CYCLE_ACTION_KEYS:
        missing = _CYCLE_ACTION_KEYS - actual_keys
        unexpected = actual_keys - _CYCLE_ACTION_KEYS
        details = []
        if missing:
            details.append(f"missing keys: {', '.join(sorted(missing))}")
        if unexpected:
            details.append(f"unexpected keys: {', '.join(sorted(unexpected))}")
        raise ValueError("cycle action " + "; ".join(details))
    if payload["pair_index"] != pair_index:
        raise ValueError(f"cycle action must use pair_index {pair_index}")
    action = payload["action"]
    if not isinstance(action, str) or action not in _CYCLE_ACTIONS:
        raise ValueError("action must be keep, downgrade_to_compose, or remove")
    confidence = payload["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be a number")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    reason = payload["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")
    if original.relation != "depend_on":
        raise ValueError("cycle actions may only review depend_on decisions")

    updates: dict[str, Any] = {
        "confidence": confidence,
        "reason": reason.strip(),
    }
    if action == "downgrade_to_compose":
        updates["relation"] = "compose_with"
    elif action == "remove":
        updates.update(
            relation="none",
            source_skill=original.candidate.skill_a,
            target_skill=original.candidate.skill_b,
            evidence=(),
        )
    return replace(original, **updates)


def _validate_cycle_replacement(
    original: RelationDecision,
    replacement: RelationDecision,
) -> None:
    if replacement.relation not in {"depend_on", "compose_with", "none"}:
        raise DependencyCycleError(
            "cycle adjudication may only monotonically weaken depend_on decisions"
        )
    updates: dict[str, Any] = {
        "relation": replacement.relation,
        "confidence": replacement.confidence,
        "reason": replacement.reason,
    }
    if replacement.relation == "none":
        updates.update(
            source_skill=original.candidate.skill_a,
            target_skill=original.candidate.skill_b,
            evidence=(),
        )
    expected = replace(original, **updates)
    if replacement != expected:
        raise DependencyCycleError(
            "cycle adjudication may only monotonically weaken depend_on decisions"
        )


def project_relation_decisions(
    decisions: list[RelationDecision] | tuple[RelationDecision, ...],
    skills: list[SkillNode],
    *,
    cycle_adjudicator: CycleAdjudicator | None = None,
) -> GraphProjectionResult:
    """Project one decision per pair and require dependency acyclicity."""

    skills_by_id = {skill.id: skill for skill in skills}
    current: dict[tuple[str, str], RelationDecision] = {}
    for decision in decisions:
        key = decision.candidate.key
        if key in current:
            raise ValueError("each unordered skill pair must have one decision")
        _validate_projection_decision(decision, skills_by_id)
        current[key] = decision

    cycle_review_count = 0
    seen_dependency_sets: set[tuple[tuple[str, str], ...]] = set()
    while True:
        cycle_keys = _first_dependency_cycle(current.values())
        if not cycle_keys:
            break
        if cycle_adjudicator is None:
            raise DependencyCycleError(f"dependency cycle requires adjudication: {cycle_keys}")
        signature = tuple(
            sorted(
                (decision.source_skill, decision.target_skill)
                for decision in current.values()
                if decision.relation == "depend_on"
            )
        )
        if signature in seen_dependency_sets:
            raise DependencyCycleError("dependency cycle remained unresolved after adjudication")
        seen_dependency_sets.add(signature)
        cycle_decisions = tuple(current[key] for key in cycle_keys)
        replacements = cycle_adjudicator.adjudicate(cycle_decisions, skills_by_id)
        if len(replacements) != len(cycle_keys) or {
            decision.candidate.key for decision in replacements
        } != set(cycle_keys):
            raise DependencyCycleError(
                "cycle adjudicator must return one replacement for every cycle pair"
            )
        for replacement in replacements:
            _validate_cycle_replacement(current[replacement.candidate.key], replacement)
            _validate_projection_decision(replacement, skills_by_id)
            current[replacement.candidate.key] = replacement
        cycle_review_count += 1
        if cycle_review_count > max(1, len(current)):
            raise DependencyCycleError("dependency cycle remained unresolved after adjudication")

    ordered_decisions = tuple(current[key] for key in sorted(current))
    edges = tuple(
        sorted(
            (
                _edge_from_decision(decision)
                for decision in ordered_decisions
                if decision.relation != "none"
            ),
            key=lambda edge: (edge.type, edge.source, edge.target),
        )
    )
    return GraphProjectionResult(
        edges=edges,
        decisions=ordered_decisions,
        cycle_review_count=cycle_review_count,
    )


def _validate_projection_decision(
    decision: RelationDecision,
    skills: dict[str, SkillNode],
) -> None:
    if set(decision.candidate.key) - set(skills):
        raise ValueError("relation decision references an unknown skill")
    if {decision.source_skill, decision.target_skill} != set(decision.candidate.key):
        raise ValueError("relation decision endpoints must match its candidate pair")
    if (
        decision.relation in {"similar_to", "none"}
        and (decision.source_skill, decision.target_skill) != decision.candidate.key
    ):
        raise ValueError("alternative and none decisions must use canonical endpoint order")
    if decision.relation != "none":
        if not decision.evidence:
            raise ValueError("accepted semantic edges require evidence")
        if {item.skill for item in decision.evidence} != set(decision.candidate.key):
            raise ValueError("accepted semantic edges require evidence from both skills")


def _edge_from_decision(decision: RelationDecision) -> Edge:
    return Edge(
        source=decision.source_skill,
        target=decision.target_skill,
        type=decision.relation,
        confidence=decision.confidence,
        evidence=list(decision.evidence),
        reason=decision.reason,
    )


def _first_dependency_cycle(
    decisions: Any,
) -> tuple[tuple[str, str], ...]:
    adjacency: dict[str, list[str]] = {}
    pair_by_edge: dict[tuple[str, str], tuple[str, str]] = {}
    for decision in decisions:
        if decision.relation != "depend_on":
            continue
        adjacency.setdefault(decision.source_skill, []).append(decision.target_skill)
        pair_by_edge[(decision.source_skill, decision.target_skill)] = decision.candidate.key
    for targets in adjacency.values():
        targets.sort()

    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(node: str) -> tuple[tuple[str, str], ...]:
        state[node] = 1
        stack.append(node)
        for target in adjacency.get(node, []):
            if state.get(target, 0) == 0:
                found = visit(target)
                if found:
                    return found
            elif state.get(target) == 1:
                start = stack.index(target)
                cycle_nodes = [*stack[start:], target]
                return tuple(pair_by_edge[(left, right)] for left, right in pairwise(cycle_nodes))
        stack.pop()
        state[node] = 2
        return ()

    for node in sorted(adjacency):
        if state.get(node, 0) != 0:
            continue
        found = visit(node)
        if found:
            return found
    return ()
