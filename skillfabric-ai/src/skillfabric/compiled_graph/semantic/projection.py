"""Project validated decisions into one acyclic semantic skill graph."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol

from skillfabric.compiled_graph.models import Edge
from skillfabric.compiled_graph.semantic.models import (
    GraphProjectionResult,
    RelationDecision,
)
from skillfabric.compiled_graph.semantic.prompts import build_cycle_adjudication_messages
from skillfabric.compiled_graph.semantic.validation import decision_from_payload
from skillfabric.registry.models import SkillNode
from skillfabric.runtime.json_utils import parse_json_response
from skillfabric.runtime.llm import LLMConfig, litellm_completion


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
            usage_operation="graph.dependency_cycle",
            usage_metadata={"cycle_pair_count": len(decisions)},
        )
        payload = parse_json_response(response)
        if set(payload) != {"decisions"} or not isinstance(payload["decisions"], list):
            raise DependencyCycleError("cycle adjudication must return a decisions list")
        original = {decision.candidate.key: decision for decision in decisions}
        replacements: dict[tuple[str, str], RelationDecision] = {}
        for raw in payload["decisions"]:
            if not isinstance(raw, dict):
                raise DependencyCycleError("cycle replacement decisions must be objects")
            source = str(raw.get("source_skill", ""))
            target = str(raw.get("target_skill", ""))
            key = tuple(sorted((source, target)))
            decision = original.get(key)
            if decision is None:
                raise DependencyCycleError("cycle adjudication returned an unknown candidate pair")
            try:
                replacements[key] = decision_from_payload(
                    decision.candidate,
                    raw,
                    skills,
                )
            except ValueError as exc:
                raise DependencyCycleError(f"invalid cycle replacement: {exc}") from exc
        if set(replacements) != set(original):
            raise DependencyCycleError(
                "cycle adjudication must replace every cycle pair exactly once"
            )
        return [replacements[decision.candidate.key] for decision in decisions]


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
        if {decision.candidate.key for decision in replacements} != set(cycle_keys):
            raise DependencyCycleError(
                "cycle adjudicator must return one replacement for every cycle pair"
            )
        for replacement in replacements:
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
        decision.relation in {"compose_with", "similar_to", "none"}
        and (decision.source_skill, decision.target_skill) != decision.candidate.key
    ):
        raise ValueError("symmetric and none decisions must use canonical endpoint order")
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
