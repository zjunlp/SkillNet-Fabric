"""Shared AgentRunSpec contract for Claude Code and Codex execution prompts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from skillfabric.router.models import RouteResult, RouteSelectedSkill
from skillfabric.wiki.pages import slug

EXECUTION_OPERATIONS = {
    "orient",
    "inspect",
    "apply_skill",
    "produce",
    "verify",
    "delegate",
    "parallelize",
    "aggregate",
    "report",
}

EXECUTION_CONTROLS = {"serial", "parallel", "conditional"}


@dataclass(slots=True)
class AgentRunSelectedSkill:
    """One selected skill exposed to an executing coding agent."""

    skill_id: str
    name: str
    native_skill_name: str
    role: str
    skill_context_path: str

    def to_dict(self) -> dict[str, str]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "native_skill_name": self.native_skill_name,
            "role": self.role,
            "skill_context_path": self.skill_context_path,
        }


@dataclass(slots=True)
class ExecutionOperation:
    """Prompt-level operation guidance for an external coding agent."""

    id: str
    operation: str
    control: str
    goal: str
    skill_ids: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    guidance: str = ""
    expected_outputs: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.operation not in EXECUTION_OPERATIONS:
            raise ValueError(f"unknown execution operation: {self.operation}")
        if self.control not in EXECUTION_CONTROLS:
            raise ValueError(f"unknown execution control: {self.control}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "operation": self.operation,
            "control": self.control,
            "goal": self.goal,
            "skill_ids": list(self.skill_ids),
            "depends_on": list(self.depends_on),
            "guidance": self.guidance,
            "expected_outputs": list(self.expected_outputs),
        }


@dataclass(slots=True)
class ExecutionStrategy:
    """Shared semantic execution strategy rendered into agent-specific prompts."""

    operations: list[ExecutionOperation]
    parallelization_policy: str
    delegation_policy: str
    verification_policy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "operations": [item.to_dict() for item in self.operations],
            "parallelization_policy": self.parallelization_policy,
            "delegation_policy": self.delegation_policy,
            "verification_policy": self.verification_policy,
        }


@dataclass(slots=True)
class AgentRunPhase:
    """Coarse execution phase for an external coding agent."""

    id: str
    goal: str
    skill_ids: list[str]
    depends_on: list[str] = field(default_factory=list)
    required_inputs: list[str] = field(default_factory=list)
    expected_outputs: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    guidance: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "skill_ids": list(self.skill_ids),
            "depends_on": list(self.depends_on),
            "required_inputs": list(self.required_inputs),
            "expected_outputs": list(self.expected_outputs),
            "evidence_refs": list(self.evidence_refs),
            "guidance": self.guidance,
        }


@dataclass(slots=True)
class AgentRunRequiredOrder:
    """Skill ordering constraint from route evidence."""

    before_skill: str
    after_skill: str
    reason: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "before_skill": self.before_skill,
            "after_skill": self.after_skill,
            "reason": self.reason,
        }


@dataclass(slots=True)
class AgentRunAcceptanceCriterion:
    """Final acceptance criterion for generated task deliverables."""

    id: str
    description: str
    expected_artifacts: list[str] = field(default_factory=list)
    covered_by: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "expected_artifacts": list(self.expected_artifacts),
            "covered_by": list(self.covered_by),
        }


@dataclass(slots=True)
class AgentRunSpec:
    """Execution contract handed to Claude Code or Codex."""

    objective: str
    selected_skills: list[AgentRunSelectedSkill]
    phases: list[AgentRunPhase]
    execution_strategy: ExecutionStrategy
    required_order: list[AgentRunRequiredOrder] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    acceptance_criteria: list[AgentRunAcceptanceCriterion] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "selected_skills": [item.to_dict() for item in self.selected_skills],
            "phases": [item.to_dict() for item in self.phases],
            "execution_strategy": self.execution_strategy.to_dict(),
            "required_order": [item.to_dict() for item in self.required_order],
            "constraints": list(self.constraints),
            "acceptance_criteria": [item.to_dict() for item in self.acceptance_criteria],
        }


def agent_run_spec_from_route(route: RouteResult) -> AgentRunSpec:
    """Build the shared execution prompt contract from a RouteResult."""

    selected_skills = [
        AgentRunSelectedSkill(
            skill_id=skill.skill_id,
            name=skill.name,
            native_skill_name=slug(skill.skill_id),
            role=skill.reason or f"Use {skill.name} for the task.",
            skill_context_path=f"selected_skills/{slug(skill.skill_id)}.md",
        )
        for skill in route.selected_skills
    ]
    phases = _phases_from_route(route)
    acceptance_criteria = _acceptance_criteria(route)
    return AgentRunSpec(
        objective=route.query,
        selected_skills=selected_skills,
        phases=phases,
        execution_strategy=_execution_strategy_from_route(route, phases, acceptance_criteria),
        required_order=[
            AgentRunRequiredOrder(
                before_skill=edge.before_skill,
                after_skill=edge.after_skill,
                reason=edge.reason,
            )
            for edge in route.required_edges
        ],
        constraints=[
            "Follow required_order unless blocked; explain any deviation in the final report.",
            "Execute the user's task directly in the active workspace.",
        ],
        acceptance_criteria=acceptance_criteria,
    )


def _phases_from_route(route: RouteResult) -> list[AgentRunPhase]:
    ordered_skills = _order_skills(route.selected_skills, route.required_edges)
    dependency_phases: dict[str, list[str]] = {skill.skill_id: [] for skill in ordered_skills}
    skill_to_phase = {skill.skill_id: f"phase_{index + 1}" for index, skill in enumerate(ordered_skills)}
    for edge in route.required_edges:
        before_phase = skill_to_phase.get(edge.before_skill)
        after_phase = skill_to_phase.get(edge.after_skill)
        if before_phase and after_phase:
            dependency_phases.setdefault(edge.after_skill, []).append(before_phase)
    phases: list[AgentRunPhase] = []
    for index, skill in enumerate(ordered_skills, start=1):
        phases.append(
            AgentRunPhase(
                id=f"phase_{index}",
                goal=skill.reason or f"Apply {skill.name} to the objective.",
                skill_ids=[skill.skill_id],
                depends_on=sorted(set(dependency_phases.get(skill.skill_id, [])), key=_phase_sort_key),
                required_inputs=[],
                expected_outputs=_expected_outputs(skill),
                evidence_refs=list(skill.evidence),
                guidance=f"Use {skill.name} only for the portion of the task described by its role.",
            )
        )
    return phases


def _order_skills(skills: list[RouteSelectedSkill], required_edges: list[Any]) -> list[RouteSelectedSkill]:
    by_id = {skill.skill_id: skill for skill in skills}
    original_rank = {skill.skill_id: index for index, skill in enumerate(skills)}
    adjacency: dict[str, set[str]] = {skill.skill_id: set() for skill in skills}
    in_degree: dict[str, int] = {skill.skill_id: 0 for skill in skills}
    for edge in required_edges:
        before_skill = str(getattr(edge, "before_skill", ""))
        after_skill = str(getattr(edge, "after_skill", ""))
        if before_skill not in by_id or after_skill not in by_id:
            continue
        if after_skill not in adjacency[before_skill]:
            adjacency[before_skill].add(after_skill)
            in_degree[after_skill] += 1
    queue = sorted([skill_id for skill_id, degree in in_degree.items() if degree == 0], key=lambda item: original_rank[item])
    output: list[str] = []
    while queue:
        skill_id = queue.pop(0)
        output.append(skill_id)
        for neighbor in sorted(adjacency[skill_id], key=lambda item: original_rank[item]):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
                queue.sort(key=lambda item: original_rank[item])
    if len(output) != len(skills):
        return skills
    return [by_id[skill_id] for skill_id in output]


def _execution_strategy_from_route(
    route: RouteResult,
    phases: list[AgentRunPhase],
    acceptance_criteria: list[AgentRunAcceptanceCriterion],
) -> ExecutionStrategy:
    operations: list[ExecutionOperation] = [
        ExecutionOperation(
            id="op_orient",
            operation="orient",
            control="serial",
            goal="Understand the objective, selected skills, deliverables, and constraints before changing files.",
            guidance=(
                "Identify deliverables, inputs, required operations, constraints, verification signals, and any coverage gaps."
            ),
            expected_outputs=["A concise working plan that maps selected skills to task stages and deliverables."],
        ),
        ExecutionOperation(
            id="op_inspect",
            operation="inspect",
            control="serial",
            goal="Read the relevant workspace context and task inputs before applying selected capability roles.",
            depends_on=["op_orient"],
            guidance=(
                "Inspect only the files needed to complete the objective. "
                "Confirm the selected capability roles apply to the inspected inputs and requested deliverables."
            ),
            expected_outputs=["A clear understanding of inputs, existing artifacts, applicable skill guidance, and required deliverables."],
        ),
    ]
    phase_operation_ids: dict[str, str] = {}
    for index, phase in enumerate(phases, start=1):
        operation_id = f"op_apply_skill_{index}"
        dependencies = [
            phase_operation_ids[dependency]
            for dependency in phase.depends_on
            if dependency in phase_operation_ids
        ]
        if not dependencies:
            dependencies = ["op_inspect"]
        control = "serial" if phase.depends_on else "parallel"
        phase_operation_ids[phase.id] = operation_id
        operations.append(
            ExecutionOperation(
                id=operation_id,
                operation="apply_skill",
                control=control,
                goal=phase.goal,
                skill_ids=list(phase.skill_ids),
                depends_on=dependencies,
                guidance=_operation_guidance(phase),
                expected_outputs=list(phase.expected_outputs),
            )
        )
    independent_operations = [
        operation
        for operation in operations
        if operation.operation == "apply_skill" and operation.depends_on == ["op_inspect"]
    ]
    if len(independent_operations) > 1:
        operations.append(
            ExecutionOperation(
                id="op_parallelize",
                operation="parallelize",
                control="parallel",
                goal="Run independent exploration, implementation, or verification work in parallel when it will reduce latency without coupling edits.",
                skill_ids=_dedupe([skill_id for item in independent_operations for skill_id in item.skill_ids]),
                depends_on=["op_inspect"],
                guidance="Parallelize only independent work. Keep file edits disjoint, avoid duplicated skill application, and merge the results before final verification.",
                expected_outputs=["Independent findings or artifacts ready for aggregation."],
            )
        )
    if route.selected_skills:
        operations.append(
            ExecutionOperation(
                id="op_produce",
                operation="produce",
                control="serial",
                goal="Create or modify the concrete deliverables requested by the objective.",
                skill_ids=list(route.selected_skill_ids),
                depends_on=[operation.id for operation in operations if operation.operation == "apply_skill"],
                guidance="Use selected skills as execution guidance to produce the requested artifacts directly in the workspace. Handle unsupported gaps directly and document them.",
                expected_outputs=_expected_artifacts_from_criteria(acceptance_criteria) or ["Final task deliverables."],
            )
        )
    if len(route.selected_skills) > 1:
        operations.append(
            ExecutionOperation(
                id="op_aggregate",
                operation="aggregate",
                control="serial",
                goal="Combine results from selected skills into coherent final deliverables.",
                skill_ids=list(route.selected_skill_ids),
                depends_on=["op_produce"] if route.selected_skills else [],
                guidance="Resolve overlaps or conflicts between skill outputs, remove redundant intermediate artifacts when appropriate, and ensure final deliverables satisfy the original objective.",
                expected_outputs=["Integrated final deliverables."],
            )
        )
    verify_dep = "op_aggregate" if len(route.selected_skills) > 1 else "op_produce"
    operations.append(
        ExecutionOperation(
            id="op_verify",
            operation="verify",
            control="serial",
            goal="Verify the deliverables and any code changes before reporting completion.",
            skill_ids=list(route.selected_skill_ids),
            depends_on=[verify_dep] if route.selected_skills else ["op_inspect"],
            guidance="Run relevant tests, builds, lint checks, file inspections, schema checks, or artifact checks. Report exact blockers for any verification that could not be run.",
            expected_outputs=["Fresh verification evidence."],
        )
    )
    operations.append(
        ExecutionOperation(
            id="op_report",
            operation="report",
            control="serial",
            goal="Write the final report requested by the execution prompt.",
            skill_ids=list(route.selected_skill_ids),
            depends_on=["op_verify"],
            guidance="Summarize completed work, skills actually used, deliverables, verification evidence, coverage gaps, deviations, and blocking issues.",
            expected_outputs=["Concise final response."],
        )
    )
    return ExecutionStrategy(
        operations=operations,
        parallelization_policy=(
            "Use parallel execution only for independent inspection, verification, or disjoint implementation work. "
            "Keep dependent skill work serial according to required_order."
        ),
        delegation_policy=(
            "delegate is optional. Use a subagent only for a bounded, independent exploration, verification, "
            "or disjoint implementation task that does not block the immediate critical path."
        ),
        verification_policy=(
            "Do not claim completion without fresh verification evidence from tests, builds, artifact inspection, "
            "or an explicit explanation of why verification could not be run."
        ),
    )


def _operation_guidance(phase: AgentRunPhase) -> str:
    guidance = phase.guidance or "Apply the selected skill to its assigned portion of the task."
    guidance = (
        f"{guidance} Confirm this skill is applicable after inspecting its context; if it is not, skip it and record why."
    )
    if phase.expected_outputs:
        return f"{guidance} Produce: {', '.join(phase.expected_outputs)}."
    return guidance


def _acceptance_criteria(route: RouteResult) -> list[AgentRunAcceptanceCriterion]:
    del route
    return []


def _expected_outputs(skill: RouteSelectedSkill) -> list[str]:
    outputs = []
    if skill.reason:
        outputs.append(skill.reason)
    return outputs


def _expected_artifacts_from_criteria(criteria: list[AgentRunAcceptanceCriterion]) -> list[str]:
    return _dedupe([artifact for criterion in criteria for artifact in criterion.expected_artifacts])


def _dedupe(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        output.append(value)
        seen.add(value)
    return output


def _phase_sort_key(phase_id: str) -> tuple[int, str]:
    prefix, _, suffix = phase_id.rpartition("_")
    if prefix == "phase" and suffix.isdigit():
        return (int(suffix), phase_id)
    return (10**9, phase_id)
