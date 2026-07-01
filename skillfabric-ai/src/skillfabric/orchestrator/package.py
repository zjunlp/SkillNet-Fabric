"""Execution package preparation and planner finalization."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skillfabric.orchestrator.agent_run_spec import (
    AgentRunSpec,
    agent_run_spec_from_route,
    agent_run_spec_from_workflow_plan,
)
from skillfabric.orchestrator.renderers.claude_code import render_execution_prompt
from skillfabric.router.models import RouteResult
from skillfabric.storage import Workspace, atomic_write_text
from skillfabric.wiki.pages import slug

PLANNER_PROMPT_ID = "skillfabric_execution_package_planner"

FORBIDDEN_EXECUTION_PROMPT_FRAGMENTS = (
    "Skill tool",
    "SkillFabric runtime",
    "SkillFabric evidence",
    "SkillFabric skill library",
    "selected_skills/",
    "selected skill pages",
    "audit context",
    "audit-context",
    "missing_dependency",
    "execution_report.json",
    "completion_report_schema.json",
)


@dataclass(slots=True)
class ExecutionPackageResult:
    """Finalized execution package build result."""

    root: Path
    spec: AgentRunSpec
    copied_skill_paths: list[str]
    prompt_path: Path
    renderer: str
    workflow_plan_path: Path | None = None
    planner_validation_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "agent_run_spec": self.spec.to_dict(),
            "copied_skill_paths": list(self.copied_skill_paths),
            "prompt_path": str(self.prompt_path),
            "renderer": self.renderer,
            "workflow_plan_path": str(self.workflow_plan_path) if self.workflow_plan_path else "",
            "planner_validation_path": str(self.planner_validation_path) if self.planner_validation_path else "",
        }


@dataclass(slots=True)
class PreparedExecutionPackageResult:
    """Execution package awaiting a planner-authored workflow and prompt."""

    root: Path
    draft_spec: AgentRunSpec
    copied_skill_paths: list[str]
    planner_request_path: Path
    planner_prompt_path: Path
    renderer: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "status": "awaiting_planner",
            "draft_agent_run_spec": self.draft_spec.to_dict(),
            "copied_skill_paths": list(self.copied_skill_paths),
            "planner_request_path": str(self.planner_request_path),
            "planner_prompt_path": str(self.planner_prompt_path),
            "planner_output_path": str(self.root / "planner_output.json"),
            "workflow_plan_path": str(self.root / "workflow_plan.json"),
            "prompt_path": str(self.root / "execution_prompt.md"),
            "renderer": self.renderer,
            "expected_schema": planner_output_json_schema(),
        }


def build_execution_package(
    workspace: Workspace | str | Path,
    route: RouteResult,
    *,
    renderer: str = "claude-code",
) -> ExecutionPackageResult:
    """Generate a finalized package using the deterministic fallback planner.

    Claude Code plugin flows should call ``prepare_execution_package`` and then
    ``finalize_execution_package`` with the workflow-planner subagent output.
    This fallback keeps the Python package usable outside agent-hosted runtimes.
    """

    prepared = prepare_execution_package(workspace, route, renderer=renderer)
    planner_output = deterministic_planner_output(route, prepared.draft_spec, renderer=renderer)
    return finalize_execution_package(prepared.root, planner_output, renderer=renderer)


def prepare_execution_package(
    workspace: Workspace | str | Path,
    route: RouteResult,
    *,
    renderer: str = "claude-code",
) -> PreparedExecutionPackageResult:
    """Prepare selected-skill context and planner request artifacts."""

    workspace = workspace if isinstance(workspace, Workspace) else Workspace(workspace)
    workspace.ensure()
    package_root = route.trace_dir / "execution_package"
    if package_root.exists():
        shutil.rmtree(package_root)
    for path in (
        package_root / "selected_skills",
        package_root / "evidence",
    ):
        path.mkdir(parents=True, exist_ok=True)
    spec = agent_run_spec_from_route(route)
    copied = _copy_selected_skills(workspace, package_root, route)
    atomic_write_text(package_root / "route.json", json.dumps(route.to_dict(), ensure_ascii=False, indent=2) + "\n")
    atomic_write_text(package_root / "agent_run_spec_draft.json", json.dumps(spec.to_dict(), ensure_ascii=False, indent=2) + "\n")
    _write_evidence(package_root, route)
    planner_request_path = package_root / "planner_request.json"
    planner_prompt_path = package_root / "PLANNER.md"
    atomic_write_text(
        planner_request_path,
        json.dumps(_planner_request(route, package_root, renderer), ensure_ascii=False, indent=2) + "\n",
    )
    atomic_write_text(planner_prompt_path, _planner_prompt(route, package_root, renderer))
    return PreparedExecutionPackageResult(
        root=package_root,
        draft_spec=spec,
        copied_skill_paths=copied,
        planner_request_path=planner_request_path,
        planner_prompt_path=planner_prompt_path,
        renderer=renderer,
    )


def finalize_execution_package(
    package_root: str | Path,
    planner_output: dict[str, Any],
    *,
    renderer: str = "claude-code",
) -> ExecutionPackageResult:
    """Validate planner output and write final workflow and prompt artifacts."""

    package_root = Path(package_root)
    route_path = package_root / "route.json"
    if not route_path.exists():
        raise ValueError(f"missing package route artifact: {route_path}")
    route = RouteResult.from_dict(json.loads(route_path.read_text(encoding="utf-8")))
    validation_errors = validate_planner_output(route, package_root, planner_output)
    planner_validation_path = package_root / "planner_validation.json"
    atomic_write_text(
        planner_validation_path,
        json.dumps(
            {
                "valid": not validation_errors,
                "errors": validation_errors,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    if validation_errors:
        raise ValueError(f"invalid planner output: {'; '.join(validation_errors)}")

    workflow_plan = planner_output["workflow_plan"]
    spec = agent_run_spec_from_workflow_plan(route, workflow_plan)
    prompt_path = package_root / "execution_prompt.md"
    workflow_plan_path = package_root / "workflow_plan.json"
    atomic_write_text(package_root / "planner_output.json", json.dumps(planner_output, ensure_ascii=False, indent=2) + "\n")
    atomic_write_text(workflow_plan_path, json.dumps(workflow_plan, ensure_ascii=False, indent=2) + "\n")
    atomic_write_text(package_root / "agent_run_spec.json", json.dumps(spec.to_dict(), ensure_ascii=False, indent=2) + "\n")
    atomic_write_text(prompt_path, str(planner_output["execution_prompt"]).rstrip() + "\n")
    return ExecutionPackageResult(
        root=package_root,
        spec=spec,
        copied_skill_paths=_copied_skill_paths(package_root),
        prompt_path=prompt_path,
        renderer=renderer,
        workflow_plan_path=workflow_plan_path,
        planner_validation_path=planner_validation_path,
    )


def write_planner_failure(package_root: str | Path, errors: list[str]) -> Path:
    """Persist planner failure validation metadata without finalizing a package."""

    package_root = Path(package_root)
    path = package_root / "planner_validation.json"
    atomic_write_text(
        path,
        json.dumps({"valid": False, "errors": list(errors)}, ensure_ascii=False, indent=2) + "\n",
    )
    return path


def deterministic_planner_output(
    route: RouteResult,
    spec: AgentRunSpec | None = None,
    *,
    renderer: str = "claude-code",
) -> dict[str, Any]:
    """Create a valid planner output without a model-backed planner."""

    spec = spec or agent_run_spec_from_route(route)
    workflow_plan = {
        "objective": spec.objective,
        "selected_skill_ids": [skill.skill_id for skill in spec.selected_skills],
        "phases": [phase.to_dict() for phase in spec.phases],
        "execution_strategy": {
            "parallelization_policy": spec.execution_strategy.parallelization_policy,
            "delegation_policy": spec.execution_strategy.delegation_policy,
            "verification_policy": spec.execution_strategy.verification_policy,
        },
        "constraints": list(spec.constraints),
        "coverage_notes": [],
        "rationale": "Deterministic fallback plan generated from the validated route.",
    }
    return {
        "workflow_plan": workflow_plan,
        "execution_prompt": render_execution_prompt(spec, renderer=renderer),
    }


def validate_planner_output(
    route: RouteResult,
    package_root: Path,
    planner_output: dict[str, Any],
) -> list[str]:
    """Return validation errors for a workflow-planner response."""

    errors: list[str] = []
    if not isinstance(planner_output, dict):
        return ["planner output must be a JSON object"]
    workflow_plan = planner_output.get("workflow_plan")
    if not isinstance(workflow_plan, dict):
        errors.append("workflow_plan must be an object")
        workflow_plan = {}
    execution_prompt = planner_output.get("execution_prompt")
    if not isinstance(execution_prompt, str) or not execution_prompt.strip():
        errors.append("execution_prompt must be a non-empty string")
    else:
        _validate_execution_prompt_surface(execution_prompt, errors)
    objective = workflow_plan.get("objective") if isinstance(workflow_plan, dict) else None
    if not isinstance(objective, str) or not objective.strip():
        errors.append("workflow_plan.objective must be a non-empty string")
    execution_strategy = workflow_plan.get("execution_strategy") if isinstance(workflow_plan, dict) else None
    if not isinstance(execution_strategy, dict):
        errors.append("workflow_plan.execution_strategy must be an object")
    else:
        for field_name in ("parallelization_policy", "delegation_policy", "verification_policy"):
            value = execution_strategy.get(field_name)
            if value is not None and not isinstance(value, str):
                errors.append(f"workflow_plan.execution_strategy.{field_name} must be a string")
    _string_list_field(workflow_plan, "constraints", "workflow_plan", errors)
    _string_list_field(workflow_plan, "coverage_notes", "workflow_plan", errors)
    selected_ids = set(route.selected_skill_ids)
    plan_selected = _string_list_field(workflow_plan, "selected_skill_ids", "workflow_plan", errors)
    unknown_selected = sorted(set(plan_selected) - selected_ids)
    if unknown_selected:
        errors.append(f"workflow_plan selected_skill_ids contains unselected skills: {', '.join(unknown_selected)}")
    phase_skill_ids: set[str] = set()
    phase_positions: dict[str, int] = {}
    phases = workflow_plan.get("phases", []) if isinstance(workflow_plan, dict) else []
    if not isinstance(phases, list) or not phases:
        errors.append("workflow_plan.phases must be a non-empty list")
    else:
        phase_ids: set[str] = set()
        for index, phase in enumerate(phases):
            if not isinstance(phase, dict):
                errors.append("workflow_plan.phases items must be objects")
                continue
            phase_id = str(phase.get("id") or "").strip()
            if not phase_id:
                errors.append(f"workflow_plan.phases[{index}].id is required")
                phase_id = f"phase_{index + 1}"
            if phase_id in phase_ids:
                errors.append(f"duplicate workflow_plan phase id: {phase_id}")
            phase_ids.add(phase_id)
            skill_ids = _string_list_field(phase, "skill_ids", f"workflow_plan phase {phase_id}", errors)
            if not skill_ids:
                errors.append(f"workflow_plan phase {phase_id} must reference at least one selected skill")
            for skill_id in skill_ids:
                if skill_id not in selected_ids:
                    errors.append(f"workflow_plan phase {phase_id} references unselected skill: {skill_id}")
                else:
                    phase_skill_ids.add(skill_id)
                    phase_positions.setdefault(skill_id, index)
            references = [
                *_string_list_field(phase, "evidence_refs", f"workflow_plan phase {phase_id}", errors),
                *_string_list_field(phase, "required_inputs", f"workflow_plan phase {phase_id}", errors),
            ]
            _string_list_field(phase, "expected_outputs", f"workflow_plan phase {phase_id}", errors)
            for ref in references:
                _validate_relative_reference(package_root, ref, errors)
        for phase in phases:
            if not isinstance(phase, dict):
                continue
            phase_id = str(phase.get("id") or "").strip()
            for dependency in _string_list_field(phase, "depends_on", f"workflow_plan phase {phase_id}", errors):
                if dependency not in phase_ids:
                    errors.append(f"workflow_plan phase {phase_id} depends on unknown phase: {dependency}")
                if dependency == phase_id:
                    errors.append(f"workflow_plan phase {phase_id} cannot depend on itself")
    missing = sorted(selected_ids - phase_skill_ids)
    if missing:
        errors.append(f"workflow_plan does not assign selected skills to phases: {', '.join(missing)}")
    for edge in route.required_edges:
        before = phase_positions.get(edge.before_skill)
        after = phase_positions.get(edge.after_skill)
        if before is None or after is None:
            continue
        if before > after:
            errors.append(f"workflow_plan violates required order: {edge.before_skill} before {edge.after_skill}")
    return errors


def _validate_execution_prompt_surface(execution_prompt: str, errors: list[str]) -> None:
    """Reject planner prompts that leak SkillFabric runtime mechanics into task execution."""

    for fragment in FORBIDDEN_EXECUTION_PROMPT_FRAGMENTS:
        if fragment.lower() in execution_prompt.lower():
            errors.append(f"execution_prompt contains forbidden runtime-mechanism wording: {fragment}")


def _string_list_field(
    payload: dict[str, Any],
    field_name: str,
    context: str,
    errors: list[str],
) -> list[str]:
    value = payload.get(field_name, [])
    if value is None:
        return []
    if not isinstance(value, list):
        errors.append(f"{context}.{field_name} must be an array")
        return []
    items: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            errors.append(f"{context}.{field_name}[{index}] must be a string")
            continue
        item = item.strip()
        if item:
            items.append(item)
    return items


def planner_output_json_schema() -> dict[str, Any]:
    """Return the workflow-planner response schema."""

    return {
        "type": "object",
        "required": ["workflow_plan", "execution_prompt"],
        "properties": {
            "workflow_plan": {
                "type": "object",
                "required": ["objective", "selected_skill_ids", "phases", "execution_strategy"],
                "properties": {
                    "objective": {"type": "string"},
                    "selected_skill_ids": {"type": "array", "items": {"type": "string"}},
                    "phases": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["id", "goal", "skill_ids"],
                            "properties": {
                                "id": {"type": "string"},
                                "goal": {"type": "string"},
                                "skill_ids": {"type": "array", "items": {"type": "string"}},
                                "depends_on": {"type": "array", "items": {"type": "string"}},
                                "required_inputs": {"type": "array", "items": {"type": "string"}},
                                "expected_outputs": {"type": "array", "items": {"type": "string"}},
                                "evidence_refs": {"type": "array", "items": {"type": "string"}},
                                "guidance": {"type": "string"},
                            },
                        },
                    },
                    "execution_strategy": {
                        "type": "object",
                        "properties": {
                            "parallelization_policy": {"type": "string"},
                            "delegation_policy": {"type": "string"},
                            "verification_policy": {"type": "string"},
                        },
                    },
                    "constraints": {"type": "array", "items": {"type": "string"}},
                    "coverage_notes": {"type": "array", "items": {"type": "string"}},
                    "rationale": {"type": "string"},
                },
            },
            "execution_prompt": {"type": "string"},
        },
        "additionalProperties": False,
    }


def _planner_request(route: RouteResult, package_root: Path, renderer: str) -> dict[str, Any]:
    return {
        "prompt_id": PLANNER_PROMPT_ID,
        "task": route.query,
        "package_root": str(package_root),
        "route_file": str(package_root / "route.json"),
        "selected_skill_context_dir": str(package_root / "selected_skills"),
        "evidence_dir": str(package_root / "evidence"),
        "draft_agent_run_spec": str(package_root / "agent_run_spec_draft.json"),
        "planner_prompt": str(package_root / "PLANNER.md"),
        "expected_output": str(package_root / "planner_output.json"),
        "final_artifacts": {
            "workflow_plan": str(package_root / "workflow_plan.json"),
            "agent_run_spec": str(package_root / "agent_run_spec.json"),
            "execution_prompt": str(package_root / "execution_prompt.md"),
        },
        "expected_schema": planner_output_json_schema(),
    }


def _planner_prompt(route: RouteResult, package_root: Path, renderer: str) -> str:
    selected = "\n".join(
        (
            f"- {skill.skill_id} ({skill.name})"
            f" | rank={skill.rank}"
            f" | role={skill.reason or 'Selected by SkillFabric route.'}"
        )
        for skill in route.selected_skills
    )
    required_edges = "\n".join(
        (
            f"- {edge.before_skill} -> {edge.after_skill}"
            f" | type={edge.edge_type}"
            f" | reason={edge.reason or edge.source}"
        )
        for edge in route.required_edges
    )
    return (
        "# Prompt Contract\n\n"
        f"{PLANNER_PROMPT_ID}\n\n"
        "# Role\n\n"
        "You are SkillFabric's execution-package planner. A router has already selected the allowed capability "
        "roles for one user task. Your job is to turn that routed context into a valid workflow plan and a clean "
        "final execution prompt for the main task agent.\n\n"
        "# Context\n\n"
        "This is a planning-only pass. It improves task decomposition, selected-skill ordering, and final prompt "
        "quality after graph/wiki routing. It must not solve the task, create deliverables, or change the package.\n\n"
        "# Inputs\n\n"
        f"- package_root: `{package_root}`\n"
        f"- planner_request: `{package_root / 'planner_request.json'}`\n"
        f"- route: `{package_root / 'route.json'}`\n"
        f"- selected_skill_context_dir: `{package_root / 'selected_skills'}`\n"
        f"- evidence_dir: `{package_root / 'evidence'}`\n"
        f"- draft_agent_run_spec: `{package_root / 'agent_run_spec_draft.json'}`\n"
        f"- target_renderer: `{renderer}`\n\n"
        "# Selected Skills\n\n"
        f"{selected or '- None'}\n\n"
        "# Required Edges\n\n"
        f"{required_edges or '- None'}\n\n"
        "# Success Criteria\n\n"
        "- The response is one strict JSON object and matches `planner_request.json.expected_schema`.\n"
        "- `workflow_plan` assigns every selected skill to at least one phase.\n"
        "- Phase order preserves every required before -> after edge.\n"
        "- Workflow claims are grounded in route evidence, selected skill pages, and the task objective.\n"
        "- `execution_prompt` is self-contained for the main agent and does not require reading package evidence.\n"
        "- `execution_prompt` describes the task, deliverables, constraints, phase order, and verification expectations clearly.\n\n"
        "# Workflow\n\n"
        "Step 1: Read `planner_request.json` first to confirm the task, package paths, final artifact names, and JSON schema.\n"
        "Step 2: Read `route.json` and `evidence/selected_skill_evidence.json` to understand why each skill was selected.\n"
        "Step 3: Read selected skill pages only as planner context for capability boundaries, prerequisites, outputs, and failure modes.\n"
        "Step 4: Read `evidence/required_edges.json` and preserve every required ordering constraint in phase dependencies.\n"
        "Step 5: Build a minimal workflow that covers the task objective with distinct, non-redundant phases.\n"
        "Step 6: Write a final execution prompt that the main agent can follow directly in the active workspace.\n"
        "Step 7: Check the JSON shape, selected skill coverage, phase dependencies, and final prompt boundary before returning.\n\n"
        "# Output Contract\n\n"
        "Return one strict JSON object with exactly these top-level keys:\n\n"
        "- `workflow_plan`: object with objective, selected_skill_ids, phases, execution_strategy, constraints, coverage_notes, and optional rationale.\n"
        "- `execution_prompt`: string containing the final prompt for the main task agent.\n\n"
        "Use `planner_request.json.expected_schema` as the authoritative schema. Do not return markdown fences, comments, "
        "extra prose, or extra top-level keys.\n\n"
        "# Final Execution Prompt Policy\n\n"
        "The final `execution_prompt` should tell the main agent what to accomplish, what deliverables to create, what "
        "order to follow, and how to verify the result. It may name selected capability roles when that helps task "
        "execution. It must not expose package paths, routing evidence paths, planner artifacts, framework explanations, "
        "or skill-loading mechanics.\n\n"
        "# Constraints\n\n"
        "- Do not execute the task.\n"
        "- Do not edit files, run shell commands, or create deliverables.\n"
        "- Do not introduce skills that are not in `route.json.selected_skills`.\n"
        "- Do not omit a selected skill from the workflow plan.\n"
        "- Do not weaken or reverse required ordering edges.\n"
        "- Do not ask the main agent to inspect the full wiki or package evidence.\n"
        "- Keep deliverables scoped to the active workspace unless the user requests another location.\n"
        "\n"
        "# Self-Check\n\n"
        "Before returning, verify internally that:\n\n"
        "- JSON parses as an object and contains only `workflow_plan` and `execution_prompt`.\n"
        "- Every selected skill id appears in `workflow_plan.selected_skill_ids` and at least one phase.\n"
        "- Every phase references only selected skill ids.\n"
        "- Required edge order is reflected by phase order or explicit `depends_on` fields.\n"
        "- Relative evidence references stay inside the package root.\n"
        "- The final `execution_prompt` is clean task guidance, not a framework explanation.\n"
    )


def _copied_skill_paths(package_root: Path) -> list[str]:
    selected_root = package_root / "selected_skills"
    if not selected_root.exists():
        return []
    return sorted(path.relative_to(package_root).as_posix() for path in selected_root.glob("*.md"))


def _validate_relative_reference(package_root: Path, reference: str, errors: list[str]) -> None:
    if not reference:
        return
    candidate = Path(reference)
    if candidate.is_absolute():
        errors.append(f"workflow_plan reference must be relative to package root: {reference}")
        return
    try:
        resolved = (package_root / candidate).resolve()
        resolved.relative_to(package_root.resolve())
    except (OSError, ValueError):
        errors.append(f"workflow_plan reference escapes package root: {reference}")


def _copy_selected_skills(workspace: Workspace, package_root: Path, route: RouteResult) -> list[str]:
    copied: list[str] = []
    for skill in route.selected_skills:
        source = workspace.wiki_skills_dir / f"{slug(skill.skill_id)}.md"
        target = package_root / "selected_skills" / f"{slug(skill.skill_id)}.md"
        if source.exists():
            shutil.copyfile(source, target)
        else:
            atomic_write_text(
                target,
                (
                    f"# {skill.name}\n\n"
                    f"- Skill id: `{skill.skill_id}`\n"
                    f"- Role: {skill.reason or 'Selected by SkillFabric route.'}\n"
                ),
            )
        copied.append(target.relative_to(package_root).as_posix())
    return copied


def _write_evidence(package_root: Path, route: RouteResult) -> None:
    atomic_write_text(
        package_root / "evidence" / "route_summary.json",
        json.dumps(route.to_dict(), ensure_ascii=False, indent=2) + "\n",
    )
    atomic_write_text(
        package_root / "evidence" / "selected_skill_evidence.json",
        json.dumps(
            [
                {
                    "skill_id": skill.skill_id,
                    "name": skill.name,
                    "role": skill.reason,
                    "evidence": list(skill.evidence),
                }
                for skill in route.selected_skills
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    atomic_write_text(
        package_root / "evidence" / "required_edges.json",
        json.dumps([edge.to_dict() for edge in route.required_edges], ensure_ascii=False, indent=2) + "\n",
    )
