"""Structured outcome classification for SkillFabric task execution."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from skillfabric.router.models import RouteResult
from skillfabric.task_understanding import RequiredDeliverable

FailureCategory = Literal[
    "missing_dependency",
    "plan_only",
    "wrong_output_path",
    "format_invalid",
    "partial_artifact",
    "router_miss",
    "execution_error",
]

OutcomeStage = Literal["routing", "package", "execution", "evaluation", "reporting"]


@dataclass(slots=True)
class ExecutionOutcome:
    """Structured classification for one external-agent execution."""

    status: str
    primary_category: FailureCategory | None = None
    categories: list[FailureCategory] = field(default_factory=list)
    stage: OutcomeStage | None = None
    evidence: list[str] = field(default_factory=list)
    remediation: list[str] = field(default_factory=list)
    completion_report_path: str | None = None
    completion_report_valid: bool | None = None
    expected_deliverables: list[dict[str, Any]] = field(default_factory=list)
    observed_deliverables: list[str] = field(default_factory=list)
    missing_deliverables: list[dict[str, Any]] = field(default_factory=list)
    wrong_path_candidates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "primary_category": self.primary_category,
            "categories": list(self.categories),
            "stage": self.stage,
            "evidence": list(self.evidence),
            "remediation": list(self.remediation),
            "completion_report_path": self.completion_report_path,
            "completion_report_valid": self.completion_report_valid,
            "expected_deliverables": [dict(item) for item in self.expected_deliverables],
            "observed_deliverables": list(self.observed_deliverables),
            "missing_deliverables": [dict(item) for item in self.missing_deliverables],
            "wrong_path_candidates": list(self.wrong_path_candidates),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExecutionOutcome:
        categories = [
            category
            for category in (str(item) for item in payload.get("categories", []) or [])
            if category in _CATEGORY_PRIORITY
        ]
        primary = str(payload.get("primary_category") or "")
        stage = str(payload.get("stage") or "")
        return cls(
            status=str(payload.get("status", "")),
            primary_category=primary if primary in _CATEGORY_PRIORITY else None,  # type: ignore[arg-type]
            categories=categories,  # type: ignore[arg-type]
            stage=stage if stage in _STAGES else None,  # type: ignore[arg-type]
            evidence=_string_list(payload.get("evidence", [])),
            remediation=_string_list(payload.get("remediation", [])),
            completion_report_path=(
                str(payload["completion_report_path"]) if payload.get("completion_report_path") is not None else None
            ),
            completion_report_valid=(
                bool(payload["completion_report_valid"])
                if payload.get("completion_report_valid") is not None
                else None
            ),
            expected_deliverables=[
                dict(item)
                for item in payload.get("expected_deliverables", []) or []
                if isinstance(item, dict)
            ],
            observed_deliverables=_string_list(payload.get("observed_deliverables", [])),
            missing_deliverables=[
                dict(item)
                for item in payload.get("missing_deliverables", []) or []
                if isinstance(item, dict)
            ],
            wrong_path_candidates=_string_list(payload.get("wrong_path_candidates", [])),
        )


@dataclass(slots=True)
class ExecutionOutcomeInputs:
    """Inputs needed to classify one SkillFabric execution."""

    route: RouteResult | None
    workspace: Path
    sdk_status: str
    sdk_error: str | None = None
    sdk_response: str = ""
    sdk_events: list[dict[str, Any]] = field(default_factory=list)
    evaluation: dict[str, Any] = field(default_factory=dict)
    evaluation_error: str | None = None
    completion_report_name: str = "execution_report.json"


_CATEGORY_PRIORITY: dict[str, int] = {
    "router_miss": 0,
    "missing_dependency": 1,
    "plan_only": 2,
    "wrong_output_path": 3,
    "format_invalid": 4,
    "partial_artifact": 5,
    "execution_error": 6,
}
_STAGES = {"routing", "package", "execution", "evaluation", "reporting"}
_DEPENDENCY_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"No module named ['\"]?([A-Za-z0-9_.-]+)",
        r"ModuleNotFoundError",
        r"ImportError",
        r"command not found",
        r"not found: [A-Za-z0-9_.-]+",
        r"No such file or directory: ['\"]?(ffmpeg|libreoffice|soffice|python|python3|pip)",
        r"cannot import name",
    )
]
_PLAN_ONLY_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bplan\b.*\bimplement",
        r"\bwould\b.*\bcreate",
        r"\bnext steps?\b",
        r"let me know",
        r"是否继续",
        r"要不要继续",
    )
]
_FORMAT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"invalid",
        r"malformed",
        r"failed to parse",
        r"not valid",
        r"schema",
        r"format",
        r"corrupt",
        r"cannot open",
    )
]


def classify_execution_outcome(inputs: ExecutionOutcomeInputs) -> ExecutionOutcome:
    """Classify one SkillFabric execution into stable failure categories."""

    expected = _expected_deliverables(inputs.route, inputs.evaluation)
    observed = _observed_deliverables(inputs.workspace)
    missing = _missing_expected_deliverables(inputs.workspace, expected)
    wrong_path_candidates = _wrong_path_candidates(inputs.workspace, expected, missing)
    report_path = inputs.workspace / inputs.completion_report_name
    report_valid, report_evidence = _completion_report_status(report_path)
    evidence: list[str] = []
    remediation: list[str] = []
    categories: list[FailureCategory] = []
    stage: OutcomeStage | None = None

    artifacts_passed = _objective_artifacts_passed(
        inputs,
        missing=missing,
        wrong_path_candidates=wrong_path_candidates,
    )

    if inputs.route is None or not inputs.route.selected_skills:
        _add_category(categories, "router_miss")
        evidence.append("No selected skills were available for the execution package.")
        remediation.append("Inspect router coverage diagnostics and selected skill evidence for this task.")
        stage = "routing"
    elif _route_has_missing_coverage(inputs.route):
        if artifacts_passed:
            evidence.append(
                "Route coverage diagnostics contain unresolved task requirements, "
                "but objective evaluation passed and expected deliverables are present."
            )
            remediation.append(
                "Treat route coverage as a routing-quality warning for this run; "
                "improve coverage diagnostics separately from execution success."
            )
        else:
            _add_category(categories, "router_miss")
            evidence.append("Route coverage diagnostics contain uncovered task requirements.")
            remediation.append("Improve routing coverage or add sharper skill interface evidence for the missing task facet.")
            stage = stage or "routing"

    combined_text = _combined_text(inputs)
    if _matches(_DEPENDENCY_PATTERNS, combined_text):
        _add_category(categories, "missing_dependency")
        evidence.append(_short_evidence("Execution text indicates an unavailable package or command", combined_text))
        remediation.append("Install the missing dependency in the task runtime environment or choose a skill that avoids it.")
        stage = stage or "execution"

    if _looks_plan_only(inputs, report_path=report_path, observed=observed):
        _add_category(categories, "plan_only")
        evidence.append("SDK completed without material task artifacts, and the response resembles planning or next-step text.")
        remediation.append("Strengthen execution prompt constraints and SDK tool permissions for direct artifact generation.")
        stage = stage or "execution"

    if wrong_path_candidates:
        _add_category(categories, "wrong_output_path")
        evidence.append(
            "Expected output filename was found outside the active workspace: "
            + ", ".join(wrong_path_candidates[:5])
        )
        remediation.append("Keep deliverables under the active workspace or normalize evaluator paths before scoring.")
        stage = stage or "execution"

    evaluation_text = _evaluation_text(inputs.evaluation, inputs.evaluation_error)
    if inputs.evaluation_error or _matches(_FORMAT_PATTERNS, evaluation_text):
        _add_category(categories, "format_invalid")
        evidence.append(_short_evidence("Evaluator reported a format or parsing issue", evaluation_text))
        remediation.append("Add artifact-specific validation before the completion report is written.")
        stage = stage or "evaluation"

    if missing and observed and not wrong_path_candidates:
        _add_category(categories, "partial_artifact")
        evidence.append("Some expected deliverables are missing while other workspace artifacts exist.")
        remediation.append("Compare expected deliverables with the completion report and rerun generation for the missing artifacts.")
        stage = stage or "execution"

    if not report_valid:
        evidence.extend(report_evidence)
        remediation.append("Write a valid completion report using completion_report_schema.json after verification.")
        stage = stage or "reporting"
        if "partial_artifact" not in categories:
            _add_category(categories, "partial_artifact")

    if inputs.sdk_status != "completed" and not categories:
        _add_category(categories, "execution_error")
        evidence.append(inputs.sdk_error or "Claude Code SDK did not return a completed status.")
        remediation.append("Inspect the SDK log and tool events for the first execution error.")
        stage = stage or "execution"

    if _evaluation_failed(inputs.evaluation) and not categories:
        _add_category(categories, "execution_error")
        evidence.append(_short_evidence("Objective evaluator did not pass", evaluation_text))
        remediation.append("Inspect evaluator results and generated artifacts for task-specific correctness failures.")
        stage = stage or "evaluation"

    status = "ok" if not categories else "failed"
    return ExecutionOutcome(
        status=status,
        primary_category=_primary_category(categories),
        categories=categories,
        stage=stage,
        evidence=_dedupe([item for item in evidence if item]),
        remediation=_dedupe([item for item in remediation if item]),
        completion_report_path=str(report_path),
        completion_report_valid=report_valid,
        expected_deliverables=expected,
        observed_deliverables=observed,
        missing_deliverables=missing,
        wrong_path_candidates=wrong_path_candidates,
    )


def _route_has_missing_coverage(route: RouteResult) -> bool:
    diagnostics = route.coverage_diagnostics
    if isinstance(diagnostics, dict) and diagnostics.get("missing"):
        return True
    for item in route.task_understanding.coverage_diagnostics:
        if not isinstance(item, dict):
            if item:
                return True
            continue
        status = str(item.get("status", "")).strip().lower()
        if status in {"missing", "ambiguous", "unresolved"}:
            return True
    return False


def _objective_artifacts_passed(
    inputs: ExecutionOutcomeInputs,
    *,
    missing: list[dict[str, Any]],
    wrong_path_candidates: list[str],
) -> bool:
    if inputs.sdk_status != "completed":
        return False
    if inputs.evaluation_error:
        return False
    if not inputs.evaluation or inputs.evaluation.get("deferred"):
        return False
    return bool(inputs.evaluation.get("passed", False)) and not missing and not wrong_path_candidates


def _expected_deliverables(route: RouteResult | None, evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    if route is not None:
        for deliverable in route.task_understanding.required_deliverables:
            record = _deliverable_record(deliverable)
            key = (str(record.get("path", "")), str(record.get("format", "")))
            if key not in seen:
                output.append(record)
                seen.add(key)
    for path in _paths_from_evaluation(evaluation):
        fmt = _format_from_path(path)
        key = (path, fmt)
        if key in seen:
            continue
        output.append({"path": path, "format": fmt, "minimum_count": 1, "source": "evaluation"})
        seen.add(key)
    return output


def _deliverable_record(deliverable: RequiredDeliverable) -> dict[str, Any]:
    return {
        "id": deliverable.id,
        "path": deliverable.path,
        "format": deliverable.format,
        "label": deliverable.label,
        "minimum_count": deliverable.minimum_count,
        "source_text": deliverable.source_text,
        "source": "task_understanding",
    }


def _paths_from_evaluation(evaluation: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for item in evaluation.get("evaluator_results", []) or []:
        if not isinstance(item, dict):
            continue
        for key in ("path", "file", "expected_path", "actual_path"):
            value = item.get(key)
            if isinstance(value, str) and _looks_like_artifact_path(value):
                paths.append(value)
        op_args = item.get("op_args")
        if isinstance(op_args, dict):
            value = op_args.get("path")
            if isinstance(value, str) and _looks_like_artifact_path(value):
                paths.append(value)
    return _dedupe(paths)


def _observed_deliverables(workspace: Path) -> list[str]:
    if not workspace.exists():
        return []
    output: list[str] = []
    for path in sorted(workspace.rglob("*")):
        if not path.is_file() or _is_context_file(path, workspace):
            continue
        rel = path.relative_to(workspace).as_posix()
        output.append(rel)
    return output


def _missing_expected_deliverables(workspace: Path, expected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for item in expected:
        path = str(item.get("path", "")).strip()
        fmt = str(item.get("format", "")).strip().lower()
        minimum_count = max(1, int(item.get("minimum_count", 1) or 1))
        if path:
            if not (workspace / path).exists():
                missing.append(dict(item))
            continue
        if fmt and len(_files_with_format(workspace, fmt)) < minimum_count:
            missing.append(dict(item))
    return missing


def _wrong_path_candidates(
    workspace: Path,
    expected: list[dict[str, Any]],
    missing: list[dict[str, Any]],
) -> list[str]:
    if not missing:
        return []
    task_root = workspace.parent
    if not task_root.exists():
        return []
    expected_names = {Path(str(item.get("path", ""))).name for item in expected if str(item.get("path", "")).strip()}
    if not expected_names:
        return []
    output: list[str] = []
    for path in sorted(task_root.rglob("*")):
        if not path.is_file() or path.is_relative_to(workspace):
            continue
        if path.name in expected_names:
            output.append(path.relative_to(task_root).as_posix())
    return output


def _completion_report_status(path: Path) -> tuple[bool, list[str]]:
    if not path.exists():
        return False, [f"Completion report is missing: {path.name}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, [f"Completion report is invalid JSON: {exc}"]
    if not isinstance(payload, dict):
        return False, ["Completion report JSON root is not an object."]
    required = {"completed_phases", "skills_used", "deliverables", "deviations", "blocking_issues"}
    missing = sorted(required - set(payload))
    if missing:
        return False, [f"Completion report is missing required keys: {', '.join(missing)}"]
    return True, []


def _looks_plan_only(inputs: ExecutionOutcomeInputs, *, report_path: Path, observed: list[str]) -> bool:
    if inputs.sdk_status != "completed":
        return False
    material_artifacts = [item for item in observed if item != report_path.name]
    if material_artifacts:
        return False
    text = inputs.sdk_response or inputs.sdk_error or ""
    if not text:
        return False
    return _matches(_PLAN_ONLY_PATTERNS, text)


def _combined_text(inputs: ExecutionOutcomeInputs) -> str:
    parts = [inputs.sdk_error or "", inputs.sdk_response or "", _evaluation_text(inputs.evaluation, inputs.evaluation_error)]
    for event in inputs.sdk_events:
        parts.append(json.dumps(event, ensure_ascii=False, sort_keys=True))
    return "\n".join(part for part in parts if part)


def _evaluation_text(evaluation: dict[str, Any], evaluation_error: str | None) -> str:
    parts = [evaluation_error or "", str(evaluation.get("summary", ""))]
    for item in evaluation.get("evaluator_results", []) or []:
        parts.append(json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, dict) else str(item))
    return "\n".join(part for part in parts if part)


def _evaluation_failed(evaluation: dict[str, Any]) -> bool:
    if not evaluation:
        return False
    if evaluation.get("deferred"):
        return False
    return not bool(evaluation.get("passed", False))


def _matches(patterns: list[re.Pattern[str]], text: str) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _short_evidence(prefix: str, text: str, *, limit: int = 240) -> str:
    normalized = " ".join(text.split())
    if len(normalized) > limit:
        normalized = normalized[: limit - 3] + "..."
    return f"{prefix}: {normalized}" if normalized else prefix


def _add_category(categories: list[FailureCategory], category: FailureCategory) -> None:
    if category not in categories:
        categories.append(category)


def _primary_category(categories: list[FailureCategory]) -> FailureCategory | None:
    if not categories:
        return None
    return sorted(categories, key=lambda item: _CATEGORY_PRIORITY[item])[0]


def _dedupe(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        output.append(value)
        seen.add(value)
    return output


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _files_with_format(workspace: Path, fmt: str) -> list[Path]:
    if not workspace.exists():
        return []
    suffix = "." + fmt.lower().lstrip(".")
    return [
        path
        for path in workspace.rglob("*")
        if path.is_file() and path.suffix.lower() == suffix and not _is_context_file(path, workspace)
    ]


def _is_context_file(path: Path, workspace: Path) -> bool:
    try:
        rel = path.relative_to(workspace).as_posix()
    except ValueError:
        return False
    return (
        rel == "completion_report_schema.json"
        or rel.startswith("selected_skills/")
        or rel.startswith(".")
    )


def _format_from_path(path: str) -> str:
    suffix = Path(path).suffix.lower().lstrip(".")
    return suffix


def _looks_like_artifact_path(value: str) -> bool:
    return bool(_format_from_path(value))
