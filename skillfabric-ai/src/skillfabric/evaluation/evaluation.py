"""Offline quality-loop evaluation for routing, wiki exploration, and execution packages."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skillfabric.compiled_graph.health import analyze_health
from skillfabric.compiled_graph.models import CommunityNode, GraphDocument
from skillfabric.orchestrator.package import ExecutionPackageResult, build_execution_package
from skillfabric.router.routing import RouterConfig, route_task
from skillfabric.storage import Workspace, atomic_write_text
from skillfabric.wiki.health import read_wiki_health_summary


@dataclass(slots=True)
class EvalConfig:
    """Configuration for local case-file evaluation."""

    workspace: str | Path = ".skillfabric"
    cases_path: str | Path = ""
    env_file: str | Path = ".env"
    use_llm_router: bool = True
    max_selected_skills: int = 8
    seed_limit: int = 8
    expanded_limit: int = 50
    workflow_confidence_threshold: float = 0.95
    max_workflow_hints: int = 12
    explorer_backend: str = "claude-code"
    explorer_model: str | None = None
    strict_explorer: bool = False
    execution_renderer: str = "claude-code"


def run_eval(config: EvalConfig) -> dict[str, Any]:
    """Run evaluation cases and write runs/eval_report.json."""

    workspace = Workspace(config.workspace)
    workspace.ensure()
    if not str(config.cases_path):
        raise ValueError("EvalConfig.cases_path is required")
    cases = _load_cases(Path(config.cases_path))
    rows = [_run_case(config, case) for case in cases]
    summary = _summary(rows)
    build_status = _build_status_summary(workspace)
    graph_health = _graph_health_summary(workspace)
    execution_health = _execution_health_summary(workspace)
    wiki_health = _wiki_health_summary(workspace)
    summary.update(_build_status_summary_fields(build_status))
    summary.update(_graph_health_summary_fields(graph_health))
    summary.update(_execution_health_summary_fields(execution_health))
    summary.update(_wiki_health_summary_fields(wiki_health))
    report = {
        "case_count": len(rows),
        "summary": summary,
        "build_status": build_status,
        "graph_health": graph_health,
        "execution_health": execution_health,
        "wiki_health": wiki_health,
        "cases": rows,
        "workspace": str(workspace.root),
    }
    atomic_write_text(
        workspace.runs_dir / "eval_report.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    return report


def _run_case(config: EvalConfig, case: dict[str, Any]) -> dict[str, Any]:
    case_id = str(case.get("id", "case")).strip() or "case"
    query = str(case.get("query", ""))
    if not query:
        raise ValueError(f"eval case {case_id} missing query")
    trace_id = f"eval-{_trace_slug(case_id)}"
    route = route_task(
        RouterConfig(
            workspace=config.workspace,
            query=query,
            env_file=config.env_file,
            use_llm_router=config.use_llm_router,
            max_selected_skills=config.max_selected_skills,
            seed_limit=config.seed_limit,
            expanded_limit=config.expanded_limit,
            workflow_confidence_threshold=config.workflow_confidence_threshold,
            max_workflow_hints=config.max_workflow_hints,
            trace_id=trace_id,
            explorer_backend=config.explorer_backend,
            explorer_model=config.explorer_model,
            strict_explorer=config.strict_explorer,
        )
    )
    package = build_execution_package(
        config.workspace,
        route,
        renderer=config.execution_renderer,
    )
    package_errors = _validate_execution_package_result(package, route)
    required_skills = _string_list(case.get("required_skills", case.get("required_skill_ids", [])))
    selected = set(route.selected_skill_ids)
    hit_required = [skill_id for skill_id in required_skills if skill_id in selected]
    missing_required = [skill_id for skill_id in required_skills if skill_id not in selected]
    route_coverage = route.coverage_diagnostics or {}
    coverage_requirements = route_coverage.get("requirements", [])
    coverage_covered = route_coverage.get("covered", [])
    coverage_missing = route_coverage.get("missing", [])
    deliverable_missing = [
        item
        for item in coverage_missing
        if isinstance(item, dict) and str(item.get("kind", "")) == "deliverable"
    ]
    return {
        "id": case_id,
        "query": query,
        "trace_id": route.trace_id,
        "selected_skill_ids": route.selected_skill_ids,
        "required_skills": required_skills,
        "hit_required_skills": hit_required,
        "missing_required_skills": missing_required,
        "required_skill_recall": _ratio(len(hit_required), len(required_skills)),
        "required_skill_coverage": not missing_required,
        "deliverable_coverage": not deliverable_missing,
        "coverage_diagnostics": route_coverage,
        "coverage_requirement_count": len(coverage_requirements),
        "coverage_covered_count": len(coverage_covered),
        "coverage_missing_count": len(coverage_missing),
        "route_warning_count": len(route.warnings),
        "skill_package_validation_error_count": sum(
            1 for warning in route.warnings if warning.startswith("skill package validation error:")
        ),
        "execution_package_root": str(package.root),
        "execution_package_valid": not package_errors,
        "execution_package_errors": package_errors,
        "agent_run_phase_count": len(package.spec.phases),
        "selected_skill_context_count": len(package.copied_skill_paths),
        "required_order_count": len(package.spec.required_order),
        "execution_renderer": config.execution_renderer,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "required_skill_recall": _average([float(row["required_skill_recall"]) for row in rows]),
        "required_skill_coverage_rate": _average([1.0 if row["required_skill_coverage"] else 0.0 for row in rows]),
        "deliverable_coverage_rate": _average([1.0 if row["deliverable_coverage"] else 0.0 for row in rows]),
        "average_coverage_missing_count": _average(
            [float(row["coverage_missing_count"]) for row in rows]
        ),
        "average_route_warning_count": _average([float(row["route_warning_count"]) for row in rows]),
        "average_skill_package_validation_error_count": _average(
            [float(row["skill_package_validation_error_count"]) for row in rows]
        ),
        "execution_package_valid_rate": _average(
            [1.0 if row["execution_package_valid"] else 0.0 for row in rows]
        ),
        "average_agent_run_phase_count": _average(
            [float(row["agent_run_phase_count"]) for row in rows]
        ),
        "average_selected_skill_context_count": _average(
            [float(row["selected_skill_context_count"]) for row in rows]
        ),
    }


def _validate_execution_package_result(
    package: ExecutionPackageResult,
    route: Any,
) -> list[str]:
    """Check execution-package artifacts against the route."""

    errors: list[str] = []
    required_files = [
        package.root / "agent_run_spec.json",
        package.root / "execution_prompt.md",
        package.root / "evidence" / "route_summary.json",
        package.root / "evidence" / "selected_skill_evidence.json",
        package.root / "evidence" / "required_edges.json",
    ]
    for path in required_files:
        if not path.exists():
            errors.append(f"missing execution package artifact: {path.name}")
    route_skill_ids = list(route.selected_skill_ids)
    spec_skill_ids = [item.skill_id for item in package.spec.selected_skills]
    if spec_skill_ids != route_skill_ids:
        errors.append("AgentRunSpec selected skill ids do not match RouteResult selected skill ids")
    copied = set(package.copied_skill_paths)
    for selected in package.spec.selected_skills:
        if selected.skill_context_path not in copied:
            errors.append(f"selected skill context was not copied: {selected.skill_context_path}")
        if not (package.root / selected.skill_context_path).exists():
            errors.append(f"selected skill context path missing: {selected.skill_context_path}")
    selected_ids = set(route_skill_ids)
    phase_ids = {phase.id for phase in package.spec.phases}
    for phase in package.spec.phases:
        for skill_id in phase.skill_ids:
            if skill_id not in selected_ids:
                errors.append(f"phase references unselected skill: {phase.id} -> {skill_id}")
        for dependency in phase.depends_on:
            if dependency not in phase_ids:
                errors.append(f"phase references unknown dependency: {phase.id} -> {dependency}")
    for order in package.spec.required_order:
        if order.before_skill not in selected_ids:
            errors.append(f"required order references unselected before_skill: {order.before_skill}")
        if order.after_skill not in selected_ids:
            errors.append(f"required order references unselected after_skill: {order.after_skill}")
    return errors


def _build_status_summary(workspace: Workspace) -> dict[str, Any]:
    status = workspace.read_json(workspace.status_path, default={})
    if not isinstance(status, dict) or not status:
        return {"available": False, "reason": "status.json missing"}
    warning = str(status.get("community_assignment_warning", "") or "")
    return {
        "available": True,
        "build_id": str(status.get("build_id", "") or ""),
        "skill_count": int(status.get("skill_count", 0) or 0),
        "community_count": int(status.get("community_count", 0) or 0),
        "edge_count": int(status.get("edge_count", 0) or 0),
        "interface_count": int(status.get("interface_count", 0) or 0),
        "interface_accepted_count": int(status.get("interface_accepted_count", 0) or 0),
        "interface_rejected_count": int(status.get("interface_rejected_count", 0) or 0),
        "community_assignment_provenance": str(status.get("community_assignment_provenance", "") or ""),
        "community_assignment_warning": warning,
        "community_assignment_warning_present": bool(warning),
        "canonicalization_model_id": str(status.get("canonicalization_model_id", "") or ""),
        "interface_model_id": str(status.get("interface_model_id", "") or ""),
        "community_assignment_model_id": str(status.get("community_assignment_model_id", "") or ""),
    }


def _build_status_summary_fields(build_status: dict[str, Any]) -> dict[str, Any]:
    if not build_status.get("available"):
        return {"build_status_available": False}
    return {
        "build_status_available": True,
        "build_skill_count": int(build_status.get("skill_count", 0) or 0),
        "build_community_count": int(build_status.get("community_count", 0) or 0),
        "build_edge_count": int(build_status.get("edge_count", 0) or 0),
        "build_interface_rejected_count": int(build_status.get("interface_rejected_count", 0) or 0),
        "build_community_assignment_warning_present": bool(
            build_status.get("community_assignment_warning_present", False)
        ),
        "build_community_assignment_provenance": str(
            build_status.get("community_assignment_provenance", "") or ""
        ),
    }


def _graph_health_summary(workspace: Workspace) -> dict[str, Any]:
    graph_path = workspace.graph_dir / "graph.json"
    if not graph_path.exists():
        return {"available": False, "reason": "graph.json missing"}
    graph = GraphDocument.from_dict(json.loads(graph_path.read_text(encoding="utf-8")))
    communities = _load_communities(workspace, graph)
    report = analyze_health(graph, communities)
    return {
        "available": True,
        "community_count": len(communities),
        "community_text_outlier_count": len(report.community_text_outliers),
        "weak_cross_community_compose_edge_count": len(report.weak_cross_community_compose_edges),
        "low_cohesion_large_community_count": len(report.low_cohesion_large_communities),
        "depend_on_cycle_count": len(report.depend_on_cycles),
        "edges_missing_evidence": report.edges_missing_evidence,
        "community_text_outliers": [
            {
                "skill_id": item.skill_id,
                "assigned_community": item.assigned_community_name,
                "suggested_community": item.suggested_community_name,
                "current_score": round(item.current_score, 4),
                "suggested_score": round(item.suggested_score, 4),
                "shared_terms": item.shared_terms,
            }
            for item in report.community_text_outliers[:10]
        ],
        "low_cohesion_large_communities": [
            {
                "community_id": item.community_id,
                "community_name": item.community_name,
                "member_count": item.member_count,
                "average_member_count": round(item.average_member_count, 4),
                "cohesion_score": round(item.cohesion_score, 4),
            }
            for item in report.low_cohesion_large_communities[:10]
        ],
    }


def _load_communities(workspace: Workspace, graph: GraphDocument) -> list[CommunityNode]:
    communities_path = workspace.graph_dir / "communities.json"
    if communities_path.exists():
        payload = json.loads(communities_path.read_text(encoding="utf-8"))
        return [
            CommunityNode.from_dict(item["community"])
            for item in payload.get("communities", [])
            if isinstance(item, dict) and isinstance(item.get("community"), dict)
        ]
    return [node for node in graph.nodes if isinstance(node, CommunityNode)]


def _graph_health_summary_fields(graph_health: dict[str, Any]) -> dict[str, Any]:
    if not graph_health.get("available"):
        return {"graph_health_available": False}
    community_text_outliers = int(graph_health.get("community_text_outlier_count", 0) or 0)
    low_cohesion_large = int(graph_health.get("low_cohesion_large_community_count", 0) or 0)
    depend_on_cycles = int(graph_health.get("depend_on_cycle_count", 0) or 0)
    edges_missing_evidence = int(graph_health.get("edges_missing_evidence", 0) or 0)
    return {
        "graph_health_available": True,
        "graph_community_text_outlier_count": community_text_outliers,
        "graph_weak_cross_community_compose_edge_count": int(
            graph_health.get("weak_cross_community_compose_edge_count", 0) or 0
        ),
        "graph_low_cohesion_large_community_count": low_cohesion_large,
        "graph_depend_on_cycle_count": depend_on_cycles,
        "graph_edges_missing_evidence": edges_missing_evidence,
        "graph_health_pass": (
            community_text_outliers == 0
            and low_cohesion_large == 0
            and depend_on_cycles == 0
            and edges_missing_evidence == 0
        ),
    }


def _execution_health_summary(workspace: Workspace) -> dict[str, Any]:
    status = workspace.read_json(workspace.status_path, default={})
    if not isinstance(status, dict) or not status:
        return {"available": False, "reason": "status.json missing"}
    candidate_count = int(status.get("execution_candidate_count", 0) or 0)
    accepted_count = int(status.get("execution_accepted_flow_count", 0) or 0)
    return {
        "available": True,
        "raw_artifact_count": int(status.get("raw_artifact_count", 0) or 0),
        "raw_scenario_count": int(status.get("raw_scenario_count", 0) or 0),
        "canonical_artifact_count": int(status.get("canonical_artifact_count", 0) or 0),
        "execution_candidate_count": candidate_count,
        "execution_accepted_flow_count": accepted_count,
        "execution_rejected_flow_count": int(status.get("execution_rejected_flow_count", 0) or 0),
        "execution_compatibility_count": int(status.get("execution_compatibility_count", 0) or 0),
        "execution_projected_edge_count": int(status.get("execution_projected_edge_count", 0) or 0),
        "execution_acceptance_rate": _ratio(accepted_count, candidate_count),
    }


def _execution_health_summary_fields(execution_health: dict[str, Any]) -> dict[str, Any]:
    if not execution_health.get("available"):
        return {"execution_health_available": False}
    return {
        "execution_health_available": True,
        "execution_candidate_count": int(execution_health.get("execution_candidate_count", 0) or 0),
        "execution_accepted_flow_count": int(execution_health.get("execution_accepted_flow_count", 0) or 0),
        "execution_rejected_flow_count": int(execution_health.get("execution_rejected_flow_count", 0) or 0),
        "execution_compatibility_count": int(execution_health.get("execution_compatibility_count", 0) or 0),
        "execution_projected_edge_count": int(execution_health.get("execution_projected_edge_count", 0) or 0),
        "execution_acceptance_rate": float(execution_health.get("execution_acceptance_rate", 1.0) or 0.0),
    }


def _wiki_health_summary(workspace: Workspace) -> dict[str, Any]:
    health_path = workspace.wiki_dir / "wiki_health_report.md"
    if not health_path.exists():
        return {"available": False, "reason": "wiki_health_report.md missing"}
    summary = read_wiki_health_summary(health_path)
    return {"available": True, **summary}


def _wiki_health_summary_fields(wiki_health: dict[str, Any]) -> dict[str, Any]:
    if not wiki_health.get("available"):
        return {"wiki_health_available": False}
    warning_count = sum(
        int(value)
        for key, value in wiki_health.items()
        if key not in {"available", "summary_fallback_count"} and isinstance(value, int)
    )
    return {
        "wiki_health_available": True,
        "wiki_warning_count": warning_count,
        "wiki_summary_fallback_count": int(wiki_health.get("summary_fallback_count", 0) or 0),
        "wiki_health_pass": warning_count == 0 and int(wiki_health.get("summary_fallback_count", 0) or 0) == 0,
    }


def _load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_cases = payload.get("cases", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_cases, list):
        raise ValueError("eval cases must be a list or an object with a cases list")
    cases = [item for item in raw_cases if isinstance(item, dict)]
    if not cases:
        raise ValueError("eval cases file contains no cases")
    return cases


def _trace_slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "-" for char in value.lower()).strip("-") or "case"


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return numerator / denominator


def _average(values: list[float]) -> float:
    if not values:
        return 1.0
    return round(sum(values) / len(values), 6)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []
