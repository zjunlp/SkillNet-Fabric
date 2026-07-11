"""Final Router orchestration over query-local SkillFabric bundles."""

from __future__ import annotations

import json

from skillfabric.router.bundle import RouterBundleConfig, build_router_bundle
from skillfabric.router.config import RouterConfig, RouterSdkRuntime
from skillfabric.router.models import RouteResult
from skillfabric.router.selection import _fallback_route
from skillfabric.router.traces import _new_trace_id, validate_trace_id
from skillfabric.storage import Workspace, atomic_write_text
from skillfabric.wiki.explorer.agent import WikiExplorerConfig, explore_query_wiki
from skillfabric.wiki.explorer.validation import route_from_skill_package
from skillfabric.wiki.query_wiki import materialize_query_wiki


def route_task(config: RouterConfig, *, sdk_runtime: RouterSdkRuntime | None = None) -> RouteResult:
    """Route one task to a final skill package and write a route trace."""

    workspace = Workspace(config.workspace)
    workspace.ensure()
    trace_id = validate_trace_id(config.trace_id or _new_trace_id(config.query))
    trace_dir = workspace.runs_dir / trace_id
    trace_dir.mkdir(parents=True, exist_ok=True)
    bundle = build_router_bundle(
        RouterBundleConfig(
            workspace=workspace.root,
            query=config.query,
            env_file=config.env_file,
            seed_limit=config.seed_limit,
            expanded_limit=config.expanded_limit,
            workflow_confidence_threshold=config.workflow_confidence_threshold,
            max_workflow_hints=config.max_workflow_hints,
        )
    )
    warnings = list(bundle.warnings)
    result: RouteResult
    if config.explorer_backend not in {"claude-code", "fallback"}:
        raise ValueError(f"unknown explorer_backend: {config.explorer_backend}")
    if config.explorer_backend == "fallback" or not config.use_llm_router:
        result = _fallback_route(
            bundle,
            query=config.query,
            trace_id=trace_id,
            trace_dir=trace_dir,
            max_selected_skills=max(1, config.max_selected_skills),
            warnings=warnings,
        )
    else:
        try:
            query_wiki = materialize_query_wiki(
                workspace,
                bundle,
                trace_dir=trace_dir,
                max_selected_skills=max(1, config.max_selected_skills),
            )
            explorer_run = explore_query_wiki(
                WikiExplorerConfig(
                    env_file=config.env_file,
                    backend=config.explorer_backend,
                    max_selected_skills=max(1, config.max_selected_skills),
                    model=config.explorer_model,
                    strict=config.strict_explorer,
                    max_turns=max(1, config.explorer_max_turns),
                    load_timeout_ms=max(1_000, config.explorer_load_timeout_ms),
                    execution_timeout_seconds=max(1.0, config.explorer_timeout_seconds),
                ),
                query=config.query,
                query_wiki_root=query_wiki.root,
                bundle=bundle,
                trace_dir=trace_dir,
                sdk_runtime=sdk_runtime,
            )
            validation = explorer_run.validation
            warnings.extend(validation.warnings)
            warnings.extend(f"skill package validation error: {error}" for error in validation.errors)
            if config.strict_explorer and validation.errors:
                raise ValueError("; ".join(validation.errors))
            if not validation.valid:
                if config.strict_explorer:
                    raise ValueError("; ".join(validation.errors) or "invalid SkillPackage")
                result = _fallback_route(
                    bundle,
                    query=config.query,
                    trace_id=trace_id,
                    trace_dir=trace_dir,
                    max_selected_skills=max(1, config.max_selected_skills),
                    warnings=warnings,
                )
            else:
                result = route_from_skill_package(
                    validation.valid_package,
                    bundle,
                    query=config.query,
                    trace_id=trace_id,
                    trace_dir=trace_dir,
                    max_selected_skills=max(1, config.max_selected_skills),
                    warnings=warnings,
                )
        except Exception as exc:  # noqa: BLE001 - fallback keeps routing usable by default.
            if config.strict_explorer:
                raise
            warnings.append(f"wiki explorer failed; deterministic fallback used: {type(exc).__name__}: {exc}")
            result = _fallback_route(
                bundle,
                query=config.query,
                trace_id=trace_id,
                trace_dir=trace_dir,
                max_selected_skills=max(1, config.max_selected_skills),
                warnings=warnings,
            )
    atomic_write_text(trace_dir / "route.json", json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n")
    return result


__all__ = ["RouterConfig", "RouterSdkRuntime", "route_task"]
