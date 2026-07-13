"""Explorer-only routing over bounded schema-v2 query wikis."""

from __future__ import annotations

import json

from skillfabric.indexing.embeddings import EmbeddingProvider
from skillfabric.router.bundle import RouterBundleConfig, build_router_bundle
from skillfabric.router.config import RouterConfig, RouterSdkRuntime
from skillfabric.router.models import RouteResult
from skillfabric.router.traces import _create_trace_dir, _new_trace_id
from skillfabric.storage import Workspace, atomic_write_text
from skillfabric.wiki.explorer.agent import WikiExplorerConfig, explore_query_wiki
from skillfabric.wiki.explorer.validation import route_from_skill_package
from skillfabric.wiki.query_wiki import materialize_query_wiki


def route_task(
    config: RouterConfig,
    *,
    sdk_runtime: RouterSdkRuntime | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> RouteResult:
    """Route one task through retrieval, semantic expansion, and strict exploration."""

    if not isinstance(config.query, str) or not config.query.strip():
        raise ValueError("route query must be a non-empty string")
    workspace = Workspace(config.workspace)
    workspace.ensure()
    trace_id = config.trace_id or _new_trace_id(config.query)
    trace_dir = _create_trace_dir(workspace.runs_dir, trace_id)
    atomic_write_text(
        trace_dir / "query.json",
        json.dumps({"query": config.query}, ensure_ascii=False, indent=2) + "\n",
    )
    bundle = build_router_bundle(
        RouterBundleConfig(
            workspace=workspace.root,
            query=config.query,
            env_file=config.env_file,
            seed_limit=config.seed_limit,
            expanded_limit=config.expanded_limit,
            max_depth=config.max_depth,
        ),
        embedding_provider=embedding_provider,
    )
    query_wiki = materialize_query_wiki(workspace, bundle, trace_dir=trace_dir)
    explorer_run = explore_query_wiki(
        WikiExplorerConfig(
            env_file=config.env_file,
            max_selected_skills=config.max_selected_skills,
            model=config.explorer_model,
            max_turns=config.explorer_max_turns,
            load_timeout_ms=config.explorer_load_timeout_ms,
            execution_timeout_seconds=config.explorer_timeout_seconds,
        ),
        query=config.query,
        query_wiki_root=query_wiki.root,
        trace_dir=trace_dir,
        sdk_runtime=sdk_runtime,
    )
    validation = explorer_run.validation
    if not validation.valid:
        raise ValueError("; ".join(validation.errors) or "invalid SkillPackage")
    result = route_from_skill_package(explorer_run.package, bundle)
    atomic_write_text(
        trace_dir / "route.json",
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
    )
    return result


__all__ = ["RouterConfig", "RouterSdkRuntime", "route_task"]
