"""Build query-local skill bundles for router and execution-package context."""

from __future__ import annotations

from skillfabric.router.assembly import (
    _communities,
    _load_graph,
    _load_registry_skills,
    _selected_communities,
    _wiki_pages,
    _workflow_hints,
)
from skillfabric.router.coverage import (
    load_execution_index,
    load_interfaces,
    resolve_coverage_requirements,
)
from skillfabric.router.expansion import _expand_seed_skills, _expand_seed_skills_ppr
from skillfabric.router.facet_preservation import preserve_facet_candidates
from skillfabric.router.models import (
    RouterBundle,
    RouterBundleConfig,
    RouterCommunityContext,
    RouterSkillCandidate,
    RouterWorkflowHint,
)
from skillfabric.router.retrieval import _apply_hard_include_scores, _seed_scores
from skillfabric.storage import Workspace
from skillfabric.task_understanding import (
    analyze_task,
    coverage_diagnostics,
    filter_task_understanding_skills,
    hard_include_skill_ids,
)


def build_router_bundle(config: RouterBundleConfig) -> RouterBundle:
    """Build a compact, query-local context bundle from compiled SkillFabric artifacts."""

    workspace = Workspace(config.workspace)
    graph = _load_graph(workspace)
    skills = _load_registry_skills(workspace)
    communities = _communities(graph)
    warnings: list[str] = []
    if not skills:
        warnings.append(f"registry skills not found: {workspace.registry_dir / 'skills.jsonl'}")
    interfaces = load_interfaces(workspace)
    execution_index = load_execution_index(workspace)
    task_understanding = resolve_coverage_requirements(
        analyze_task(config.query),
        skills=skills,
        interfaces=interfaces,
        execution_index=execution_index,
        graph=graph,
    )
    task_understanding = filter_task_understanding_skills(task_understanding, set(skills))
    seed_scores = _seed_scores(workspace, config.query, skills, warnings=warnings, env_file=config.env_file)
    hard_includes = hard_include_skill_ids(task_understanding, set(skills))
    _apply_hard_include_scores(seed_scores, skills, hard_includes)
    if config.graph_expansion_mode == "one_hop":
        selected = _expand_seed_skills(
            graph.edges,
            skills,
            seed_scores,
            hard_include_ids=set(hard_includes),
            seed_limit=max(config.seed_limit, 0),
            expanded_limit=max(config.expanded_limit, 0),
        )
    elif config.graph_expansion_mode == "ppr":
        selected = _expand_seed_skills_ppr(
            graph.edges,
            skills,
            seed_scores,
            hard_include_ids=set(hard_includes),
            seed_limit=max(config.seed_limit, 0),
            expanded_limit=max(config.expanded_limit, 0),
            alpha=config.ppr_alpha,
            max_iter=max(config.ppr_max_iter, 1),
            tol=max(config.ppr_tol, 0.0),
        )
    else:
        warnings.append(f"unknown graph expansion mode {config.graph_expansion_mode!r}; using ppr")
        selected = _expand_seed_skills_ppr(
            graph.edges,
            skills,
            seed_scores,
            hard_include_ids=set(hard_includes),
            seed_limit=max(config.seed_limit, 0),
            expanded_limit=max(config.expanded_limit, 0),
            alpha=config.ppr_alpha,
            max_iter=max(config.ppr_max_iter, 1),
            tol=max(config.ppr_tol, 0.0),
        )
    selected = preserve_facet_candidates(
        selected,
        seed_scores=seed_scores,
        skills=skills,
        interfaces=interfaces,
        understanding=task_understanding,
        hard_include_ids=set(hard_includes),
        expanded_limit=max(config.expanded_limit, 0),
        wiki_skills_dir=workspace.wiki_skills_dir,
        warnings=warnings,
    )
    selected_ids = {item.skill_id for item in selected}
    community_context = _selected_communities(graph.edges, communities, selected_ids)
    workflow_hints = _workflow_hints(
        workspace,
        selected_ids,
        confidence_threshold=config.workflow_confidence_threshold,
        limit=max(config.max_workflow_hints, 0),
    )
    wiki_pages = _wiki_pages(workspace, selected_ids, community_context)
    coverage = coverage_diagnostics(task_understanding, selected_ids)
    return RouterBundle(
        query=config.query,
        selected_skills=selected,
        communities=community_context,
        workflow_hints=workflow_hints,
        wiki_pages=wiki_pages,
        task_understanding=task_understanding,
        coverage_diagnostics=coverage,
        warnings=warnings,
    )


__all__ = [
    "RouterBundle",
    "RouterBundleConfig",
    "RouterCommunityContext",
    "RouterSkillCandidate",
    "RouterWorkflowHint",
    "build_router_bundle",
]
