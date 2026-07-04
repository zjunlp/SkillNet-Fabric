"""Public Python facade for SkillFabric workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from skillfabric.compiled_graph.builder import BuildConfig, BuildResult, build_graph
from skillfabric.indexing.embeddings import (
    ApiEmbeddingProvider,
    DisabledEmbeddingProvider,
)
from skillfabric.orchestrator.package import (
    ExecutionPackageResult,
    PreparedExecutionPackageResult,
    finalize_execution_package,
    prepare_execution_package,
)
from skillfabric.router.config import RouterConfig
from skillfabric.router.models import RouteResult
from skillfabric.router.routing import route_task
from skillfabric.runtime.defaults import default_build_options, default_router_options
from skillfabric.runtime.llm import llm_usage_context
from skillfabric.runtime.metrics import merge_wiki_metrics
from skillfabric.storage import Workspace
from skillfabric.wiki.materializer import build_wiki
from skillfabric.wiki.models import WikiBuildConfig


@dataclass(slots=True)
class SkillFabric:
    """Small facade over the core SkillFabric build, route, and plan workflows."""

    workspace: Workspace | str | Path = ".skillfabric"
    env_file: str | Path = ".env"

    def __post_init__(self) -> None:
        if not isinstance(self.workspace, Workspace):
            self.workspace = Workspace(self.workspace)

    def build(self, skill_root: str | Path, **overrides: object) -> BuildResult:
        """Build registry, indexes, graph, execution artifacts, and wiki for a skill root."""

        _reject_removed_profile(overrides)
        env_file = overrides.pop("env_file", self.env_file)
        defaults = default_build_options()
        skip_wiki = bool(overrides.pop("skip_wiki", False))
        skip_llm_validation = bool(overrides.pop("skip_llm_validation", defaults.skip_llm_validation))
        embedding_model = overrides.pop("embedding_model", None)
        embedding_provider = overrides.pop("embedding_provider", None)
        if embedding_provider is None:
            embedding_provider = _embedding_provider_for_name(
                defaults.embedding_provider,
                env_file=env_file,
                model_id=embedding_model,
            )
        elif isinstance(embedding_provider, str):
            embedding_provider = _embedding_provider_for_name(
                embedding_provider,
                env_file=env_file,
                model_id=embedding_model,
            )
        wiki_summary_mode = str(overrides.pop("wiki_summary_mode", defaults.wiki_summary_mode))
        llm_concurrency = int(overrides.get("llm_concurrency", defaults.llm_concurrency) or defaults.llm_concurrency)
        llm_batch_size = int(overrides.get("llm_batch_size", defaults.llm_batch_size) or defaults.llm_batch_size)
        overrides.setdefault("similar_top_k", defaults.similar_top_k)
        overrides.setdefault("candidate_top_k", defaults.candidate_top_k)
        overrides.setdefault("llm_concurrency", llm_concurrency)
        overrides.setdefault("llm_batch_size", llm_batch_size)
        config = BuildConfig(
            skill_root=skill_root,
            workspace=self.workspace.root,
            llm_env_path=env_file,
            skip_llm_validation=skip_llm_validation,
            embedding_provider=embedding_provider,
            **overrides,
        )
        result = build_graph(config)
        if not skip_wiki:
            with llm_usage_context(
                log_path=self.workspace.reports_dir / "llm_usage.jsonl",
                metadata={"build_id": result.graph.build_id},
            ):
                wiki_result = build_wiki(
                    WikiBuildConfig(
                        workspace=self.workspace.root,
                        env_file=env_file,
                        use_llm_summaries=wiki_summary_mode == "all" and not skip_llm_validation,
                        llm_concurrency=llm_concurrency,
                        llm_batch_size=llm_batch_size,
                    )
                )
            merge_wiki_metrics(self.workspace, wiki_result)
        return result

    def route(self, query: str, **overrides: object) -> RouteResult:
        """Route a user task to selected skills."""

        _reject_removed_profile(overrides)
        defaults = default_router_options()
        explorer_backend = str(overrides.pop("explorer_backend", defaults.explorer_backend))
        use_llm_router = bool(overrides.pop("use_llm_router", defaults.use_llm_router))
        if explorer_backend == "fallback":
            use_llm_router = False
        config = RouterConfig(
            workspace=self.workspace.root,
            query=query,
            env_file=overrides.pop("env_file", self.env_file),
            use_llm_router=use_llm_router,
            explorer_backend=explorer_backend,
            max_selected_skills=int(overrides.pop("max_selected_skills", defaults.max_selected_skills)),
            seed_limit=int(overrides.pop("seed_limit", defaults.seed_limit)),
            expanded_limit=int(overrides.pop("expanded_limit", defaults.expanded_limit)),
            **overrides,
        )
        return route_task(config)

    def plan(
        self,
        query: str | None = None,
        *,
        route: RouteResult | None = None,
        route_file: str | Path | None = None,
        renderer: str = "claude-code",
        **route_overrides: object,
    ) -> ExecutionPackageResult:
        """Direct finalized planning is not available without an agent planner."""

        del renderer, route_overrides
        if query is None and route is None and route_file is None:
            raise ValueError("plan requires query, route, or route_file")
        raise ValueError(
            "SkillFabric.plan() no longer creates a finalized execution package without an agent planner. "
            "Use prepare_plan(...), run the prompt planner, then finalize_plan(...)."
        )

    def prepare_plan(
        self,
        query: str | None = None,
        *,
        route: RouteResult | None = None,
        route_file: str | Path | None = None,
        renderer: str = "claude-code",
        **route_overrides: object,
    ) -> PreparedExecutionPackageResult:
        """Prepare route evidence and selected skill context for a planner pass."""

        resolved_route = route or self._route_from_file(route_file)
        if resolved_route is None:
            if query is None:
                raise ValueError("prepare_plan requires query, route, or route_file")
            resolved_route = self.route(query, **route_overrides)
        return prepare_execution_package(self.workspace, resolved_route, renderer=renderer)

    def finalize_plan(
        self,
        package_root: str | Path,
        planner_output: dict[str, object],
        *,
        renderer: str = "claude-code",
    ) -> ExecutionPackageResult:
        """Validate planner output and write the final workflow/prompt artifacts."""

        return finalize_execution_package(package_root, planner_output, renderer=renderer)

    @staticmethod
    def _route_from_file(route_file: str | Path | None) -> RouteResult | None:
        if route_file is None:
            return None
        payload = json.loads(Path(route_file).read_text(encoding="utf-8"))
        return RouteResult.from_dict(payload)


def _embedding_provider_for_name(
    provider: str,
    *,
    env_file: str | Path,
    model_id: object = None,
):
    normalized = provider.strip().lower()
    if normalized == "disabled":
        return DisabledEmbeddingProvider()
    if normalized == "api":
        return ApiEmbeddingProvider.from_env(
            env_path=env_file,
            model_id=str(model_id) if model_id else None,
        )
    raise ValueError(f"unsupported embedding provider: {provider}. Use 'api' or 'disabled'.")


def _reject_removed_profile(overrides: dict[str, object]) -> None:
    if "profile" not in overrides:
        return
    raise TypeError(
        "SkillFabric public no longer exposes profile=. Use the single public default "
        "settings, or pass explicit options such as embedding_provider='disabled', "
        "wiki_summary_mode='all', or explorer_backend='claude-code'."
    )
