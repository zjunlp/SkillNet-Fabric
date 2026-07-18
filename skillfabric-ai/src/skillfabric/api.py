"""Public Python facade for SkillFabric workflows."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skillfabric.compiled_graph.builder import (
    BuildConfig,
    BuildResult,
    _BuildDependencies,
    build_graph,
)
from skillfabric.indexing.embeddings import (
    ApiEmbeddingProvider,
    EmbeddingProvider,
    default_embedding_provider,
)
from skillfabric.orchestrator.package import (
    DEFAULT_PLANNER_CONTEXT_MAX_TOKENS,
    DEFAULT_PLANNER_MAX_ATTEMPTS,
    DEFAULT_PLANNER_RETRY_DELAY_SECONDS,
    ExecutionPackageResult,
    plan_execution_package,
)
from skillfabric.router.config import RouterConfig
from skillfabric.router.models import RouteResult
from skillfabric.router.routing import route_task
from skillfabric.runtime.defaults import default_build_options, default_router_options
from skillfabric.runtime.jobs import LLMJobOptions
from skillfabric.runtime.llm import llm_usage_context
from skillfabric.runtime.metrics import merge_wiki_metrics
from skillfabric.storage import Workspace
from skillfabric.wiki.explorer.backends.base import WikiExplorerBackend
from skillfabric.wiki.materializer import build_wiki
from skillfabric.wiki.models import WikiBuildConfig


@dataclass(slots=True)
class SkillFabric:
    """Small facade over build, route, and planner-package workflows."""

    workspace: Workspace | str | Path = ".skillfabric"
    env_file: str | Path = ".env"

    def __post_init__(self) -> None:
        if not isinstance(self.workspace, Workspace):
            self.workspace = Workspace(self.workspace)

    def build(self, skill_root: str | Path, **overrides: Any) -> BuildResult:
        env_file = overrides.pop("env_file", self.env_file)
        defaults = default_build_options()
        skip_wiki = _required_bool(
            overrides.pop("skip_wiki", False),
            name="skip_wiki",
        )
        wiki_summary_mode = overrides.pop("wiki_summary_mode", defaults.wiki_summary_mode)
        if not isinstance(wiki_summary_mode, str):
            raise ValueError("wiki_summary_mode must be a string")
        if wiki_summary_mode not in {"off", "all"}:
            raise ValueError("wiki_summary_mode must be 'off' or 'all'")
        provider = _embedding_provider(
            overrides.pop("embedding_provider", None),
            env_file=env_file,
            model_id=overrides.pop("embedding_model", None),
        )
        llm_options = LLMJobOptions.from_env(
            env_path=env_file,
            concurrency=_optional_int(overrides.pop("llm_concurrency", None)),
            rate_limit_per_minute=_optional_float(overrides.pop("llm_rate_limit_per_minute", None)),
            max_retries=_optional_int(overrides.pop("llm_max_retries", None)),
            retry_backoff_seconds=_optional_float(overrides.pop("llm_retry_backoff_seconds", None)),
            progress_every=_optional_int(overrides.pop("llm_progress_every", None)),
            batch_size=_optional_int(overrides.pop("llm_batch_size", None)),
            checkpoint_interval=_optional_int(overrides.pop("llm_checkpoint_interval", None)),
            circuit_breaker_threshold=_optional_int(
                overrides.pop("llm_circuit_breaker_threshold", None)
            ),
        )
        llm_model = _optional_string(overrides.pop("llm_model", None), name="llm_model")
        llm_reasoning_effort = _optional_string(
            overrides.pop("llm_reasoning_effort", None),
            name="llm_reasoning_effort",
        )
        if overrides:
            raise TypeError(f"unsupported build option(s): {', '.join(sorted(overrides))}")
        result = build_graph(
            BuildConfig(
                skill_root=skill_root,
                workspace=self.workspace.root,
                llm_env_path=env_file,
                llm_options=llm_options,
                llm_model=llm_model,
                llm_reasoning_effort=llm_reasoning_effort,
            ),
            dependencies=_BuildDependencies(embedding_provider=provider),
        )
        if not skip_wiki:
            with llm_usage_context(
                log_path=self.workspace.reports_dir / "llm_usage.jsonl",
                metadata={"build_id": result.graph.build_id},
            ):
                wiki_result = build_wiki(
                    WikiBuildConfig(
                        workspace=self.workspace.root,
                        env_file=env_file,
                        use_llm_summaries=wiki_summary_mode == "all",
                        llm_options=llm_options,
                    )
                )
            merge_wiki_metrics(self.workspace, wiki_result)
        return result

    def route(
        self,
        query: str,
        *,
        explorer_backend: WikiExplorerBackend | None = None,
        **overrides: Any,
    ) -> RouteResult:
        defaults = default_router_options()
        sdk_runtime = overrides.pop("sdk_runtime", None)
        embedding_provider = overrides.pop("embedding_provider", None)
        config = RouterConfig(
            workspace=self.workspace.root,
            query=query,
            env_file=overrides.pop("env_file", self.env_file),
            max_selected_skills=overrides.pop(
                "max_selected_skills",
                defaults.max_selected_skills,
            ),
            seed_limit=overrides.pop("seed_limit", defaults.seed_limit),
            expanded_limit=overrides.pop("expanded_limit", defaults.expanded_limit),
            max_depth=overrides.pop("max_depth", defaults.max_depth),
            trace_id=overrides.pop("trace_id", None),
            explorer_model=overrides.pop("explorer_model", None),
            explorer_max_turns=overrides.pop(
                "explorer_max_turns",
                defaults.explorer_max_turns,
            ),
            explorer_load_timeout_ms=overrides.pop(
                "explorer_load_timeout_ms",
                defaults.explorer_load_timeout_ms,
            ),
            explorer_timeout_seconds=overrides.pop(
                "explorer_timeout_seconds",
                defaults.explorer_timeout_seconds,
            ),
            explorer_max_attempts=overrides.pop(
                "explorer_max_attempts",
                defaults.explorer_max_attempts,
            ),
            explorer_retry_delay_seconds=overrides.pop(
                "explorer_retry_delay_seconds",
                defaults.explorer_retry_delay_seconds,
            ),
        )
        if overrides:
            raise TypeError(f"unsupported route option(s): {', '.join(sorted(overrides))}")
        return route_task(
            config,
            sdk_runtime=sdk_runtime,
            embedding_provider=embedding_provider,
            explorer_backend=explorer_backend,
        )

    def plan(
        self,
        query: str | None = None,
        *,
        route: RouteResult | None = None,
        route_file: str | Path | None = None,
        package_root: str | Path | None = None,
        usage_log_path: str | Path | None = None,
        env_file: str | Path | None = None,
        planner_context_max_tokens: int = DEFAULT_PLANNER_CONTEXT_MAX_TOKENS,
        planner_max_attempts: int = DEFAULT_PLANNER_MAX_ATTEMPTS,
        planner_retry_delay_seconds: float = DEFAULT_PLANNER_RETRY_DELAY_SECONDS,
        **route_overrides: Any,
    ) -> ExecutionPackageResult:
        if route is not None and route_file is not None:
            raise TypeError("plan accepts route and route_file as mutually exclusive inputs")
        route_from_file, file_query, default_root = self._route_context(route_file)
        if route_file is not None:
            if not file_query:
                raise ValueError("route trace has no original query")
            if query is not None and query != file_query:
                raise ValueError("plan query differs from the route query")
        resolved_query = query or file_query
        resolved_route = route or route_from_file
        if resolved_route is None:
            if not resolved_query:
                raise ValueError("plan requires a query, route, or route_file")
            resolved_route = self.route(
                resolved_query,
                env_file=env_file or self.env_file,
                **route_overrides,
            )
        elif route_overrides:
            raise TypeError("route options are only valid when plan performs routing")
        if not resolved_query:
            raise ValueError("plan requires the original task query")
        resolved_root = package_root if package_root is not None else default_root
        if resolved_root is not None:
            resolved_root = self._runs_path(resolved_root, label="package_root")
        resolved_usage_log = (
            None if usage_log_path is None else Path(usage_log_path).expanduser().resolve()
        )
        return plan_execution_package(
            self.workspace,
            resolved_route,
            query=resolved_query,
            env_file=env_file or self.env_file,
            package_root=resolved_root,
            usage_log_path=resolved_usage_log,
            planner_context_max_tokens=planner_context_max_tokens,
            planner_max_attempts=planner_max_attempts,
            planner_retry_delay_seconds=planner_retry_delay_seconds,
        )

    def _route_context(
        self,
        route_file: str | Path | None,
    ) -> tuple[RouteResult | None, str, Path | None]:
        if route_file is None:
            return None, "", None
        route_path = self._runs_path(route_file, label="route_file")
        route = RouteResult.from_dict(json.loads(route_path.read_text(encoding="utf-8")))
        query_path = route_path.parent / "query.json"
        query = ""
        if query_path.is_file():
            payload = json.loads(query_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("route query artifact must be a JSON object")
            raw_query = payload.get("query")
            if not isinstance(raw_query, str) or not raw_query.strip():
                raise ValueError("route query artifact must contain a non-empty string query")
            query = raw_query
        return route, query, route_path.parent / "execution_package"

    def _runs_path(self, value: str | Path, *, label: str) -> Path:
        runs_root = self.workspace.runs_dir.resolve()
        raw_path = Path(value)
        if not str(raw_path):
            raise ValueError(f"{label} must not be empty")
        path = (runs_root / raw_path if not raw_path.is_absolute() else raw_path).resolve()
        try:
            path.relative_to(runs_root)
        except ValueError as exc:
            raise ValueError(f"{label} must stay inside {runs_root}") from exc
        return path


def _embedding_provider(
    provider: Any,
    *,
    env_file: str | Path,
    model_id: Any,
) -> EmbeddingProvider:
    resolved_model = _optional_string(model_id, name="embedding_model")
    if provider is None:
        if resolved_model is not None:
            return ApiEmbeddingProvider.from_env(
                env_path=env_file,
                model_id=resolved_model,
            )
        return default_embedding_provider(env_path=env_file)
    if provider == "api":
        return ApiEmbeddingProvider.from_env(
            env_path=env_file,
            model_id=resolved_model,
        )
    if isinstance(provider, str):
        raise ValueError(f"unsupported embedding provider: {provider}. Use 'api'.")
    if not callable(getattr(provider, "embed", None)):
        raise TypeError("embedding_provider must implement embed(text)")
    return provider


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("integer build options must be integers")
    return value


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("numeric build options must be numbers")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError("numeric build options must be finite")
    return resolved


def _required_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _optional_string(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


__all__ = ["SkillFabric"]
