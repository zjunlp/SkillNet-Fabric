"""Public Python facade for SkillFabric workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skillfabric.graph.builder import (
    BuildConfig,
    BuildResult,
    _BuildDependencies,
    build_workspace,
)
from skillfabric.indexing.embeddings import (
    ApiEmbeddingProvider,
    EmbeddingProvider,
    default_embedding_provider,
)
from skillfabric.planner.package import (
    DEFAULT_PLANNER_CONTEXT_MAX_TOKENS,
    ExecutionPackageResult,
    plan_execution_package,
)
from skillfabric.router.config import RouterConfig
from skillfabric.router.models import RouteResult
from skillfabric.router.routing import route_task
from skillfabric.runtime.defaults import default_router_options
from skillfabric.runtime.jobs import LLMJobOptions
from skillfabric.storage import Workspace
from skillfabric.wiki.explorer.backends.base import ExplorerBackendName, WikiExplorerBackend


@dataclass(slots=True)
class SkillFabric:
    """Small facade over build, route, and planner-package workflows."""

    workspace: Workspace | str | Path = ".skillfabric"
    env_file: str | Path = ".env"

    def __post_init__(self) -> None:
        if not isinstance(self.workspace, Workspace):
            self.workspace = Workspace(self.workspace)

    def build(
        self,
        skill_root: str | Path,
        *,
        env_file: str | Path | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        embedding_model: str | None = None,
        llm_model: str | None = None,
        llm_reasoning_effort: str | None = None,
        llm_progress_every: int | None = None,
    ) -> BuildResult:
        env_file = self.env_file if env_file is None else env_file
        provider = _embedding_provider(
            embedding_provider,
            env_file=env_file,
            model_id=embedding_model,
        )
        llm_options = LLMJobOptions.from_env(
            env_path=env_file,
            progress_every=_optional_int(llm_progress_every, name="llm_progress_every"),
        )
        llm_model = _optional_string(llm_model, name="llm_model")
        llm_reasoning_effort = _optional_string(
            llm_reasoning_effort,
            name="llm_reasoning_effort",
        )
        return build_workspace(
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

    def route(
        self,
        query: str,
        *,
        explorer_backend: WikiExplorerBackend | None = None,
        backend: ExplorerBackendName = "claude",
        sdk_runtime: Any = None,
        embedding_provider: EmbeddingProvider | None = None,
        env_file: str | Path | None = None,
        max_selected_skills: int | None = None,
        required_selected_skills: int | None = None,
        trace_id: str | None = None,
        explorer_model: str | None = None,
        explorer_reasoning_effort: str | None = None,
    ) -> RouteResult:
        if explorer_backend is not None and backend != "claude":
            raise TypeError("backend and explorer_backend cannot be used together")
        defaults = default_router_options()
        config = RouterConfig(
            workspace=self.workspace.root,
            query=query,
            env_file=self.env_file if env_file is None else env_file,
            max_selected_skills=(
                defaults.max_selected_skills if max_selected_skills is None else max_selected_skills
            ),
            required_selected_skills=required_selected_skills,
            seed_limit=defaults.seed_limit,
            expanded_limit=defaults.expanded_limit,
            max_depth=defaults.max_depth,
            trace_id=trace_id,
            explorer_model=explorer_model,
            explorer_reasoning_effort=explorer_reasoning_effort,
            explorer_backend=backend,
            explorer_max_turns=defaults.explorer_max_turns,
            explorer_load_timeout_ms=defaults.explorer_load_timeout_ms,
            explorer_timeout_seconds=defaults.explorer_timeout_seconds,
            explorer_max_attempts=defaults.explorer_max_attempts,
            explorer_retry_delay_seconds=defaults.explorer_retry_delay_seconds,
        )
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
        env_file: str | Path | None = None,
        llm_model: str | None = None,
        llm_reasoning_effort: str | None = None,
        llm_api_key: str | None = None,
        llm_api_base: str | None = None,
        llm_timeout_seconds: float | None = None,
        planner_context_max_tokens: int = DEFAULT_PLANNER_CONTEXT_MAX_TOKENS,
        explorer_backend: WikiExplorerBackend | None = None,
        backend: ExplorerBackendName = "claude",
        sdk_runtime: Any = None,
        embedding_provider: EmbeddingProvider | None = None,
        max_selected_skills: int | None = None,
        required_selected_skills: int | None = None,
        explorer_model: str | None = None,
        explorer_reasoning_effort: str | None = None,
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
                explorer_backend=explorer_backend,
                backend=backend,
                sdk_runtime=sdk_runtime,
                embedding_provider=embedding_provider,
                env_file=env_file,
                max_selected_skills=max_selected_skills,
                required_selected_skills=required_selected_skills,
                explorer_model=explorer_model,
                explorer_reasoning_effort=explorer_reasoning_effort,
            )
        elif any(
            option is not None
            for option in (
                explorer_backend,
                sdk_runtime,
                embedding_provider,
                max_selected_skills,
                required_selected_skills,
                explorer_model,
                explorer_reasoning_effort,
                backend if backend != "claude" else None,
            )
        ):
            raise TypeError("routing options require plan to perform routing")
        if not resolved_query:
            raise ValueError("plan requires the original task query")
        resolved_root = package_root if package_root is not None else default_root
        if resolved_root is not None:
            resolved_root = self._runs_path(resolved_root, label="package_root")
        return plan_execution_package(
            self.workspace,
            resolved_route,
            query=resolved_query,
            env_file=self.env_file if env_file is None else env_file,
            package_root=resolved_root,
            llm_model=llm_model,
            llm_reasoning_effort=llm_reasoning_effort,
            llm_api_key=llm_api_key,
            llm_api_base=llm_api_base,
            llm_timeout_seconds=llm_timeout_seconds,
            planner_context_max_tokens=planner_context_max_tokens,
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


def _optional_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _optional_string(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


__all__ = ["SkillFabric"]
