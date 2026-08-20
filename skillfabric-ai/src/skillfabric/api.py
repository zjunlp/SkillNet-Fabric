"""Public Python facade for SkillFabric workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from skillfabric.graph.builder import (
    BuildConfig,
    BuildResult,
    build_workspace,
)
from skillfabric.planner.package import ExecutionPackageResult, plan_execution_package
from skillfabric.router.config import RouterConfig
from skillfabric.router.models import RouteResult
from skillfabric.router.routing import route_task
from skillfabric.runtime.defaults import default_router_options
from skillfabric.storage import Workspace
from skillfabric.wiki.explorer.backends.base import ExplorerBackendName


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
    ) -> BuildResult:
        return build_workspace(
            BuildConfig(
                skill_root=skill_root,
                workspace=self.workspace.root,
                llm_env_path=self.env_file,
            )
        )

    def route(
        self,
        query: str,
        *,
        backend: ExplorerBackendName = "claude",
        max_selected_skills: int | None = None,
    ) -> RouteResult:
        defaults = default_router_options()
        config = RouterConfig(
            workspace=self.workspace.root,
            query=query,
            env_file=self.env_file,
            max_selected_skills=(
                defaults.max_selected_skills if max_selected_skills is None else max_selected_skills
            ),
            seed_limit=defaults.seed_limit,
            expanded_limit=defaults.expanded_limit,
            max_depth=defaults.max_depth,
            explorer_backend=backend,
            explorer_max_turns=defaults.explorer_max_turns,
            explorer_load_timeout_ms=defaults.explorer_load_timeout_ms,
            explorer_timeout_seconds=defaults.explorer_timeout_seconds,
            explorer_max_attempts=defaults.explorer_max_attempts,
            explorer_retry_delay_seconds=defaults.explorer_retry_delay_seconds,
        )
        return route_task(config)

    def plan(
        self,
        query: str | None = None,
        *,
        route: RouteResult | None = None,
        route_file: str | Path | None = None,
        backend: ExplorerBackendName = "claude",
        max_selected_skills: int | None = None,
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
                backend=backend,
                max_selected_skills=max_selected_skills,
            )
        elif backend != "claude" or max_selected_skills is not None:
            raise TypeError("routing options require plan to perform routing")
        if not resolved_query:
            raise ValueError("plan requires the original task query")
        resolved_root = default_root
        return plan_execution_package(
            self.workspace,
            resolved_route,
            query=resolved_query,
            env_file=self.env_file,
            package_root=resolved_root,
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


__all__ = ["SkillFabric"]
