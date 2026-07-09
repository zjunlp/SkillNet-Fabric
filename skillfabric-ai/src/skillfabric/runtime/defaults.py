"""Public default runtime options for SkillFabric workflows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BuildOptions:
    """Default build behavior for the public package."""

    embedding_provider: str = "api"
    wiki_summary_mode: str = "all"
    llm_concurrency: int = 2
    llm_batch_size: int = 8


@dataclass(frozen=True, slots=True)
class RouterOptions:
    """Default route/plan behavior for the public package."""

    use_llm_router: bool = True
    explorer_backend: str = "claude-code"
    max_selected_skills: int = 8
    seed_limit: int = 8
    expanded_limit: int = 100


def default_build_options() -> BuildOptions:
    """Return the single public build configuration."""

    return BuildOptions()


def default_router_options() -> RouterOptions:
    """Return the single public router configuration."""

    return RouterOptions()
