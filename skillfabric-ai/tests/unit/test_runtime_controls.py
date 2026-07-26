from __future__ import annotations

import contextlib
import io

import pytest

from skillfabric.cli import main as cli_main
from skillfabric.runtime.defaults import default_router_options
from skillfabric.runtime.progress import ProgressReporter


def test_public_defaults_match_the_documented_workflow() -> None:
    router = default_router_options()

    assert router.max_selected_skills == 8
    assert router.seed_limit == 24
    assert router.expanded_limit == 100
    assert router.max_depth == 2


def test_router_defaults_reject_nonfinite_timeout(monkeypatch) -> None:
    monkeypatch.setenv("SKILLFABRIC_EXPLORER_TIMEOUT_SECONDS", "nan")

    with pytest.raises(ValueError, match="SKILLFABRIC_EXPLORER_TIMEOUT_SECONDS"):
        default_router_options()


def test_router_defaults_allow_zero_to_disable_explorer_timeout(monkeypatch) -> None:
    monkeypatch.setenv("SKILLFABRIC_EXPLORER_TIMEOUT_SECONDS", "0")

    router = default_router_options()

    assert router.explorer_timeout_seconds == 0


def test_router_defaults_preserve_explicit_zero_budgets(monkeypatch) -> None:
    monkeypatch.setenv("SKILLFABRIC_MAX_SELECTED_SKILLS", "0")
    monkeypatch.setenv("SKILLFABRIC_SEED_LIMIT", "0")
    monkeypatch.setenv("SKILLFABRIC_EXPANDED_LIMIT", "0")

    router = default_router_options()

    assert router.max_selected_skills == 0
    assert router.seed_limit == 0
    assert router.expanded_limit == 0


def test_router_defaults_reject_expanded_limit_below_seed_limit(monkeypatch) -> None:
    monkeypatch.setenv("SKILLFABRIC_SEED_LIMIT", "3")
    monkeypatch.setenv("SKILLFABRIC_EXPANDED_LIMIT", "2")

    with pytest.raises(ValueError, match="expanded_limit"):
        default_router_options()


def test_help_exposes_embedding_and_route_controls() -> None:
    build_help = io.StringIO()
    with pytest.raises(SystemExit) as build_exit, contextlib.redirect_stdout(build_help):
        cli_main(["build", "--help"])
    route_help = io.StringIO()
    with pytest.raises(SystemExit) as route_exit, contextlib.redirect_stdout(route_help):
        cli_main(["route", "--help"])

    assert build_exit.value.code == 0
    assert route_exit.value.code == 0
    assert "--embedding-model" in build_help.getvalue()
    assert "--llm-model" in build_help.getvalue()
    assert "--llm-reasoning-effort" in build_help.getvalue()
    assert "--wiki-summary-mode" not in build_help.getvalue()
    assert "--max-depth" in route_help.getvalue()


def test_progress_reporter_quiet_suppresses_events() -> None:
    stream = io.StringIO()
    reporter = ProgressReporter(enabled=True, json_mode=True, quiet=True, stream=stream)

    with reporter.phase("test.phase"):
        pass

    assert stream.getvalue() == ""
