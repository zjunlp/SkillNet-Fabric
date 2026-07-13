from __future__ import annotations

import contextlib
import io

import pytest

from skillfabric.cli import main as cli_main
from skillfabric.runtime.defaults import default_build_options, default_router_options
from skillfabric.runtime.progress import ProgressReporter


def test_public_defaults_have_no_removed_switches_or_unused_wiki_llm_calls() -> None:
    build = default_build_options()
    router = default_router_options()

    assert build.wiki_summary_mode == "off"
    assert not hasattr(build, "embedding_provider")
    assert not hasattr(build, "llm_concurrency")
    assert not hasattr(build, "llm_batch_size")
    assert not hasattr(router, "use_llm_router")
    assert not hasattr(router, "explorer_backend")
    assert router.max_depth == 2


def test_wiki_llm_summaries_remain_an_explicit_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("SKILLFABRIC_WIKI_SUMMARY_MODE", "all")

    assert default_build_options().wiki_summary_mode == "all"


def test_build_defaults_do_not_parse_unrelated_llm_job_environment(monkeypatch) -> None:
    monkeypatch.setenv("SKILLFABRIC_LLM_CONCURRENCY", "not-an-integer")

    assert default_build_options().wiki_summary_mode == "off"


def test_router_defaults_reject_nonfinite_timeout(monkeypatch) -> None:
    monkeypatch.setenv("SKILLFABRIC_EXPLORER_TIMEOUT_SECONDS", "nan")

    with pytest.raises(ValueError, match="SKILLFABRIC_EXPLORER_TIMEOUT_SECONDS"):
        default_router_options()


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


@pytest.mark.parametrize(
    "command,removed_flag",
    [
        ("build", "--embedding-provider"),
        ("build", "--skip-llm-validation"),
        ("route", "--skip-llm-router"),
        ("route", "--explorer-backend"),
        ("route", "--strict-explorer"),
        ("route", "--workflow-confidence-threshold"),
    ],
)
def test_removed_cli_flags_are_rejected(command: str, removed_flag: str) -> None:
    argv = [command]
    if command == "build":
        argv.extend(["--skill-root", "skills"])
    else:
        argv.append("test query")
    argv.append(removed_flag)
    if removed_flag in {
        "--embedding-provider",
        "--explorer-backend",
        "--workflow-confidence-threshold",
    }:
        argv.append("removed")

    with pytest.raises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
        cli_main(argv)


def test_help_exposes_only_useful_embedding_and_route_controls() -> None:
    build_help = io.StringIO()
    with pytest.raises(SystemExit) as build_exit, contextlib.redirect_stdout(build_help):
        cli_main(["build", "--help"])
    route_help = io.StringIO()
    with pytest.raises(SystemExit) as route_exit, contextlib.redirect_stdout(route_help):
        cli_main(["route", "--help"])

    assert build_exit.value.code == 0
    assert route_exit.value.code == 0
    assert "--embedding-model" in build_help.getvalue()
    assert "--embedding-provider" not in build_help.getvalue()
    assert "--max-depth" in route_help.getvalue()
    assert "fallback" not in route_help.getvalue().lower()


def test_progress_reporter_quiet_suppresses_events() -> None:
    stream = io.StringIO()
    reporter = ProgressReporter(enabled=True, json_mode=True, quiet=True, stream=stream)

    with reporter.phase("test.phase"):
        pass

    assert stream.getvalue() == ""
