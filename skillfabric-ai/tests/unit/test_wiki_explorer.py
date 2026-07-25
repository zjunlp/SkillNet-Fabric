from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from skillfabric.wiki.explorer.agent import WikiExplorerConfig, explore_query_wiki
from skillfabric.wiki.explorer.backends.claude_code import ClaudeCodeWikiExplorerBackend
from skillfabric.wiki.explorer.skill_package import SkillPackage, skill_package_json_schema


def _empty_package() -> dict[str, Any]:
    return {
        "selected_skills": [],
        "near_misses": [],
        "coverage_gaps": ["No matching skill in this test corpus."],
        "wiki_pages_read": [],
        "rationale": "The bounded corpus does not cover the task.",
    }


class StubRuntime:
    class ResultMessage:
        def __init__(self, structured_output: Any, **metrics: Any) -> None:
            self.structured_output = structured_output
            self.duration_ms = metrics.get("duration_ms", 1)
            self.total_cost_usd = metrics.get("total_cost_usd", 0.0)
            self.usage = metrics.get(
                "usage",
                {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            )
            self.model_usage = metrics.get("model_usage", {})
            self.num_turns = metrics.get("num_turns", 1)
            self.is_error = metrics.get("is_error", False)
            self.subtype = metrics.get("subtype", "success")
            self.result = metrics.get("result", "")

    class ClaudeAgentOptions:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class HookMatcher:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    def __init__(self, output: Any, *, metrics: dict[str, Any] | None = None) -> None:
        self.output = output
        self.metrics = metrics or {}
        self.options: Any = None
        self.calls = 0

    async def query(self, *, prompt: Any, options: Any):
        self.calls += 1
        self.options = options
        async for _event in prompt:
            pass
        yield self.ResultMessage(self.output, **self.metrics)


def _query_root(tmp_path: Path) -> Path:
    root = tmp_path / "query_wiki"
    root.mkdir()
    (root / "index.md").write_text("# Query Wiki\n", encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "query": "x",
                "skills": [],
                "semantic_edges_path": "edges/semantic_edges.jsonl",
                "alternatives": [],
            }
        ),
        encoding="utf-8",
    )
    return root


def test_backend_uses_the_canonical_schema_and_writes_route_artifacts(tmp_path) -> None:
    root = _query_root(tmp_path)
    runtime = StubRuntime(
        _empty_package(),
        metrics={
            "usage": {
                "input_tokens": 123,
                "output_tokens": 17,
                "cache_creation_input_tokens": 19,
                "cache_read_input_tokens": 23,
            }
        },
    )
    trace = tmp_path / "trace"

    package = ClaudeCodeWikiExplorerBackend(sdk_runtime=runtime).explore(
        query="unsupported task",
        query_wiki_root=root,
        trace_dir=trace,
    )

    assert package.coverage_gaps
    assert runtime.options.output_format["schema"] == skill_package_json_schema()
    assert json.loads((trace / "cc_explorer" / "skill_package.json").read_text())["coverage_gaps"]
    usage = json.loads((trace / "cc_explorer" / "usage.json").read_text())
    assert usage["input_tokens"] == 123
    assert usage["output_tokens"] == 17
    assert usage["cache_creation_input_tokens"] == 19
    assert usage["cache_read_input_tokens"] == 23


def test_backend_uses_explicit_model_and_reasoning_over_env_file(tmp_path) -> None:
    root = _query_root(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MODEL=gpt-5.4-mini\nSKILLFABRIC_LLM_REASONING_EFFORT=medium\n",
        encoding="utf-8",
    )
    runtime = StubRuntime(_empty_package())

    ClaudeCodeWikiExplorerBackend(
        env_file=env_file,
        model="gpt-5.6-terra",
        reasoning_effort="xhigh",
        sdk_runtime=runtime,
    ).explore(query="unsupported task", query_wiki_root=root, trace_dir=tmp_path / "trace")

    assert runtime.options.model == "gpt-5.6-terra"
    assert runtime.options.effort == "xhigh"
    assert runtime.options.env["ANTHROPIC_MODEL"] == "gpt-5.6-terra"
    assert runtime.options.env["ANTHROPIC_REASONING_EFFORT"] == "xhigh"


def test_backend_default_tool_budget_scales_with_the_selection_limit(tmp_path) -> None:
    root = _query_root(tmp_path)
    trace = tmp_path / "trace"

    ClaudeCodeWikiExplorerBackend(
        sdk_runtime=StubRuntime(_empty_package()),
        max_selected_skills=12,
    ).explore(query="unsupported task", query_wiki_root=root, trace_dir=trace)

    context = json.loads((trace / "cc_explorer" / "prompt_context.json").read_text())
    assert context["tool_budget"]["Read"] >= 26
    assert context["tool_budget"]["total"] >= context["tool_budget"]["Read"]


def test_backend_omits_thinking_token_system_events(tmp_path) -> None:
    class ThinkingRuntime(StubRuntime):
        class SystemMessage:
            subtype = "thinking_tokens"

        async def query(self, *, prompt: Any, options: Any):
            self.calls += 1
            self.options = options
            async for _event in prompt:
                pass
            yield self.SystemMessage()
            yield self.ResultMessage(self.output, **self.metrics)

    root = _query_root(tmp_path)
    trace = tmp_path / "trace"

    ClaudeCodeWikiExplorerBackend(sdk_runtime=ThinkingRuntime(_empty_package())).explore(
        query="unsupported task",
        query_wiki_root=root,
        trace_dir=trace,
    )

    events = [
        json.loads(line)
        for line in (trace / "cc_explorer" / "agent_events.jsonl").read_text().splitlines()
    ]
    assert not any(event.get("subtype") == "thinking_tokens" for event in events)
    assert any(event.get("type") == "ResultMessage" for event in events)


def test_backend_rejects_missing_structured_output_without_text_recovery(tmp_path) -> None:
    root = _query_root(tmp_path)
    trace = tmp_path / "trace"

    with pytest.raises(RuntimeError, match="structured SkillPackage"):
        ClaudeCodeWikiExplorerBackend(sdk_runtime=StubRuntime(None)).explore(
            query="x",
            query_wiki_root=root,
            trace_dir=trace,
        )

    assert (trace / "cc_explorer" / "error.json").exists()


def test_permissions_enforce_read_root_and_count_each_allowed_tool_once(tmp_path) -> None:
    root = _query_root(tmp_path)
    runtime = StubRuntime(_empty_package())
    backend = ClaudeCodeWikiExplorerBackend(
        sdk_runtime=runtime,
        tool_budget={"Read": 1, "LS": 1, "Glob": 1, "Grep": 1, "total": 1},
    )
    backend.explore(query="x", query_wiki_root=root, trace_dir=tmp_path / "trace")

    hook = runtime.options.hooks["PreToolUse"][0].hooks[0]

    async def check_permissions():
        allowed = await hook(
            {
                "tool_name": "Read",
                "tool_input": {"file_path": str(root / "index.md")},
            },
            None,
            {},
        )
        exhausted = await hook(
            {
                "tool_name": "Read",
                "tool_input": {"file_path": str(root / "index.md")},
            },
            None,
            {},
        )
        outside = await hook(
            {
                "tool_name": "Read",
                "tool_input": {"file_path": str(tmp_path / "outside.md")},
            },
            None,
            {},
        )
        return allowed, exhausted, outside

    allowed, exhausted, outside = asyncio.run(check_permissions())
    assert allowed["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert exhausted["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert outside["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.parametrize(
    "tool_budget",
    [
        {"Read": -1},
        {"Unknown": 1},
        {"Read": "many"},
    ],
)
def test_backend_rejects_invalid_tool_budgets(tmp_path, tool_budget) -> None:
    root = _query_root(tmp_path)

    with pytest.raises(ValueError, match="tool_budget"):
        ClaudeCodeWikiExplorerBackend(
            sdk_runtime=StubRuntime(_empty_package()),
            tool_budget=tool_budget,
        ).explore(query="x", query_wiki_root=root, trace_dir=tmp_path / "trace")


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_selected_skills": -1},
        {"max_turns": 0},
        {"load_timeout_ms": 999},
        {"execution_timeout_seconds": float("inf")},
    ],
)
def test_backend_rejects_invalid_runtime_limits(overrides) -> None:
    with pytest.raises(ValueError):
        ClaudeCodeWikiExplorerBackend(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_attempts": 0},
        {"retry_delay_seconds": -1},
        {"retry_delay_seconds": float("nan")},
    ],
)
def test_explorer_rejects_invalid_retry_limits(overrides) -> None:
    with pytest.raises(ValueError):
        WikiExplorerConfig(**overrides)


def test_transient_sdk_failure_retries_once_then_returns_strict_output(tmp_path, caplog) -> None:
    root = _query_root(tmp_path)

    class FlakyRuntime(StubRuntime):
        async def query(self, *, prompt: Any, options: Any):
            self.calls += 1
            self.options = options
            async for _event in prompt:
                pass
            if self.calls == 1:
                raise RuntimeError("503 service temporarily unavailable")
            yield self.ResultMessage(self.output)

    runtime = FlakyRuntime(_empty_package())
    package = explore_query_wiki(
        WikiExplorerConfig(max_attempts=2, retry_delay_seconds=0),
        query="x",
        query_wiki_root=root,
        trace_dir=tmp_path / "trace",
        sdk_runtime=runtime,
    )

    assert package.package.coverage_gaps
    assert runtime.calls == 2
    assert "explorer_retry attempt=1/2" in caplog.text


def test_injected_backend_failure_uses_the_existing_outer_recovery(tmp_path) -> None:
    root = _query_root(tmp_path)

    class Backend:
        def __init__(self) -> None:
            self.calls = 0

        def explore(self, **_kwargs: object) -> SkillPackage:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient Codex SDK failure")
            return SkillPackage.from_dict(_empty_package())

    backend = Backend()
    run = explore_query_wiki(
        WikiExplorerConfig(max_attempts=2, retry_delay_seconds=0),
        query="find a skill",
        query_wiki_root=root,
        trace_dir=tmp_path / "trace",
        backend=backend,
    )

    assert backend.calls == 2
    assert run.package.to_dict() == _empty_package()


def test_explorer_publishes_all_attempts_and_terminal_closure(tmp_path) -> None:
    root = _query_root(tmp_path)

    class Backend:
        def __init__(self) -> None:
            self.calls = 0

        def explore(self, *, trace_dir: Path, **_kwargs: object) -> SkillPackage:
            self.calls += 1
            explorer = trace_dir / "cc_explorer"
            explorer.mkdir(parents=True)
            (explorer / "usage.json").write_text(
                json.dumps({"input_tokens": self.calls, "output_tokens": 1}),
                encoding="utf-8",
            )
            if self.calls == 1:
                raise RuntimeError("503 service temporarily unavailable")
            return SkillPackage.from_dict(_empty_package())

    trace = tmp_path / "trace"
    run = explore_query_wiki(
        WikiExplorerConfig(max_attempts=2, retry_delay_seconds=0),
        query="find a skill",
        query_wiki_root=root,
        trace_dir=trace,
        backend=Backend(),
    )

    assert run.validation.valid
    closure = json.loads((trace / "cc_explorer" / "closure.json").read_text())
    assert closure["status"] == "completed"
    assert closure["outcome"] == "completed_empty"
    assert closure["winning_attempt"] == 2
    assert [item["status"] for item in closure["attempts"]] == ["failed", "completed"]
    assert closure["attempts"][0]["failure_kind"] == "retryable_runtime"
    assert (trace / "cc_explorer" / "attempts" / "attempt-01" / "error.json").exists()
    assert (trace / "cc_explorer" / "attempts" / "attempt-02" / "skill_package.json").exists()
    usage = json.loads((trace / "cc_explorer" / "usage.json").read_text())
    assert usage["attempts"] == [
        {"input_tokens": 1, "output_tokens": 1},
        {"input_tokens": 2, "output_tokens": 1},
    ]


def test_explorer_stops_after_an_unmetered_started_attempt(tmp_path) -> None:
    root = _query_root(tmp_path)

    class Backend:
        calls = 0

        def explore(self, *, trace_dir: Path, **_kwargs: object) -> SkillPackage:
            self.calls += 1
            explorer = trace_dir / "cc_explorer"
            explorer.mkdir(parents=True)
            (explorer / "turn_state.json").write_text(
                '{"schema_version":1,"turn_started":true}',
                encoding="utf-8",
            )
            raise RuntimeError("503 service temporarily unavailable")

    backend = Backend()
    with pytest.raises(RuntimeError, match="503 service"):
        explore_query_wiki(
            WikiExplorerConfig(max_attempts=2, retry_delay_seconds=0),
            query="x",
            query_wiki_root=root,
            trace_dir=tmp_path / "trace",
            backend=backend,
        )
    assert backend.calls == 1
    closure = json.loads(
        (tmp_path / "trace" / "cc_explorer" / "closure.json").read_text(encoding="utf-8")
    )
    assert closure["status"] == "route_failed"
    assert closure["attempts"][0]["unmetered_attempt"] is True
    assert closure["attempts"][0]["failure_kind"] == "unmetered_attempt"
    assert closure["attempts"][0]["retryable"] is False


def test_explorer_does_not_retry_runtime_or_authentication_mismatch(tmp_path) -> None:
    root = _query_root(tmp_path)

    class Backend:
        calls = 0

        def explore(self, **_kwargs: object) -> SkillPackage:
            self.calls += 1
            raise RuntimeError("runtime mismatch: invalid API key")

    backend = Backend()
    with pytest.raises(RuntimeError, match="runtime mismatch"):
        explore_query_wiki(
            WikiExplorerConfig(max_attempts=2, retry_delay_seconds=0),
            query="x",
            query_wiki_root=root,
            trace_dir=tmp_path / "trace",
            backend=backend,
        )

    assert backend.calls == 1
    closure = json.loads(
        (tmp_path / "trace" / "cc_explorer" / "closure.json").read_text(encoding="utf-8")
    )
    assert closure["attempts"][0]["retryable"] is False
    assert closure["attempts"][0]["failure_kind"] == "non_retryable_runtime"


def test_outer_attempt_closure_redacts_runtime_paths_and_secrets(tmp_path) -> None:
    root = _query_root(tmp_path)
    env_file = tmp_path / "private-runtime.env"

    class Backend:
        def explore(self, *, query_wiki_root: Path, trace_dir: Path, **_kwargs: object):
            explorer = trace_dir / "cc_explorer"
            explorer.mkdir(parents=True)
            raise RuntimeError(
                f"config mismatch at {query_wiki_root} via {trace_dir}; "
                f"env={env_file}; OPENAI_API_KEY=sk-private-token"
            )

    trace = tmp_path / "public-trace"
    with pytest.raises(RuntimeError, match="config mismatch"):
        explore_query_wiki(
            WikiExplorerConfig(
                env_file=env_file,
                max_attempts=1,
                retry_delay_seconds=0,
            ),
            query="x",
            query_wiki_root=root,
            trace_dir=trace,
            backend=Backend(),
        )

    attempt_text = (trace / "cc_explorer" / "attempts" / "attempt-01" / "attempt.json").read_text(
        encoding="utf-8"
    )
    closure_text = (trace / "cc_explorer" / "closure.json").read_text(encoding="utf-8")
    for sensitive in (str(root), str(trace), str(env_file), "sk-private-token"):
        assert sensitive not in attempt_text
        assert sensitive not in closure_text


def test_explorer_rejects_sdk_runtime_with_an_explicit_backend(tmp_path) -> None:
    root = _query_root(tmp_path)

    class Backend:
        def explore(self, **_kwargs: object) -> SkillPackage:
            return SkillPackage.from_dict(_empty_package())

    with pytest.raises(TypeError, match=r"sdk_runtime.*backend"):
        explore_query_wiki(
            WikiExplorerConfig(),
            query="find a skill",
            query_wiki_root=root,
            trace_dir=tmp_path / "trace",
            sdk_runtime=object(),
            backend=Backend(),
        )


def test_explorer_uses_an_explicit_falsey_backend(tmp_path) -> None:
    root = _query_root(tmp_path)

    class Backend:
        def __bool__(self) -> bool:
            return False

        def explore(self, **_kwargs: object) -> SkillPackage:
            return SkillPackage.from_dict(_empty_package())

    with patch(
        "skillfabric.wiki.explorer.agent.ClaudeCodeWikiExplorerBackend",
        side_effect=AssertionError("explicit backend was ignored"),
    ):
        run = explore_query_wiki(
            WikiExplorerConfig(),
            query="find a skill",
            query_wiki_root=root,
            trace_dir=tmp_path / "trace",
            backend=Backend(),
        )

    assert run.package.to_dict() == _empty_package()


def test_non_retryable_failure_propagates_and_is_redacted_in_trace(tmp_path) -> None:
    root = _query_root(tmp_path)

    class BrokenRuntime(StubRuntime):
        async def query(self, *, prompt: Any, options: Any):
            del prompt, options
            self.calls += 1
            raise RuntimeError("invalid request API_KEY=sk-secret-value")
            yield

    trace = tmp_path / "trace"
    with pytest.raises(RuntimeError, match="invalid request"):
        ClaudeCodeWikiExplorerBackend(sdk_runtime=BrokenRuntime(_empty_package())).explore(
            query="x",
            query_wiki_root=root,
            trace_dir=trace,
        )

    error_text = (trace / "cc_explorer" / "error.json").read_text(encoding="utf-8")
    assert "sk-secret-value" not in error_text
    assert "[redacted]" in error_text


def test_backend_runs_when_called_inside_an_existing_event_loop(tmp_path) -> None:
    root = _query_root(tmp_path)
    runtime = StubRuntime(_empty_package())

    async def run_backend():
        return ClaudeCodeWikiExplorerBackend(sdk_runtime=runtime).explore(
            query="x",
            query_wiki_root=root,
            trace_dir=tmp_path / "trace",
        )

    package = asyncio.run(run_backend())

    assert package.coverage_gaps
