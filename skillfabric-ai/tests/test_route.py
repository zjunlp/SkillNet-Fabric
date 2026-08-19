from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pytest

import skillfabric.router.traces as trace_module
from skillfabric.router.config import RouterConfig
from skillfabric.router.routing import route_task
from skillfabric.router.traces import _new_trace_id
from skillfabric.wiki.explorer.skill_package import SkillPackage
from tests.support import FakeEmbeddingProvider, build_fixture_workspace


class StubSdkRuntime:
    class ResultMessage:
        def __init__(self, structured_output: dict[str, Any]) -> None:
            self.structured_output = structured_output
            self.is_error = False
            self.subtype = "success"

    class ClaudeAgentOptions:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class HookMatcher:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    def __init__(self, output: dict[str, Any], *, error: Exception | None = None) -> None:
        self.output = output
        self.error = error
        self.prompts: list[str] = []

    async def query(self, *, prompt: Any, options: Any):
        del options
        async for event in prompt:
            self.prompts.append(str(event["message"]["content"]))
        if self.error is not None:
            raise self.error
        yield self.ResultMessage(self.output)


def _package() -> dict[str, Any]:
    return {
        "selected_skills": [
            {
                "skill_id": "skill:pdf-table-parser",
                "role": "Extract structured tables from the PDF.",
                "evidence": [
                    {
                        "path": "skills/cards/pdf-table-parser.md",
                    }
                ],
            },
            {
                "skill_id": "skill:financial-kpi-extractor",
                "role": "Extract KPI values from the normalized tables.",
                "evidence": [
                    {
                        "path": "skills/cards/financial-kpi-extractor.md",
                    }
                ],
            },
        ],
        "near_misses": [
            {"skill_id": "skill:report-writer", "reason": "A final report was not requested."}
        ],
        "coverage_gaps": [],
        "wiki_pages_read": [
            "skills/cards/pdf-table-parser.md",
            "skills/cards/financial-kpi-extractor.md",
            "edges/semantic_edges.jsonl",
        ],
        "rationale": "Parse the PDF before extracting KPIs.",
    }


def test_route_result_uses_only_canonical_fields(tmp_path) -> None:
    workspace = tmp_path / ".skillfabric"
    build_fixture_workspace(workspace)
    runtime = StubSdkRuntime(_package())

    result = route_task(
        RouterConfig(
            workspace=workspace,
            query="extract financial KPIs from a PDF report",
            trace_id="route-success",
            max_selected_skills=4,
        ),
        sdk_runtime=runtime,
        embedding_provider=FakeEmbeddingProvider(),
    )
    payload = result.to_dict()

    assert set(payload) == {
        "selected_skills",
        "relation_evidence",
        "near_misses",
        "coverage_gaps",
        "wiki_pages_read",
        "rationale",
    }
    assert set(payload["selected_skills"][0]) == {"skill_id", "name", "reason", "evidence"}
    assert result.selected_skill_ids == [
        "skill:pdf-table-parser",
        "skill:financial-kpi-extractor",
    ]
    assert result.relation_evidence[0].source_skill == "skill:pdf-table-parser"
    assert result.relation_evidence[0].target_skill == "skill:financial-kpi-extractor"
    trace = workspace / "runs" / "route-success"
    assert json.loads((trace / "route.json").read_text(encoding="utf-8")) == payload
    assert json.loads((trace / "query.json").read_text(encoding="utf-8")) == {
        "query": "extract financial KPIs from a PDF report"
    }


def test_invalid_explorer_skill_fails_without_ranked_fallback(tmp_path) -> None:
    workspace = tmp_path / ".skillfabric"
    build_fixture_workspace(workspace)
    output = _package()
    output["selected_skills"][0]["skill_id"] = "skill:not-real"
    runtime = StubSdkRuntime(output)

    with pytest.raises(ValueError, match="not in task_wiki manifest"):
        route_task(
            RouterConfig(workspace=workspace, query="extract KPIs", trace_id="invalid-skill"),
            sdk_runtime=runtime,
            embedding_provider=FakeEmbeddingProvider(),
        )

    assert not (workspace / "runs" / "invalid-skill" / "route.json").exists()


def test_route_retries_invalid_skill_package_on_the_same_task_wiki(tmp_path) -> None:
    workspace = tmp_path / ".skillfabric"
    build_fixture_workspace(workspace)
    invalid = _package()
    invalid["selected_skills"][0]["skill_id"] = "skill:not-real"

    class RetryingRuntime(StubSdkRuntime):
        def __init__(self) -> None:
            super().__init__(_package())
            self.outputs = [invalid, _package()]
            self.calls = 0

        async def query(self, *, prompt: Any, options: Any):
            del options
            self.calls += 1
            async for event in prompt:
                self.prompts.append(str(event["message"]["content"]))
            yield self.ResultMessage(self.outputs.pop(0))

    runtime = RetryingRuntime()
    result = route_task(
        RouterConfig(
            workspace=workspace,
            query="extract KPIs",
            trace_id="retry-invalid-package",
            explorer_max_attempts=2,
            explorer_retry_delay_seconds=0,
        ),
        sdk_runtime=runtime,
        embedding_provider=FakeEmbeddingProvider(),
    )

    assert runtime.calls == 2
    assert result.selected_skill_ids == [
        "skill:pdf-table-parser",
        "skill:financial-kpi-extractor",
    ]


def test_route_uses_an_explicit_explorer_backend(tmp_path) -> None:
    workspace = tmp_path / ".skillfabric"
    build_fixture_workspace(workspace)

    class Backend:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def explore(self, **kwargs: object) -> SkillPackage:
            self.calls.append(kwargs)
            return SkillPackage.from_dict(_package())

    backend = Backend()
    result = route_task(
        RouterConfig(
            workspace=workspace,
            query="extract KPIs",
            trace_id="explicit-explorer-backend",
        ),
        explorer_backend=backend,
        embedding_provider=FakeEmbeddingProvider(),
    )

    assert result.selected_skill_ids == [
        "skill:pdf-table-parser",
        "skill:financial-kpi-extractor",
    ]
    assert len(backend.calls) == 1
    assert backend.calls[0]["query"] == "extract KPIs"
    assert backend.calls[0]["task_wiki_root"] == (
        workspace / "runs" / "explicit-explorer-backend" / "task_wiki"
    )


def test_route_rejects_sdk_runtime_with_an_explicit_backend(tmp_path) -> None:
    with pytest.raises(TypeError, match=r"sdk_runtime.*explorer_backend"):
        route_task(
            RouterConfig(workspace=tmp_path, query="extract KPIs"),
            sdk_runtime=object(),
            explorer_backend=object(),
            embedding_provider=FakeEmbeddingProvider(),
        )


def test_route_rejects_named_and_explicit_explorer_backends(tmp_path) -> None:
    with pytest.raises(TypeError, match=r"explorer_backend.*config.explorer_backend"):
        route_task(
            RouterConfig(
                workspace=tmp_path,
                query="extract KPIs",
                explorer_backend="codex",
            ),
            explorer_backend=object(),
            embedding_provider=FakeEmbeddingProvider(),
        )


def test_graph_dependency_does_not_expand_explorer_selection(tmp_path) -> None:
    workspace = tmp_path / ".skillfabric"
    build_fixture_workspace(workspace)
    output = _package()
    output["selected_skills"] = output["selected_skills"][1:]
    output["wiki_pages_read"] = ["skills/cards/financial-kpi-extractor.md"]

    result = route_task(
        RouterConfig(workspace=workspace, query="extract KPIs", trace_id="no-closure"),
        sdk_runtime=StubSdkRuntime(output),
        embedding_provider=FakeEmbeddingProvider(),
    )

    assert result.selected_skill_ids == ["skill:financial-kpi-extractor"]
    assert result.relation_evidence == ()


def test_explorer_failure_is_propagated_and_diagnostic_is_written(tmp_path) -> None:
    workspace = tmp_path / ".skillfabric"
    build_fixture_workspace(workspace)

    with pytest.raises(RuntimeError, match="explorer unavailable"):
        route_task(
            RouterConfig(workspace=workspace, query="parse PDF", trace_id="sdk-failure"),
            sdk_runtime=StubSdkRuntime({}, error=RuntimeError("explorer unavailable")),
            embedding_provider=FakeEmbeddingProvider(),
        )

    trace = workspace / "runs" / "sdk-failure"
    assert (trace / "explorer" / "error.json").exists()
    assert not (trace / "route.json").exists()


def test_empty_selection_with_coverage_gap_is_a_valid_result(tmp_path) -> None:
    workspace = tmp_path / ".skillfabric"
    build_fixture_workspace(workspace)
    output = {
        "selected_skills": [],
        "near_misses": [],
        "coverage_gaps": ["No skill can perform the requested database migration."],
        "wiki_pages_read": [],
        "rationale": "The test skill corpus does not cover the task.",
    }

    result = route_task(
        RouterConfig(workspace=workspace, query="migrate an Oracle database", trace_id="gap-route"),
        sdk_runtime=StubSdkRuntime(output),
        embedding_provider=FakeEmbeddingProvider(),
    )

    assert result.selected_skills == ()
    assert result.coverage_gaps


@pytest.mark.parametrize(
    "trace_id",
    [
        "",
        ".",
        "..",
        "../escape",
        "nested/trace",
        r"nested\trace",
        "-leading",
        "has space",
        "line\nbreak",
        "name:colon",
        "a" * 129,
    ],
)
def test_router_config_rejects_unsafe_trace_ids(trace_id) -> None:
    with pytest.raises(ValueError, match="trace_id"):
        RouterConfig(trace_id=trace_id)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"max_selected_skills": -1}, "max_selected_skills"),
        ({"required_selected_skills": True}, "required_selected_skills"),
        ({"required_selected_skills": -1}, "required_selected_skills"),
        (
            {"max_selected_skills": 5, "required_selected_skills": 6},
            "required_selected_skills",
        ),
        ({"seed_limit": -1}, "seed_limit"),
        ({"seed_limit": 2, "expanded_limit": 1}, "expanded_limit"),
        ({"max_depth": -1}, "max_depth"),
        ({"explorer_max_turns": 0}, "explorer_max_turns"),
        ({"explorer_load_timeout_ms": 999}, "explorer_load_timeout_ms"),
        ({"explorer_timeout_seconds": -0.5}, "explorer_timeout_seconds"),
        ({"explorer_timeout_seconds": float("nan")}, "explorer_timeout_seconds"),
        ({"explorer_timeout_seconds": float("inf")}, "explorer_timeout_seconds"),
        ({"explorer_max_attempts": 0}, "explorer_max_attempts"),
        ({"explorer_retry_delay_seconds": -1}, "explorer_retry_delay_seconds"),
        ({"explorer_retry_delay_seconds": float("nan")}, "explorer_retry_delay_seconds"),
    ],
)
def test_router_config_rejects_invalid_numeric_limits(overrides, match) -> None:
    with pytest.raises(ValueError, match=match):
        RouterConfig(**overrides)


def test_router_config_allows_explicit_zero_budgets() -> None:
    config = RouterConfig(
        max_selected_skills=0,
        required_selected_skills=0,
        seed_limit=0,
        expanded_limit=0,
        max_depth=0,
    )

    assert config.max_selected_skills == 0
    assert config.required_selected_skills == 0
    assert config.seed_limit == 0
    assert config.expanded_limit == 0
    assert config.max_depth == 0


def test_router_config_allows_zero_to_disable_explorer_timeout() -> None:
    config = RouterConfig(explorer_timeout_seconds=0)

    assert config.explorer_timeout_seconds == 0


def test_route_rejects_blank_query_before_creating_workspace(tmp_path) -> None:
    workspace = tmp_path / ".skillfabric"

    with pytest.raises(ValueError, match="query"):
        route_task(RouterConfig(workspace=workspace, query="   "))

    assert not workspace.exists()


def test_generated_trace_ids_are_unique_within_the_same_second(monkeypatch) -> None:
    class FixedDatetime:
        @classmethod
        def now(cls):
            return datetime(2026, 7, 12, 12, 0, 0)

    monkeypatch.setattr(trace_module, "datetime", FixedDatetime)

    assert _new_trace_id("same query") != _new_trace_id("same query")


def test_route_refuses_to_overwrite_an_existing_trace(tmp_path) -> None:
    workspace = tmp_path / ".skillfabric"
    build_fixture_workspace(workspace)
    trace = workspace / "runs" / "existing-trace"
    trace.mkdir(parents=True)
    route_path = trace / "route.json"
    route_path.write_text('{"preserve": true}\n', encoding="utf-8")

    with pytest.raises(FileExistsError, match="route trace already exists"):
        route_task(
            RouterConfig(
                workspace=workspace,
                query="extract KPIs",
                trace_id="existing-trace",
            ),
            sdk_runtime=StubSdkRuntime(_package()),
            embedding_provider=FakeEmbeddingProvider(),
        )

    assert json.loads(route_path.read_text(encoding="utf-8")) == {"preserve": True}
