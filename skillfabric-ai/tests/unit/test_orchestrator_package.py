from __future__ import annotations

import json

import pytest

import skillfabric.orchestrator.package as package_module
from skillfabric.orchestrator.package import (
    PLANNER_PROMPT_ID,
    plan_execution_package,
    planner_output_json_schema,
    validate_planner_output,
)
from skillfabric.router.models import (
    RouteRelationEvidence,
    RouteResult,
    RouteSelectedSkill,
)
from tests.unit.wiki_helpers import build_fixture_workspace


@pytest.fixture(autouse=True)
def _planner_config(monkeypatch):
    config = package_module.LLMConfig(
        api_base="https://example.test/v1",
        api_key="test-key",
        model="openai/test-model",
    )
    monkeypatch.setattr(
        package_module.LLMConfig,
        "from_env",
        lambda **_kwargs: config,
    )


def _route() -> RouteResult:
    return RouteResult(
        selected_skills=(
            RouteSelectedSkill(
                skill_id="skill:pdf-table-parser",
                name="pdf-table-parser",
                reason="Parse PDF tables.",
                evidence=("skills/cards/pdf-table-parser.md",),
            ),
            RouteSelectedSkill(
                skill_id="skill:financial-kpi-extractor",
                name="financial-kpi-extractor",
                reason="Extract financial KPI values.",
                evidence=("skills/cards/financial-kpi-extractor.md",),
            ),
        ),
        relation_evidence=(
            RouteRelationEvidence(
                relation_type="depend_on",
                source_skill="skill:financial-kpi-extractor",
                target_skill="skill:pdf-table-parser",
                confidence=0.94,
                reason="Normalized tables may be useful before KPI extraction.",
                evidence=("skill:financial-kpi-extractor:12", "skill:pdf-table-parser:9"),
            ),
        ),
        near_misses=(),
        coverage_gaps=("Narrative report writing is outside this route.",),
        wiki_pages_read=(
            "skills/cards/pdf-table-parser.md",
            "skills/cards/financial-kpi-extractor.md",
            "edges/semantic_edges.jsonl",
        ),
        rationale="These skills cover parsing and KPI extraction.",
    )


def _planner_response() -> str:
    return json.dumps(
        {
            "execution_prompt": (
                "Parse the PDF tables, extract the requested KPIs, and verify every value "
                "against the source. Use independent extraction checks in parallel only when "
                "they do not share mutable state."
            )
        }
    )


def test_plan_calls_llm_once_with_complete_selected_context(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / ".skillfabric"
    build_fixture_workspace(workspace)
    calls: list[dict[str, object]] = []

    def completion(**kwargs):
        calls.append(kwargs)
        return _planner_response()

    counted_models: list[str | None] = []

    def count_tokens(_messages, *, model=None):
        counted_models.append(model)
        return 1200

    monkeypatch.setattr(package_module, "litellm_completion", completion)
    monkeypatch.setattr(package_module, "count_message_tokens", count_tokens)
    package_root = workspace / "runs" / "planner-test" / "execution_package"

    result = plan_execution_package(
        workspace,
        _route(),
        query="extract financial KPIs from a PDF report",
        env_file=tmp_path / "unused.env",
        package_root=package_root,
        planner_context_max_tokens=10_000,
    )

    assert len(calls) == 1
    assert PLANNER_PROMPT_ID == "skillfabric_execution_planner"
    messages = calls[0]["messages"]
    prompt = "\n".join(str(item["content"]) for item in messages)  # type: ignore[index]
    assert "skill:pdf-table-parser" in prompt
    assert "skill:financial-kpi-extractor" in prompt
    assert "&lt;untrusted_skill_source" in prompt
    assert "&quot;" not in prompt
    assert "relation_evidence" in prompt
    assert counted_models == ["openai/test-model"]
    assert result.estimated_prompt_tokens == 1200
    assert result.prompt_path.read_text().startswith("Parse the PDF tables")
    assert not (package_root / "workflow_plan.json").exists()
    assert not (package_root / "PLANNER.md").exists()
    assert json.loads(result.planner_output_path.read_text()) == json.loads(_planner_response())
    request = json.loads((package_root / "planner_request.json").read_text())
    assert request["expected_schema"] == planner_output_json_schema()
    assert request["estimated_prompt_tokens"] == 1200


def test_plan_rejects_context_overflow_before_creating_package(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / ".skillfabric"
    build_fixture_workspace(workspace)
    package_root = workspace / "runs" / "overflow" / "execution_package"
    monkeypatch.setattr(package_module, "count_message_tokens", lambda *_args, **_kwargs: 5001)

    def unexpected_completion(**_kwargs):
        raise AssertionError("planner must not be called after context overflow")

    monkeypatch.setattr(package_module, "litellm_completion", unexpected_completion)

    with pytest.raises(ValueError, match="planner context requires 5001 tokens"):
        plan_execution_package(
            workspace,
            _route(),
            query="extract financial KPIs",
            package_root=package_root,
            planner_context_max_tokens=5000,
        )

    assert not package_root.exists()


def test_plan_rejects_invalid_planner_output_without_prompt(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / ".skillfabric"
    build_fixture_workspace(workspace)
    monkeypatch.setattr(package_module, "count_message_tokens", lambda *_args, **_kwargs: 100)
    monkeypatch.setattr(
        package_module,
        "litellm_completion",
        lambda **_kwargs: json.dumps(
            {
                "workflow_plan": {"steps": []},
                "execution_prompt": "Do the task.",
            }
        ),
    )
    package_root = workspace / "runs" / "invalid" / "execution_package"

    with pytest.raises(ValueError, match="exactly execution_prompt"):
        plan_execution_package(
            workspace,
            _route(),
            query="extract financial KPIs",
            package_root=package_root,
        )

    assert not (package_root / "execution_prompt.md").exists()
    validation = json.loads((package_root / "planner_validation.json").read_text())
    assert validation["valid"] is False


def test_plan_refuses_to_overwrite_an_existing_package(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / ".skillfabric"
    build_fixture_workspace(workspace)
    package_root = workspace / "runs" / "existing-package"
    package_root.mkdir(parents=True)
    marker = package_root / "preserve.txt"
    marker.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(package_module, "count_message_tokens", lambda *_args, **_kwargs: 100)

    with pytest.raises(FileExistsError, match="already exists"):
        plan_execution_package(
            workspace,
            _route(),
            query="extract financial KPIs",
            package_root=package_root,
        )

    assert marker.read_text() == "keep"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"execution_prompt": ""},
        {"execution_prompt": "Run it.", "workflow_plan": {}},
        ["not", "an", "object"],
    ],
)
def test_planner_output_schema_is_exact(payload) -> None:
    assert validate_planner_output(payload)


def test_planner_schema_contains_only_execution_prompt() -> None:
    schema = planner_output_json_schema()

    assert schema["required"] == ["execution_prompt"]
    assert set(schema["properties"]) == {"execution_prompt"}
    assert schema["additionalProperties"] is False


def test_route_loader_rejects_non_object_items() -> None:
    payload = _route().to_dict()
    payload["relation_evidence"] = ["not-an-object"]

    with pytest.raises(ValueError, match=r"relation_evidence\[0\] must be an object"):
        RouteResult.from_dict(payload)
