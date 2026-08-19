from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import skillfabric.planner.package as package_module
from skillfabric.planner.package import (
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
from skillfabric.runtime.llm import LLMRequestError
from tests.support import build_fixture_workspace


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
                source_skill="skill:pdf-table-parser",
                target_skill="skill:financial-kpi-extractor",
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
                "Produce `result.json` from the PDF report as a JSON array whose objects contain "
                "exactly `name` and `value`.\n\n"
                "1. Apply `skill:pdf-table-parser` to recover the report's table structure while "
                "preserving the source values.\n"
                "2. Pass the recovered rows to `skill:financial-kpi-extractor` and retain only the "
                "requested KPI names and values.\n"
                "3. Write `result.json`, reload it, and verify that every object has exactly the "
                "required keys and every value agrees with the PDF."
            )
        }
    )


def _planner_contract_prompt() -> str:
    messages = package_module._planner_messages(
        query="Create the requested deliverables.",
        route=_route(),
        contexts=[],
    )
    prompt = "\n".join(str(message["content"]) for message in messages)
    return " ".join(prompt.split())


def test_planner_makes_task_outcomes_authoritative_and_assigns_skill_roles() -> None:
    prompt = _planner_contract_prompt()

    assert "quality of the final deliverables" in prompt
    assert "The original task defines success" in prompt
    assert "evidence-backed methods and constraints" in prompt
    assert "exact `skill_id`" in prompt
    assert "materially supports the workflow" in prompt
    assert "Keep overlapping capabilities coherent" in prompt
    assert "without reducing task quality" in prompt
    assert "Do not repeat Skill source text" in prompt
    assert "Use every selected Skill" not in prompt
    assert "Do not omit, reject, or replace" not in prompt


def test_planner_preserves_open_requirements_and_handles_missing_information() -> None:
    prompt = _planner_contract_prompt()

    assert "deliverables, filenames, paths, formats, quantities, constraints" in prompt
    assert "reasonable, conservative, internally consistent assumptions" in prompt
    assert "Use placeholders only when" in prompt
    assert "does not mean the executor lacks the ability" in prompt


def test_planner_builds_a_short_end_to_end_handoff_with_targeted_checks() -> None:
    prompt = _planner_contract_prompt()

    assert "shortest complete end-to-end workflow" in prompt
    assert "task-critical checks" in prompt
    assert "Inspect or execute the actual final output" in prompt
    assert "normally 150-350 words" in prompt
    assert "up to 500 words" in prompt
    assert "three to six ordered steps" in prompt
    assert "normally 300-700 words" not in prompt
    assert "Blueprint, Production, and Inspection and Repair" not in prompt
    assert "inspection-and-repair cycle" not in prompt
    assert "Prefer 800-1600 words" not in prompt


def test_planner_uses_one_source_grounded_primary_path() -> None:
    prompt = _planner_contract_prompt()

    assert "one primary execution path" in prompt
    assert "traceable to the original task or a selected Skill's canonical source" in prompt
    assert "Do not invent thresholds, algorithms, libraries, commands, parameters" in prompt
    assert "Prefer the simplest method that fully satisfies the task" in prompt
    assert "Do not present alternatives" in prompt


def test_planner_uses_skills_concisely_without_repeating_sources() -> None:
    prompt = _planner_contract_prompt()

    assert "one clear role" in prompt
    assert "identify it by its exact `skill_id` when first introduced" in prompt
    assert "refer back concisely if the same guidance is needed later" in prompt
    assert "Do not add a separate inventory of selected Skills" in prompt
    assert "Do not restate Skill instructions" in prompt


def test_planner_contract_contains_a_short_positive_example() -> None:
    prompt = _planner_contract_prompt()

    assert "<example>" in prompt
    assert "<example_input>" in prompt
    assert "<example_execution_prompt>" in prompt
    assert "The route selected `skill:domain-parser-example` for parsing" in prompt
    assert "Produce `output.json` from `input.dat`" in prompt
    assert "This example demonstrates shape, not a mandatory template" in prompt


def test_planner_does_not_promote_method_dependencies_to_task_dependencies() -> None:
    prompt = _planner_contract_prompt()

    assert "not automatically a task dependency" in prompt
    assert "concrete blocker is observed" in prompt
    assert "preserve the requested deliverables and constraints" in prompt
    assert "plan an explicit failure" not in prompt
    assert "OPENROUTER" not in prompt
    assert "SkillsBench" not in prompt
    assert "SkillRouter" not in prompt
    assert "golden" not in prompt.lower()
    assert "visual_creation_task" not in prompt


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

    query = (
        "Extract financial KPIs from a PDF report. Write result.json with each object using "
        "the exact keys 'name' and 'value'."
    )
    result = plan_execution_package(
        workspace,
        _route(),
        query=query,
        env_file=tmp_path / "unused.env",
        package_root=package_root,
        planner_context_max_tokens=10_000,
    )

    assert len(calls) == 1
    assert PLANNER_PROMPT_ID == "skillfabric_execution_planner_task_grounded_handoff"
    messages = calls[0]["messages"]
    prompt = " ".join(
        "\n".join(str(item["content"]) for item in messages).split()  # type: ignore[index]
    )
    assert "skill:pdf-table-parser" in prompt
    assert "skill:financial-kpi-extractor" in prompt
    assert "&lt;untrusted_skill_source" in prompt
    assert "&quot;" not in prompt
    assert "relation_evidence" in prompt
    assert "source before target" in prompt
    assert "producer-to-consumer handoff" in prompt
    assert "compact, task specific execution prompt" in prompt
    assert "materially supports the workflow" in prompt
    assert "shortest complete end-to-end workflow" in prompt
    assert "task specific final checks" in prompt
    assert counted_models == ["openai/test-model"]
    assert result.estimated_prompt_tokens == 1200
    execution_prompt = result.prompt_path.read_text()
    assert execution_prompt.startswith(f"<original_task>\n{query}\n</original_task>")
    assert f"<original_task>\n{query}\n</original_task>" in execution_prompt
    assert "<execution_plan>\nProduce `result.json`" in execution_prompt
    assert execution_prompt.index("<original_task>") < execution_prompt.index("<execution_plan>")
    assert "<execution_contract>" not in execution_prompt
    assert execution_prompt.count(query) == 1
    assert execution_prompt.count("`skill:pdf-table-parser`") == 1
    assert execution_prompt.count("`skill:financial-kpi-extractor`") == 1
    assert all(f"\n{step}. " in execution_prompt for step in range(1, 4))
    assert "\n4. " not in execution_prompt
    assert "independent checks in parallel" not in execution_prompt
    assert not (package_root / "workflow_plan.json").exists()
    assert not (package_root / "PLANNER.md").exists()
    assert json.loads(result.planner_output_path.read_text()) == json.loads(_planner_response())
    request = json.loads((package_root / "planner_request.json").read_text())
    assert request["prompt_id"] == PLANNER_PROMPT_ID
    assert request["expected_schema"] == planner_output_json_schema()
    assert request["estimated_prompt_tokens"] == 1200


def test_plan_forwards_explicit_timeout_to_llm_config(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / ".skillfabric"
    build_fixture_workspace(workspace)
    config_calls: list[dict[str, object]] = []

    def from_env(**options):
        config_calls.append(options)
        return package_module.LLMConfig(
            api_base="https://example.test/v1",
            api_key="test-key",
            model="openai/test-model",
            timeout=options["timeout"],
        )

    monkeypatch.setattr(package_module.LLMConfig, "from_env", from_env)
    monkeypatch.setattr(package_module, "litellm_completion", lambda **_kwargs: _planner_response())

    plan_execution_package(
        workspace,
        _route(),
        query="extract financial KPIs",
        package_root=workspace / "runs" / "timeout-test" / "execution_package",
        llm_timeout_seconds=0,
    )

    assert config_calls[0]["timeout"] == 0


def test_plan_uses_explicit_runtime_identity_instead_of_env_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / ".skillfabric"
    build_fixture_workspace(workspace)
    captured: dict[str, object] = {}
    config = package_module.LLMConfig(
        api_base="https://example.test/v1",
        api_key="test-key",
        model="openai/responses/gpt-5.5",
        reasoning_effort="xhigh",
    )

    def from_env(**kwargs):
        captured.update(kwargs)
        return config

    monkeypatch.setattr(package_module.LLMConfig, "from_env", from_env)
    monkeypatch.setattr(package_module, "litellm_completion", lambda **_kwargs: _planner_response())

    env_file = tmp_path / "runtime.env"
    plan_execution_package(
        workspace,
        _route(),
        query="Extract financial KPIs from a PDF report.",
        env_file=env_file,
        package_root=workspace / "runs" / "explicit-runtime" / "execution_package",
        llm_model="gpt-5.5",
        llm_reasoning_effort="xhigh",
        llm_api_key="skillsbench-key",
        llm_api_base="https://skillsbench.example/v1",
    )

    assert captured == {
        "env_path": env_file,
        "model": "gpt-5.5",
        "reasoning_effort": "xhigh",
        "api_key": "skillsbench-key",
        "api_base": "https://skillsbench.example/v1",
    }


def test_message_token_count_has_a_local_fallback(monkeypatch) -> None:
    from skillfabric.runtime.tokens import count_message_tokens

    monkeypatch.setitem(sys.modules, "litellm", None)

    assert count_message_tokens([{"role": "user", "content": "Plan this task."}]) > 0


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


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"planner_max_attempts": 0}, "planner_max_attempts"),
        ({"planner_retry_delay_seconds": -1}, "planner_retry_delay_seconds"),
        ({"planner_retry_delay_seconds": float("nan")}, "planner_retry_delay_seconds"),
    ],
)
def test_plan_rejects_invalid_retry_limits(tmp_path, options, message) -> None:
    with pytest.raises(ValueError, match=message):
        plan_execution_package(
            tmp_path / ".skillfabric",
            _route(),
            query="extract financial KPIs",
            **options,
        )


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
            planner_max_attempts=1,
        )

    assert not (package_root / "execution_prompt.md").exists()
    validation = json.loads((package_root / "planner_validation.json").read_text())
    assert validation["valid"] is False
    assert validation["errors"] == ["planner output must contain exactly execution_prompt"]


def test_plan_retries_invalid_output_without_rebuilding_context(
    tmp_path,
    monkeypatch,
    caplog,
) -> None:
    workspace = tmp_path / ".skillfabric"
    build_fixture_workspace(workspace)
    token_counts = 0
    responses = [json.dumps({"execution_prompt": ""}), _planner_response()]

    def count_tokens(*_args, **_kwargs):
        nonlocal token_counts
        token_counts += 1
        return 100

    monkeypatch.setattr(package_module, "count_message_tokens", count_tokens)
    monkeypatch.setattr(
        package_module,
        "litellm_completion",
        lambda **_kwargs: responses.pop(0),
    )

    result = plan_execution_package(
        workspace,
        _route(),
        query="extract financial KPIs",
        package_root=workspace / "runs" / "retry" / "execution_package",
        planner_max_attempts=2,
        planner_retry_delay_seconds=0,
    )

    assert token_counts == 1
    assert not responses
    assert result.prompt_path.is_file()
    assert json.loads(result.planner_validation_path.read_text()) == {"valid": True, "errors": []}
    assert "planner_retry attempt=1/2" in caplog.text


def test_plan_does_not_repeat_a_provider_request_failure(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / ".skillfabric"
    build_fixture_workspace(workspace)
    calls = 0

    def failed_completion(**_kwargs):
        nonlocal calls
        calls += 1
        raise LLMRequestError("provider retries exhausted")

    monkeypatch.setattr(package_module, "count_message_tokens", lambda *_args, **_kwargs: 100)
    monkeypatch.setattr(package_module, "litellm_completion", failed_completion)
    package_root = workspace / "runs" / "provider-failure" / "execution_package"

    with pytest.raises(LLMRequestError, match="provider retries exhausted"):
        plan_execution_package(
            workspace,
            _route(),
            query="extract financial KPIs",
            package_root=package_root,
            planner_max_attempts=2,
            planner_retry_delay_seconds=0,
        )

    assert calls == 1
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


def test_plan_allows_planner_to_use_the_relevant_skill_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / ".skillfabric"
    build_fixture_workspace(workspace)
    monkeypatch.setattr(
        package_module,
        "litellm_completion",
        lambda **_kwargs: json.dumps(
            {
                "execution_prompt": (
                    "Use `skill:pdf-table-parser` to extract the requested table and "
                    "check the resulting values against the source."
                )
            }
        ),
    )

    result = plan_execution_package(
        workspace,
        _route(),
        query="Extract the requested table.",
        package_root=workspace / "runs" / "relevant-subset" / "execution_package",
        planner_max_attempts=1,
    )

    assert result.prompt_path.is_file()


def test_route_loader_rejects_non_object_items() -> None:
    payload = _route().to_dict()
    payload["relation_evidence"] = ["not-an-object"]

    with pytest.raises(ValueError, match=r"relation_evidence\[0\] must be an object"):
        RouteResult.from_dict(payload)
