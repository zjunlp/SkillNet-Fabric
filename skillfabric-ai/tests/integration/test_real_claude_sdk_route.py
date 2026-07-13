from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from skillfabric.compiled_graph.builder import BuildConfig, _BuildDependencies, build_graph
from skillfabric.router.config import RouterConfig
from skillfabric.router.routing import route_task
from skillfabric.wiki.explorer.prompting import EXPLORER_PROMPT_ID
from tests.unit.fake_embeddings import FakeEmbeddingProvider
from tests.unit.semantic_fixtures import FixtureContractExtractor, FixtureRelationJudge

FIXTURE_SKILLS = Path(__file__).resolve().parents[1] / "fixtures" / "skills"


@pytest.mark.skipif(
    os.environ.get("SKILLFABRIC_REAL_CC_SDK") != "1",
    reason="set SKILLFABRIC_REAL_CC_SDK=1 to run the real Claude Agent SDK smoke test",
)
def test_real_claude_sdk_routes_from_semantic_query_wiki(tmp_path) -> None:
    env_file_value = os.environ.get("SKILLFABRIC_REAL_ENV_FILE", "")
    if not env_file_value:
        pytest.skip("set SKILLFABRIC_REAL_ENV_FILE to a configured env-file path")
    env_file = Path(env_file_value)
    if not env_file.is_file():
        pytest.skip("SKILLFABRIC_REAL_ENV_FILE does not exist")
    workspace = tmp_path / ".skillfabric"
    build_graph(
        BuildConfig(skill_root=FIXTURE_SKILLS, workspace=workspace),
        dependencies=_BuildDependencies(
            contract_extractor=FixtureContractExtractor(),
            relation_judge=FixtureRelationJudge(),
            embedding_provider=FakeEmbeddingProvider(),
            build_id="real-sdk-route-fixture",
        ),
    )

    result = route_task(
        RouterConfig(
            workspace=workspace,
            query="extract financial KPIs from a PDF report",
            env_file=env_file,
            trace_id="real-sdk-route",
            explorer_model=os.environ.get("ANTHROPIC_MODEL") or None,
            max_selected_skills=4,
        ),
        embedding_provider=FakeEmbeddingProvider(),
    )

    assert "skill:pdf-table-parser" in result.selected_skill_ids
    assert "skill:financial-kpi-extractor" in result.selected_skill_ids
    assert any(
        relation.relation_type == "depend_on"
        and relation.source_skill == "skill:financial-kpi-extractor"
        and relation.target_skill == "skill:pdf-table-parser"
        for relation in result.relation_evidence
    )
    trace = workspace / "runs" / "real-sdk-route"
    prompt_contract = json.loads(
        (trace / "cc_explorer" / "prompt_contract.json").read_text(encoding="utf-8")
    )
    assert prompt_contract["prompt_id"] == EXPLORER_PROMPT_ID
    assert (trace / "route.json").exists()
