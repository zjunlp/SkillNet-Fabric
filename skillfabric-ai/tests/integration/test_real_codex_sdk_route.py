from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from skillfabric.compiled_graph.builder import BuildConfig, _BuildDependencies, build_graph
from skillfabric.router.config import RouterConfig
from skillfabric.router.routing import route_task
from skillfabric.wiki.explorer.backends.codex import (
    CODEX_EXECUTION_CONTRACT,
    CodexWikiExplorerBackend,
)
from tests.unit.fake_embeddings import FakeEmbeddingProvider
from tests.unit.semantic_fixtures import FixtureContractExtractor, FixtureRelationJudge

FIXTURE_SKILLS = Path(__file__).resolve().parents[1] / "fixtures" / "skills"


@pytest.mark.skipif(
    os.environ.get("SKILLFABRIC_REAL_CODEX_SDK") != "1",
    reason="set SKILLFABRIC_REAL_CODEX_SDK=1 to run the real Codex SDK smoke test",
)
def test_real_codex_sdk_routes_from_semantic_query_wiki(tmp_path: Path) -> None:
    env_file_value = os.environ.get("SKILLFABRIC_REAL_ENV_FILE", "")
    model = os.environ.get("SKILLFABRIC_REAL_CODEX_MODEL", "")
    if not env_file_value or not model:
        pytest.skip(
            "set SKILLFABRIC_REAL_ENV_FILE and SKILLFABRIC_REAL_CODEX_MODEL for the smoke test"
        )
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
            build_id="real-codex-sdk-route-fixture",
        ),
    )
    backend = CodexWikiExplorerBackend(
        env_file=env_file,
        max_selected_skills=4,
        model=model,
        reasoning_effort="medium",
        execution_timeout_seconds=300,
        execution_contract=CODEX_EXECUTION_CONTRACT.to_dict(),
    )

    result = route_task(
        RouterConfig(
            workspace=workspace,
            query="extract financial KPIs from a PDF report",
            env_file=env_file,
            trace_id="real-codex-sdk-route",
            max_selected_skills=4,
            explorer_max_attempts=1,
        ),
        explorer_backend=backend,
        embedding_provider=FakeEmbeddingProvider(),
    )

    assert len(result.selected_skill_ids) <= 4
    trace = workspace / "runs" / "real-codex-sdk-route"
    backend_artifact = json.loads(
        (trace / "cc_explorer" / "backend.json").read_text(encoding="utf-8")
    )
    assert backend_artifact["backend"] == "codex"
    assert backend_artifact["model"] == model
    assert (trace / "route.json").exists()
