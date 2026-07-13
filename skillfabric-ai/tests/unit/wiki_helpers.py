from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from skillfabric.compiled_graph.builder import BuildConfig, _BuildDependencies, build_graph
from tests.unit.fake_embeddings import FakeEmbeddingProvider
from tests.unit.semantic_fixtures import FixtureContractExtractor, FixtureRelationJudge

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_SKILLS = ROOT / "fixtures" / "skills"
_FIXTURE_CACHE_DIR: TemporaryDirectory[str] | None = None
_FIXTURE_CACHE_WORKSPACE: Path | None = None


def build_fixture_workspace(workspace: Path) -> None:
    """Copy a deterministic fixture KG workspace into ``workspace``.

    Building the fixture graph dominates route/wiki unit test runtime. The
    cached source is process-local so each fresh unittest run still rebuilds
    the fixture once, while each test receives an isolated copy it can mutate.
    """

    source = _cached_fixture_workspace()
    if workspace.exists():
        shutil.rmtree(workspace)
    shutil.copytree(source, workspace)


def _cached_fixture_workspace() -> Path:
    global _FIXTURE_CACHE_DIR, _FIXTURE_CACHE_WORKSPACE
    if _FIXTURE_CACHE_WORKSPACE is not None:
        return _FIXTURE_CACHE_WORKSPACE
    _FIXTURE_CACHE_DIR = TemporaryDirectory(prefix="skillfabric-fixture-")
    workspace = Path(_FIXTURE_CACHE_DIR.name) / ".skillfabric"
    _build_fixture_workspace_uncached(workspace)
    _FIXTURE_CACHE_WORKSPACE = workspace
    return workspace


def _build_fixture_workspace_uncached(workspace: Path) -> None:
    build_graph(
        BuildConfig(
            skill_root=FIXTURE_SKILLS,
            workspace=workspace,
        ),
        dependencies=_BuildDependencies(
            contract_extractor=FixtureContractExtractor(),
            relation_judge=FixtureRelationJudge(),
            embedding_provider=FakeEmbeddingProvider(),
            build_id="wiki-test-build",
        ),
    )
