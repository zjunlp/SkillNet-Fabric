from __future__ import annotations

import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from skillfabric.compiled_graph.builder import BuildConfig, _BuildDependencies, build_graph
from skillfabric.indexing.embeddings import DisabledEmbeddingProvider
from tests.unit.fake_canonicalization import FixtureCanonicalizationProvider
from tests.unit.fixture_interfaces import FixtureInterfaceExtractor

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_SKILLS = ROOT / "fixtures" / "skills"
_FIXTURE_CACHE_DIR: TemporaryDirectory[str] | None = None
_FIXTURE_CACHE_WORKSPACE: Path | None = None


class FixtureExecutionValidator:
    model_id = "fixture-execution"

    def validate(self, candidate, source_skill, target_skill, *, interfaces):
        del source_skill, target_skill, interfaces
        accepted = bool(candidate.evidence)
        return {
            "accepted": accepted,
            "flow_type": candidate.flow_type if accepted else "none",
            "projected_edge_type": "depend_on" if accepted else "none",
            "confidence": 0.92 if accepted else 0.0,
            "evidence": [item.to_dict() for item in candidate.evidence],
        }


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
    _rewrite_status_artifact_paths(workspace)


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
            interface_extractor=FixtureInterfaceExtractor(),
            execution_validator=FixtureExecutionValidator(),
            canonicalization_provider=FixtureCanonicalizationProvider(),
            embedding_provider=DisabledEmbeddingProvider(),
            build_id="wiki-test-build",
        ),
    )


def _rewrite_status_artifact_paths(workspace: Path) -> None:
    status_path = workspace / "status.json"
    if not status_path.exists():
        return
    status = json.loads(status_path.read_text(encoding="utf-8"))
    artifacts = status.get("artifacts", {})
    if isinstance(artifacts, dict):
        status["artifacts"] = {
            key: str(workspace / Path(str(value)).relative_to(_FIXTURE_CACHE_WORKSPACE or workspace))
            if _FIXTURE_CACHE_WORKSPACE is not None and _path_is_relative_to(Path(str(value)), _FIXTURE_CACHE_WORKSPACE)
            else str(value)
            for key, value in artifacts.items()
        }
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
