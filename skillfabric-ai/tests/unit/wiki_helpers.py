from __future__ import annotations

import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from skillfabric.compiled_graph.builder import BuildConfig, build_graph
from skillfabric.compiled_graph.canonicalization.compiler import (
    DeterministicCanonicalizationProvider,
)
from skillfabric.compiled_graph.execution.validation import DeterministicExecutionFlowValidator
from skillfabric.compiled_graph.interface.extraction import DeterministicInterfaceExtractor
from skillfabric.compiled_graph.relations.validation import StaticPairValidator
from tests.unit.fake_embeddings import FakeEmbeddingProvider

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
    validator = StaticPairValidator(
        {
            ("skill:financial-kpi-extractor", "skill:pdf-table-parser"): {
                "edge_type": "depend_on",
                "direction": "A->B",
                "confidence": 0.92,
                "evidence": [
                    {
                        "skill": "skill:financial-kpi-extractor",
                        "line": 7,
                        "text": "Use this after `pdf-table-parser` has produced `.csv` tables.",
                    }
                ],
                "reason": "KPI extraction consumes CSV tables produced by PDF table parsing.",
            },
            ("skill:report-writer", "skill:financial-kpi-extractor"): {
                "edge_type": "depend_on",
                "direction": "A->B",
                "confidence": 0.9,
                "evidence": [
                    {
                        "skill": "skill:report-writer",
                        "line": 5,
                        "text": "Use KPI JSON and chart artifacts to compose a final `.md` report.",
                    }
                ],
                "reason": "Report writing consumes KPI JSON.",
            },
            ("skill:testing-python", "skill:analyze-ci"): {
                "edge_type": "compose_with",
                "direction": "undirected",
                "confidence": 0.82,
                "evidence": [
                    {
                        "skill": "skill:testing-python",
                        "line": 6,
                        "text": "This skill composes with `analyze-ci` when a CI job fails.",
                    }
                ],
                "reason": "CI analysis and focused pytest diagnosis are commonly chained.",
            },
        }
    )
    build_graph(
        BuildConfig(
            skill_root=FIXTURE_SKILLS,
            workspace=workspace,
            similar_top_k=3,
            candidate_top_k=6,
            validator=validator,
            interface_extractor=DeterministicInterfaceExtractor(),
            execution_validator=DeterministicExecutionFlowValidator(),
            canonicalization_provider=DeterministicCanonicalizationProvider(),
            embedding_provider=FakeEmbeddingProvider(),
            build_id="wiki-test-build",
        )
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
