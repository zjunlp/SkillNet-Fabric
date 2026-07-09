from __future__ import annotations

import contextlib
import io
import json
import unittest
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from skillfabric.cli import main as cli_main
from skillfabric.compiled_graph.models import GraphDocument
from skillfabric.registry.models import SkillNode
from skillfabric.storage import Workspace
from skillfabric.wiki.models import WikiBuildResult, WikiHealthReport

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_SKILLS = ROOT / "fixtures" / "skills"


@dataclass(slots=True)
class _FakeBuildResult:
    workspace: Workspace
    skills: list[SkillNode]
    graph: GraphDocument
    stats: dict[str, object]


def _fake_build_result(workspace_root: Path) -> _FakeBuildResult:
    workspace = Workspace(workspace_root)
    workspace.ensure()
    (workspace.graph_dir / "compiled.json").write_text("{}\n", encoding="utf-8")
    (workspace.graph_dir / "registry.jsonl").write_text("{}\n", encoding="utf-8")
    (workspace.cache_dir / "interface_cache.json").write_text("{}\n", encoding="utf-8")
    (workspace.reports_dir / "build_summary.json").write_text("{}\n", encoding="utf-8")
    skill = SkillNode(
        id="skill:test",
        type="skill",
        name="test",
        description="Test skill.",
        content_hash="hash-test",
    )
    graph = GraphDocument(
        schema_version="1.0",
        build_id="test-build",
        nodes=[skill],
        edges=[],
        stats={},
        config_digest="digest",
    )
    return _FakeBuildResult(
        workspace=workspace,
        skills=[skill],
        graph=graph,
        stats={"skill_count": 1, "skipped_unchanged": 0},
    )


def _write_env_file(path: Path) -> None:
    path.write_text(
        "API_KEY=sk-test\n"
        "BASE_URL=https://api.example.test/v1\n"
        "MODEL=openai/test-model\n"
        "EMBEDDING_MODEL=openai/text-embedding-3-small\n",
        encoding="utf-8",
    )


def _fake_wiki_result(workspace: Path) -> WikiBuildResult:
    return WikiBuildResult(
        pages_written=1,
        cache_hits=0,
        llm_calls=0,
        fallback_count=0,
        health=WikiHealthReport(),
        workspace=workspace,
    )


class BuildCliTests(unittest.TestCase):
    def test_build_cli_generates_graph_and_wiki_by_default(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            env_file = Path(tmp) / ".env"
            _write_env_file(env_file)

            output = io.StringIO()
            with patch("skillfabric.cli.build_graph", return_value=_fake_build_result(workspace)):
                with patch(
                    "skillfabric.cli.build_wiki",
                    return_value=_fake_wiki_result(workspace),
                ):
                    (workspace / "reports").mkdir(parents=True, exist_ok=True)
                    (workspace / "reports" / "wiki_health_report.md").write_text("# Health\n", encoding="utf-8")
                    (workspace / "wiki").mkdir(parents=True, exist_ok=True)
                    (workspace / "wiki" / "index.md").write_text("# Wiki\n", encoding="utf-8")
                    with contextlib.redirect_stdout(output):
                        cli_main(
                            [
                                "build",
                                "--skill-root",
                                str(FIXTURE_SKILLS),
                                "--workspace",
                                str(workspace),
                                "--env-file",
                                str(env_file),
                            ]
                        )
            payload = json.loads(output.getvalue())

            self.assertEqual(payload["workspace"], str(workspace))
            self.assertNotIn("profile", payload)
            self.assertEqual(payload["skill_count"], 1)
            self.assertNotIn("estimate", payload)
            self.assertNotIn("budget", payload)
            self.assertIn("cache", payload)
            self.assertTrue((workspace / "graph" / "compiled.json").exists())
            self.assertTrue((workspace / "graph" / "registry.jsonl").exists())
            self.assertTrue((workspace / "cache" / "interface_cache.json").exists())
            self.assertTrue((workspace / "reports" / "build_summary.json").exists())
            self.assertTrue((workspace / "reports" / "wiki_health_report.md").exists())
            self.assertTrue((workspace / "wiki" / "index.md").exists())
            self.assertFalse((workspace / "wiki" / "wiki_health_report.md").exists())
            self.assertIn("graph", payload["artifacts"])
            self.assertIn("wiki", payload["artifacts"])

    def test_build_cli_skip_wiki_generates_only_graph_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            env_file = Path(tmp) / ".env"
            _write_env_file(env_file)

            output = io.StringIO()
            with patch("skillfabric.cli.build_graph", return_value=_fake_build_result(workspace)):
                with contextlib.redirect_stdout(output):
                    cli_main(
                        [
                            "build",
                            "--skill-root",
                            str(FIXTURE_SKILLS),
                            "--workspace",
                            str(workspace),
                            "--env-file",
                            str(env_file),
                            "--skip-wiki",
                        ]
                    )
            payload = json.loads(output.getvalue())

            self.assertTrue((workspace / "graph" / "compiled.json").exists())
            self.assertFalse((workspace / "wiki" / "index.md").exists())
            self.assertNotIn("wiki", payload["artifacts"])


if __name__ == "__main__":
    unittest.main()
