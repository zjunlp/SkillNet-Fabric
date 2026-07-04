from __future__ import annotations

import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from skillfabric.cli import main as cli_main

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_SKILLS = ROOT / "fixtures" / "skills"


class BuildCliTests(unittest.TestCase):
    def test_build_cli_generates_graph_and_wiki_by_default(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                cli_main(
                    [
                        "build",
                        "--skill-root",
                        str(FIXTURE_SKILLS),
                        "--workspace",
                        str(workspace),
                        "--skip-llm-validation",
                        "--embedding-provider",
                        "disabled",
                    ]
                )
            payload = json.loads(output.getvalue())

            self.assertEqual(payload["workspace"], str(workspace))
            self.assertNotIn("profile", payload)
            self.assertGreater(payload["skill_count"], 0)
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

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                cli_main(
                    [
                        "build",
                        "--skill-root",
                        str(FIXTURE_SKILLS),
                        "--workspace",
                        str(workspace),
                        "--skip-llm-validation",
                        "--embedding-provider",
                        "disabled",
                        "--skip-wiki",
                    ]
                )
            payload = json.loads(output.getvalue())

            self.assertTrue((workspace / "graph" / "compiled.json").exists())
            self.assertFalse((workspace / "wiki" / "index.md").exists())
            self.assertNotIn("wiki", payload["artifacts"])


if __name__ == "__main__":
    unittest.main()
