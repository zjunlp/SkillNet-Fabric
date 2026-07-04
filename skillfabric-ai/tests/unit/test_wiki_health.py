from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from skillfabric.storage import Workspace
from skillfabric.wiki.health import analyze_wiki_health
from skillfabric.wiki.materializer import build_wiki
from skillfabric.wiki.models import WikiBuildConfig
from tests.unit.wiki_helpers import build_fixture_workspace


class WikiHealthTests(unittest.TestCase):
    def test_health_detects_broken_link_and_raw_output_leak(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))

            page = workspace / "wiki" / "skills" / "pdf-table-parser.md"
            page.write_text(
                page.read_text(encoding="utf-8") + "\n[[skills/missing-skill]]\nraw_output should not be present\n",
                encoding="utf-8",
            )

            report = analyze_wiki_health(Workspace(workspace), fallback_count=0)

            self.assertTrue(report.broken_links)
            self.assertTrue(report.raw_llm_output_leaks)
            self.assertGreaterEqual(report.summary["broken_link_count"], 1)

    def test_health_ignores_source_excerpt_and_fenced_code_wikilinks(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))

            page = workspace / "wiki" / "skills" / "pdf-table-parser.md"
            page.write_text(
                page.read_text(encoding="utf-8") + "\n```markdown\n[[skills/not-a-generated-link]]\n```\n",
                encoding="utf-8",
            )
            source_page = workspace / "wiki" / "skills" / "source" / "pdf-table-parser.md"
            source_page.write_text(
                source_page.read_text(encoding="utf-8") + "\n[[skills/source-example-link]]\n",
                encoding="utf-8",
            )

            report = analyze_wiki_health(Workspace(workspace), fallback_count=0)

            self.assertFalse(report.broken_links)


if __name__ == "__main__":
    unittest.main()
