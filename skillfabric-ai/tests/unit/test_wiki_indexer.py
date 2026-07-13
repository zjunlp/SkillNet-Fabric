from __future__ import annotations

import inspect
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from skillfabric.wiki.indexer import _clean_summary, render_index
from skillfabric.wiki.materializer import build_wiki
from skillfabric.wiki.models import WikiBuildConfig
from tests.unit.wiki_helpers import build_fixture_workspace


class WikiIndexerTests(unittest.TestCase):
    def test_render_index_has_no_unused_section_parameter(self) -> None:
        self.assertNotIn("section", inspect.signature(render_index).parameters)

    def test_build_wiki_generates_index_and_append_only_log_without_dead_page_index(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)

            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))
            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))

            index_text = (workspace / "wiki" / "index.md").read_text(encoding="utf-8")
            log_text = (workspace / "reports" / "wiki_log.md").read_text(encoding="utf-8")
            self.assertIn("# SkillFabric Wiki", index_text)
            self.assertIn("## Corpus", index_text)
            self.assertIn("## Skill Cards", index_text)
            self.assertIn("[pdf-table-parser](skills/cards/pdf-table-parser.md)", index_text)
            self.assertIn("  summary: Extract tables from PDF files", index_text)
            self.assertIn(
                "  source: [full SKILL.md](skills/sources/pdf-table-parser.md)", index_text
            )
            self.assertIn("[full SKILL.md](skills/sources/pdf-table-parser.md)", index_text)
            self.assertIn("## Full Skill Sources", index_text)
            self.assertNotIn(" Source: [full SKILL.md]", index_text)
            self.assertFalse((workspace / "wiki" / "hot.md").exists())
            self.assertFalse((workspace / "wiki" / "log.md").exists())
            self.assertFalse((workspace / "wiki" / "wiki_page_index.jsonl").exists())
            self.assertFalse((workspace / "graph" / "wiki_page_index.jsonl").exists())
            self.assertGreaterEqual(log_text.count("wiki-build | build_id="), 2)
            self.assertNotIn("fallback_count", log_text)

    def test_clean_summary_truncates_on_word_boundary(self) -> None:
        text = " ".join(["routing"] * 80)

        summary = _clean_summary(text)

        self.assertLessEqual(len(summary), 240)
        self.assertTrue(summary.endswith("..."))
        self.assertNotIn(" ...", summary)


if __name__ == "__main__":
    unittest.main()
