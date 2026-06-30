from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from skillfabric.wiki.explorer.search_index import load_page_index
from skillfabric.wiki.materializer import build_wiki
from skillfabric.wiki.models import WikiBuildConfig
from tests.unit.wiki_helpers import build_fixture_workspace


class WikiIndexerTests(unittest.TestCase):
    def test_build_wiki_generates_index_page_index_and_append_only_log(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)

            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))
            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))

            index_text = (workspace / "wiki" / "index.md").read_text(encoding="utf-8")
            log_text = (workspace / "wiki" / "log.md").read_text(encoding="utf-8")
            pages = load_page_index(workspace)
            self.assertIn("# SkillFabric Wiki", index_text)
            self.assertIn("## Counts", index_text)
            self.assertIn("## Skills by Community", index_text)
            self.assertTrue(pages)
            self.assertTrue(any(page.page_type == "workflow" for page in pages))
            self.assertFalse((workspace / "wiki" / "hot.md").exists())
            self.assertGreaterEqual(log_text.count("wiki-build | build_id="), 2)


if __name__ == "__main__":
    unittest.main()
