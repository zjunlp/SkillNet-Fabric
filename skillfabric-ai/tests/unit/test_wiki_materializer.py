from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from skillfabric.wiki.explorer.search_index import load_page_index
from skillfabric.wiki.materializer import build_wiki
from skillfabric.wiki.models import WikiBuildConfig
from skillfabric.wiki.renderers import _first_paragraph
from tests.unit.wiki_helpers import build_fixture_workspace


class WikiMaterializerTests(unittest.TestCase):
    def test_build_wiki_generates_pages_with_links_and_sections(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)

            result = build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))

            self.assertGreater(result.pages_written, 0)
            self.assertEqual(result.pages_written, len(list((workspace / "wiki").rglob("*.md"))))
            skill_page = workspace / "wiki" / "skills" / "cards" / "pdf-table-parser.md"
            self.assertTrue(skill_page.exists())
            text = skill_page.read_text(encoding="utf-8")
            self.assertIn("type: Skill Card", text)
            self.assertIn("title: pdf-table-parser", text)
            self.assertIn("skill_id: skill:pdf-table-parser", text)
            self.assertIn("source: ../sources/pdf-table-parser.md", text)
            self.assertIn("# Skill Card", text)
            self.assertIn("## Purpose", text)
            self.assertIn("## Use When", text)
            self.assertIn("## Do Not Use When", text)
            self.assertIn("## Inputs", text)
            self.assertIn("## Outputs", text)
            self.assertIn("## Tools And Dependencies", text)
            self.assertIn("## Composition Notes", text)
            self.assertIn("## Read Full Source", text)
            self.assertNotIn("## Evidence", text)
            self.assertNotIn("## Source", text)
            self.assertNotIn("content_hash", text)
            self.assertNotIn("skillfabric", text.lower())
            self.assertNotIn("/Users/", text)
            self.assertNotIn("[[artifacts/", text)
            self.assertNotIn("[[scenarios/", text)
            self.assertNotIn("raw_output", text)
            self.assertNotIn("\n```markdown\n", text)
            source_page = workspace / "wiki" / "skills" / "sources" / "pdf-table-parser.md"
            self.assertTrue(source_page.exists())
            source_text = source_page.read_text(encoding="utf-8")
            self.assertIn("type: Skill Source", source_text)
            self.assertIn("card: ../cards/pdf-table-parser.md", source_text)
            self.assertIn("# Full SKILL.md", source_text)
            self.assertIn("Extract tables", source_text)
            self.assertNotIn("/Users/", source_text)
            root_index_text = (workspace / "wiki" / "index.md").read_text(encoding="utf-8")
            self.assertIn("## Skill Cards", root_index_text)
            self.assertIn("[pdf-table-parser](skills/cards/pdf-table-parser.md)", root_index_text)
            self.assertIn("[full SKILL.md](skills/sources/pdf-table-parser.md)", root_index_text)
            self.assertIn("## Full Skill Sources", root_index_text)
            self.assertNotIn("## Communities", root_index_text)
            self.assertNotIn("- communities:", root_index_text)
            self.assertNotIn("title: pdf-table-parser", root_index_text)
            self.assertNotIn("Source Source", root_index_text)
            self.assertNotIn(": Skill: pdf-table-parser Source:", root_index_text)
            self.assertIn("Extract tables from PDF files", root_index_text)
            self.assertFalse((workspace / "wiki" / "communities").exists())
            self.assertFalse((workspace / "wiki" / "skills" / "index.md").exists())
            self.assertFalse((workspace / "wiki" / "communities" / "index.md").exists())
            self.assertFalse((workspace / "wiki" / "workflows" / "index.md").exists())
            self.assertFalse((workspace / "wiki" / "references" / "index.md").exists())
            self.assertFalse((workspace / "wiki" / "skills" / "sources" / "index.md").exists())
            self.assertFalse((workspace / "wiki" / "overview.md").exists())
            self.assertFalse((workspace / "wiki" / "resolver.md").exists())
            self.assertFalse((workspace / "wiki" / "deliverables.md").exists())
            self.assertFalse((workspace / "wiki" / "artifacts").exists())
            self.assertFalse((workspace / "wiki" / "scenarios").exists())
            self.assertTrue((workspace / "wiki" / "workflows").exists())

    def test_build_wiki_removes_stale_flat_skill_pages(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            stale_card = workspace / "wiki" / "skills" / "pdf-table-parser.md"
            stale_card.parent.mkdir(parents=True, exist_ok=True)
            stale_card.write_text("# stale\n", encoding="utf-8")

            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))

            self.assertFalse(stale_card.exists())
            self.assertTrue((workspace / "wiki" / "skills" / "cards" / "pdf-table-parser.md").exists())

    def test_build_wiki_can_emit_debug_extraction_pages_when_requested(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)

            build_wiki(
                WikiBuildConfig(
                    workspace=workspace,
                    use_llm_summaries=False,
                    include_debug_pages=True,
                )
            )

            self.assertFalse((workspace / "wiki" / "debug").exists())
            self.assertTrue(list((workspace / "reports" / "wiki-debug" / "raw_artifacts").glob("*.md")))
            self.assertTrue((workspace / "reports" / "wiki-debug" / "extraction_report.md").exists())

    def test_build_wiki_removes_stale_artifact_and_scenario_directories(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            stale_artifact = workspace / "wiki" / "artifacts" / "old.md"
            stale_scenario = workspace / "wiki" / "scenarios" / "old.md"
            stale_artifact.parent.mkdir(parents=True, exist_ok=True)
            stale_scenario.parent.mkdir(parents=True, exist_ok=True)
            stale_artifact.write_text("# old\n", encoding="utf-8")
            stale_scenario.write_text("# old\n", encoding="utf-8")

            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))

            self.assertFalse(stale_artifact.parent.exists())
            self.assertFalse(stale_scenario.parent.exists())

    def test_build_wiki_removes_stale_hot_page_and_excludes_it_from_index(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            stale_hot = workspace / "wiki" / "hot.md"
            stale_hot.parent.mkdir(parents=True, exist_ok=True)
            stale_hot.write_text("# Legacy Hot Page\n", encoding="utf-8")

            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))

            self.assertFalse(stale_hot.exists())
            self.assertFalse(any(page.path == "hot.md" for page in load_page_index(workspace)))

    def test_build_wiki_removes_stale_resolver_page_and_excludes_it_from_index(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            stale_resolver = workspace / "wiki" / "resolver.md"
            stale_resolver.parent.mkdir(parents=True, exist_ok=True)
            stale_resolver.write_text("# Legacy Resolver\n", encoding="utf-8")

            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))

            self.assertFalse(stale_resolver.exists())
            self.assertFalse(any(page.path == "resolver.md" for page in load_page_index(workspace)))

    def test_source_excerpt_uses_fence_that_survives_nested_code_blocks(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            skills_path = workspace / "graph" / "registry.jsonl"
            rows = skills_path.read_text(encoding="utf-8").splitlines()
            patched = []
            for row in rows:
                if '"id": "skill:pdf-table-parser"' in row:
                    patched.append(row.replace("Validate column headers.", "Validate column headers.\\n```python\\nprint('x')\\n```"))
                else:
                    patched.append(row)
            skills_path.write_text("\n".join(patched) + "\n", encoding="utf-8")

            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))

            text = (workspace / "wiki" / "skills" / "cards" / "pdf-table-parser.md").read_text(encoding="utf-8")
            self.assertNotIn("````markdown", text)
            source_text = (
                workspace / "wiki" / "skills" / "sources" / "pdf-table-parser.md"
            ).read_text(encoding="utf-8")
            self.assertIn("```python", source_text)

    def test_first_paragraph_truncates_on_word_boundary(self) -> None:
        summary = _first_paragraph("routing " * 80)

        self.assertLessEqual(len(summary), 240)
        self.assertTrue(summary.endswith("..."))
        self.assertNotIn(" ...", summary)
        self.assertTrue(summary.removesuffix("...").endswith("routing"))


if __name__ == "__main__":
    unittest.main()
