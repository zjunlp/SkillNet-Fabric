from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from skillfabric.storage import atomic_write_text
from skillfabric.wiki.materializer import build_wiki
from skillfabric.wiki.models import WikiBuildConfig
from skillfabric.wiki.renderers import _first_paragraph
from tests.unit.wiki_helpers import build_fixture_workspace


class WikiMaterializerTests(unittest.TestCase):
    def test_build_wiki_preserves_existing_pages_until_source_and_summaries_succeed(self) -> None:
        failures = (
            ("skillfabric.wiki.materializer.load_wiki_source", ValueError("invalid graph")),
            (
                "skillfabric.wiki.materializer.WikiSummarizer.summarize_many",
                RuntimeError("summary failed"),
            ),
        )

        for target, error in failures:
            with self.subTest(target=target), TemporaryDirectory() as tmp:
                workspace = Path(tmp) / ".skillfabric"
                build_fixture_workspace(workspace)
                existing = workspace / "wiki" / "hot.md"
                existing.parent.mkdir(parents=True, exist_ok=True)
                existing.write_text("# Existing wiki\n", encoding="utf-8")

                with patch(target, side_effect=error), self.assertRaises(type(error)):
                    build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))

                self.assertEqual(existing.read_text(encoding="utf-8"), "# Existing wiki\n")

    def test_build_wiki_writes_summary_cache_only_when_summaries_change(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)

            with patch(
                "skillfabric.wiki.summarizer.atomic_write_text",
                wraps=atomic_write_text,
            ) as write:
                build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))

            cache_writes = [
                call
                for call in write.call_args_list
                if Path(call.args[0]).name == "wiki_summary_cache.json"
            ]
            self.assertEqual(len(cache_writes), 1)

    def test_build_wiki_generates_pages_with_links_and_sections(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)

            result = build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))

            self.assertGreater(result.pages_written, 0)
            self.assertNotIn("fallback_count", result.summary)
            self.assertNotIn("summary_fallback_count", result.health.summary)
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
            self.assertFalse((workspace / "wiki" / "skills" / "index.md").exists())
            self.assertFalse((workspace / "wiki" / "workflows" / "index.md").exists())
            self.assertFalse((workspace / "wiki" / "references" / "index.md").exists())
            self.assertFalse((workspace / "wiki" / "skills" / "sources" / "index.md").exists())
            self.assertFalse((workspace / "wiki" / "overview.md").exists())
            self.assertFalse((workspace / "wiki" / "resolver.md").exists())
            self.assertFalse((workspace / "wiki" / "deliverables.md").exists())
            self.assertFalse((workspace / "wiki" / "artifacts").exists())
            self.assertFalse((workspace / "wiki" / "scenarios").exists())
            self.assertTrue((workspace / "wiki" / "workflows").exists())

    def test_skill_summary_fields_render_in_their_semantic_sections(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)

            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))

            text = (workspace / "wiki" / "skills" / "cards" / "pdf-table-parser.md").read_text(
                encoding="utf-8"
            )
            purpose = text.split("## Purpose\n\n", 1)[1].split("\n\n## Use When", 1)[0]
            use_when = text.split("## Use When\n\n", 1)[1].split("\n\n## Do Not Use When", 1)[0]
            composition = text.split("## Composition Notes\n\n", 1)[1].split(
                "\n\n## Read Full Source",
                1,
            )[0]

            self.assertEqual(
                purpose,
                "Extract tables from PDF files and save structured CSV output.",
            )
            self.assertIn("Extract tables from PDF files", use_when)
            self.assertNotIn("Produces normalized_csv_table", use_when)
            self.assertIn("Produces normalized_csv_table for downstream use.", composition)

    def test_build_wiki_replaces_the_generated_directory_as_one_unit(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            stale_file = workspace / "wiki" / "unknown-generated-file.md"
            removed_file = workspace / "wiki" / "communities" / "legacy.md"
            removed_file.parent.mkdir(parents=True, exist_ok=True)
            stale_file.write_text("# Stale generated output\n", encoding="utf-8")
            removed_file.write_text("# Removed output\n", encoding="utf-8")

            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))

            self.assertFalse(stale_file.exists())
            self.assertFalse(removed_file.exists())

    def test_build_wiki_removes_stale_flat_skill_pages(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            stale_card = workspace / "wiki" / "skills" / "pdf-table-parser.md"
            stale_card.parent.mkdir(parents=True, exist_ok=True)
            stale_card.write_text("# stale\n", encoding="utf-8")

            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))

            self.assertFalse(stale_card.exists())
            self.assertTrue(
                (workspace / "wiki" / "skills" / "cards" / "pdf-table-parser.md").exists()
            )

    def test_build_wiki_does_not_manage_reports_outside_its_output_directory(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            stale_debug = workspace / "reports" / "wiki-debug" / "raw_artifacts" / "old.md"
            stale_debug.parent.mkdir(parents=True, exist_ok=True)
            stale_debug.write_text("# stale debug\n", encoding="utf-8")

            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))

            self.assertFalse((workspace / "wiki" / "debug").exists())
            self.assertTrue(stale_debug.exists())

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

    def test_build_wiki_removes_stale_resolver_page_and_excludes_it_from_index(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            stale_resolver = workspace / "wiki" / "resolver.md"
            stale_resolver.parent.mkdir(parents=True, exist_ok=True)
            stale_resolver.write_text("# Legacy Resolver\n", encoding="utf-8")

            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))

            self.assertFalse(stale_resolver.exists())

    def test_source_excerpt_uses_fence_that_survives_nested_code_blocks(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            skills_path = workspace / "graph" / "registry.jsonl"
            rows = skills_path.read_text(encoding="utf-8").splitlines()
            patched = []
            for row in rows:
                if '"id": "skill:pdf-table-parser"' in row:
                    patched.append(
                        row.replace(
                            "Validate column headers.",
                            "Validate column headers.\\n```python\\nprint('x')\\n```",
                        )
                    )
                else:
                    patched.append(row)
            skills_path.write_text("\n".join(patched) + "\n", encoding="utf-8")

            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))

            text = (workspace / "wiki" / "skills" / "cards" / "pdf-table-parser.md").read_text(
                encoding="utf-8"
            )
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
