from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from skillfabric.wiki.materializer import build_wiki
from skillfabric.wiki.models import WikiBuildConfig
from skillfabric.wiki.renderers import _first_paragraph
from tests.unit.wiki_helpers import build_fixture_workspace


class WikiMaterializerTests(unittest.TestCase):
    def test_build_wiki_preserves_existing_pages_until_source_and_summaries_succeed(self) -> None:
        failures = (
            ("skillfabric.wiki.materializer.load_wiki_source", ValueError("invalid graph")),
            (
                "skillfabric.wiki.materializer.summary_from_payload",
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
                    build_wiki(WikiBuildConfig(workspace=workspace))

                self.assertEqual(existing.read_text(encoding="utf-8"), "# Existing wiki\n")

    def test_build_wiki_is_deterministic_without_summary_cache(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)

            build_wiki(WikiBuildConfig(workspace=workspace))
            first_pages = {
                path.relative_to(workspace / "wiki"): path.read_bytes()
                for path in sorted((workspace / "wiki").rglob("*.md"))
            }
            build_wiki(WikiBuildConfig(workspace=workspace))
            second_pages = {
                path.relative_to(workspace / "wiki"): path.read_bytes()
                for path in sorted((workspace / "wiki").rglob("*.md"))
            }

            self.assertEqual(second_pages, first_pages)
            self.assertFalse((workspace / "cache" / "wiki_summary_cache.json").exists())

    def test_build_wiki_generates_pages_with_links_and_sections(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)

            config = WikiBuildConfig(workspace=workspace)
            build_wiki(config)
            result = build_wiki(config)

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
            self.assertIn("provides_to: [[skills/cards/financial-kpi-extractor]]", text)
            self.assertIn("## Read Full Source", text)
            self.assertNotIn("/Users/", text)
            source_page = workspace / "wiki" / "skills" / "sources" / "pdf-table-parser.md"
            self.assertTrue(source_page.exists())
            source_text = source_page.read_text(encoding="utf-8")
            self.assertIn("type: Skill Source", source_text)
            self.assertIn("card: ../cards/pdf-table-parser.md", source_text)
            self.assertIn("# Full SKILL.md", source_text)
            self.assertIn("Extract tables", source_text)
            self.assertNotIn("/Users/", source_text)
            consumer_text = (
                workspace / "wiki" / "skills" / "cards" / "financial-kpi-extractor.md"
            ).read_text(encoding="utf-8")
            self.assertIn("consumes_from: [[skills/cards/pdf-table-parser]]", consumer_text)
            root_index_text = (workspace / "wiki" / "index.md").read_text(encoding="utf-8")
            self.assertIn("## Skill Cards", root_index_text)
            self.assertIn("[pdf-table-parser](skills/cards/pdf-table-parser.md)", root_index_text)
            self.assertIn("[full SKILL.md](skills/sources/pdf-table-parser.md)", root_index_text)
            self.assertIn("## Full Skill Sources", root_index_text)
            self.assertIn("Extract tables from PDF files", root_index_text)
            self.assertTrue((workspace / "wiki" / "workflows").exists())
            log_text = (workspace / "reports" / "wiki_log.md").read_text(encoding="utf-8")
            self.assertEqual(log_text.count("wiki-build | build_id="), 2)

    def test_skill_summary_fields_render_in_their_semantic_sections(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)

            build_wiki(WikiBuildConfig(workspace=workspace))

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
            stale_file.write_text("# Stale generated output\n", encoding="utf-8")

            build_wiki(WikiBuildConfig(workspace=workspace))

            self.assertFalse(stale_file.exists())

    def test_build_wiki_does_not_manage_reports_outside_its_output_directory(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            stale_debug = workspace / "reports" / "wiki-debug" / "raw_artifacts" / "old.md"
            stale_debug.parent.mkdir(parents=True, exist_ok=True)
            stale_debug.write_text("# stale debug\n", encoding="utf-8")

            build_wiki(WikiBuildConfig(workspace=workspace))

            self.assertFalse((workspace / "wiki" / "debug").exists())
            self.assertTrue(stale_debug.exists())

    def test_first_paragraph_truncates_on_word_boundary(self) -> None:
        summary = _first_paragraph("routing " * 80)

        self.assertLessEqual(len(summary), 240)
        self.assertTrue(summary.endswith("..."))
        self.assertNotIn(" ...", summary)
        self.assertTrue(summary.removesuffix("...").endswith("routing"))


if __name__ == "__main__":
    unittest.main()
