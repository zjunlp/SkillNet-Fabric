from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from skillfabric.storage import Workspace
from skillfabric.wiki.explorer.search_index import load_page_index
from skillfabric.wiki.loader import load_wiki_source
from skillfabric.wiki.materializer import _deliverables_page, build_wiki
from skillfabric.wiki.models import WikiBuildConfig
from tests.unit.wiki_helpers import build_fixture_workspace


class WikiMaterializerTests(unittest.TestCase):
    def test_build_wiki_generates_pages_with_links_and_sections(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)

            result = build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))

            self.assertGreater(result.pages_written, 0)
            skill_page = workspace / "wiki" / "skills" / "pdf-table-parser.md"
            self.assertTrue(skill_page.exists())
            text = skill_page.read_text(encoding="utf-8")
            self.assertIn("type: skill", text)
            self.assertIn("## Routing Summary", text)
            self.assertIn("## Routing Fit", text)
            self.assertIn("## When To Use", text)
            self.assertIn("## Produces", text)
            self.assertIn("## Capability Contract", text)
            self.assertIn("Granularity:", text)
            self.assertIn("Execution Role:", text)
            self.assertIn("## Works With", text)
            self.assertIn("## Depends On", text)
            self.assertIn("confidence=0.92", text)
            self.assertIn("## Workflow Hints", text)
            self.assertIn("## Failure Modes", text)
            self.assertIn("## Evidence", text)
            self.assertNotIn("[[artifacts/", text)
            self.assertNotIn("[[scenarios/", text)
            self.assertNotIn("raw_output", text)
            self.assertIn("\n```markdown\n", text)
            self.assertTrue((workspace / "wiki" / "communities").exists())
            self.assertTrue((workspace / "wiki" / "overview.md").exists())
            self.assertFalse((workspace / "wiki" / "resolver.md").exists())
            self.assertTrue((workspace / "wiki" / "deliverables.md").exists())
            overview_text = (workspace / "wiki" / "overview.md").read_text(encoding="utf-8")
            self.assertNotIn("resolver.md", overview_text)
            self.assertFalse((workspace / "wiki" / "artifacts").exists())
            self.assertFalse((workspace / "wiki" / "scenarios").exists())
            self.assertTrue((workspace / "wiki" / "workflows").exists())

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

            self.assertTrue(list((workspace / "wiki" / "debug" / "raw_artifacts").glob("*.md")))
            self.assertTrue((workspace / "wiki" / "debug" / "extraction_report.md").exists())

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
            skills_path = workspace / "registry" / "skills.jsonl"
            rows = skills_path.read_text(encoding="utf-8").splitlines()
            patched = []
            for row in rows:
                if '"id": "skill:pdf-table-parser"' in row:
                    patched.append(row.replace("Validate column headers.", "Validate column headers.\\n```python\\nprint('x')\\n```"))
                else:
                    patched.append(row)
            skills_path.write_text("\n".join(patched) + "\n", encoding="utf-8")

            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))

            text = (workspace / "wiki" / "skills" / "pdf-table-parser.md").read_text(encoding="utf-8")
            self.assertIn("````markdown", text)
            self.assertIn("```python", text)

    def test_deliverables_are_interface_based(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_path = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace_path)
            workspace = Workspace(workspace_path)
            source = load_wiki_source(workspace)

            deliverables_text = _deliverables_page(source, workspace).text

            self.assertIn("Producer Index", deliverables_text)
            self.assertNotIn("Canonical Deliverable Requirements", deliverables_text)
            self.assertNotIn("deliverable:pptx", deliverables_text)


if __name__ == "__main__":
    unittest.main()
