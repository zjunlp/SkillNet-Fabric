from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from skillfabric.storage import Workspace
from skillfabric.wiki.loader import load_wiki_source
from tests.unit.wiki_helpers import build_fixture_workspace


class WikiLoaderTests(unittest.TestCase):
    def test_loads_compiled_graph_views_for_wiki(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)

            source = load_wiki_source(Workspace(workspace))

            self.assertIn("skill:pdf-table-parser", source.skills)
            self.assertIn("skill:pdf-table-parser", source.interfaces)
            self.assertTrue(source.raw_artifacts)
            self.assertFalse(hasattr(source, "artifacts"))
            self.assertFalse(hasattr(source, "scenarios"))
            self.assertFalse(hasattr(source, "skill_artifact_edges"))
            self.assertFalse(hasattr(source, "skill_scenario_edges"))
            self.assertFalse(hasattr(source, "communities"))
            self.assertFalse(hasattr(source, "community_members"))
            self.assertTrue(source.skill_core_links("skill:financial-kpi-extractor"))
            execution_links = source.skill_execution_links("skill:pdf-table-parser")
            self.assertTrue(execution_links["workflow_hints"])
            self.assertEqual(set(execution_links), {"workflow_hints"})
            self.assertTrue(source.evidence_lookup)


if __name__ == "__main__":
    unittest.main()
