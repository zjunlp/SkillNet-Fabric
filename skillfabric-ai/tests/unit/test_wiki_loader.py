from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from skillfabric.storage import Workspace
from skillfabric.wiki.loader import load_wiki_source
from tests.unit.wiki_helpers import build_fixture_workspace


class WikiLoaderTests(unittest.TestCase):
    def test_loads_schema_v2_semantic_views_for_wiki(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)

            source = load_wiki_source(Workspace(workspace))

            self.assertIn("skill:pdf-table-parser", source.skills)
            self.assertIn("skill:pdf-table-parser", source.contracts)
            self.assertTrue(source.skill_core_links("skill:financial-kpi-extractor"))
            self.assertTrue(source.operational_edges)

    def test_rejects_obsolete_graph_schema(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            graph_path = workspace / "graph" / "graph.json"
            payload = json.loads(graph_path.read_text(encoding="utf-8"))
            payload["schema_version"] = "1.0"
            graph_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "rebuild"):
                load_wiki_source(Workspace(workspace))

    def test_rejects_non_ready_workspace_even_when_graph_artifacts_remain(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            status_path = workspace / "status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status.update({"state": "failed", "failed_stage": "contracts"})
            status_path.write_text(json.dumps(status), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "not ready"):
                load_wiki_source(Workspace(workspace))

    def test_rejects_non_object_graph_edge_instead_of_skipping_it(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            graph_path = workspace / "graph" / "graph.json"
            payload = json.loads(graph_path.read_text(encoding="utf-8"))
            payload["edges"].append("invalid-edge")
            graph_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "graph edges"):
                load_wiki_source(Workspace(workspace))

    def test_rejects_duplicate_registry_rows_instead_of_overwriting_them(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            registry_path = workspace / "graph" / "registry.jsonl"
            rows = registry_path.read_text(encoding="utf-8").splitlines()
            registry_path.write_text("\n".join([*rows, rows[0]]) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate skill id"):
                load_wiki_source(Workspace(workspace))


if __name__ == "__main__":
    unittest.main()
