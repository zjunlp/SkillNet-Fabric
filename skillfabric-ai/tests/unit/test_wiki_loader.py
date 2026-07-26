from __future__ import annotations

import json
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from skillfabric.storage import Workspace
from skillfabric.wiki.loader import clear_wiki_source_cache, load_wiki_source
from tests.unit.wiki_helpers import build_fixture_workspace


class WikiLoaderTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_wiki_source_cache()

    def test_reuses_one_workspace_source_under_concurrency(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_path = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace_path)
            workspace = Workspace(workspace_path)
            calls = 0
            calls_lock = threading.Lock()

            from skillfabric.wiki import loader

            original_reader = loader._read_json_object

            def counted_reader(path: Path) -> dict:
                nonlocal calls
                if path == workspace.status_path:
                    with calls_lock:
                        calls += 1
                    time.sleep(0.02)
                return original_reader(path)

            with (
                mock.patch.object(loader, "_read_json_object", counted_reader),
                ThreadPoolExecutor(max_workers=8) as pool,
            ):
                sources = list(pool.map(lambda _index: load_wiki_source(workspace), range(8)))

            self.assertEqual(calls, 1)
            self.assertEqual(len({id(source) for source in sources}), 1)

    def test_reuses_source_across_query_wiki_and_planner_loaders(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_path = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace_path)
            workspace = Workspace(workspace_path)

            from skillfabric.orchestrator.package import (
                load_wiki_source as planner_source_loader,
            )
            from skillfabric.wiki.query_wiki import (
                load_wiki_source as query_wiki_source_loader,
            )

            route_source = load_wiki_source(workspace)
            query_wiki_source = query_wiki_source_loader(workspace)
            planner_source = planner_source_loader(workspace)

            self.assertIs(query_wiki_source, route_source)
            self.assertIs(planner_source, route_source)

    def test_reloads_source_after_ready_artifact_identity_changes(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_path = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace_path)
            workspace = Workspace(workspace_path)

            first = load_wiki_source(workspace)
            status_path = workspace.status_path
            status = status_path.read_bytes()
            status_path.write_bytes(status + b" ")
            second = load_wiki_source(workspace)

            self.assertIsNot(second, first)
            self.assertEqual(second.build_id, first.build_id)

    def test_clear_releases_cached_source(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_path = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace_path)
            workspace = Workspace(workspace_path)

            first = load_wiki_source(workspace)
            clear_wiki_source_cache()
            second = load_wiki_source(workspace)

            self.assertIsNot(second, first)
            self.assertEqual(second.build_id, first.build_id)

    def test_loads_semantic_views_for_wiki(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)

            source = load_wiki_source(Workspace(workspace))

            self.assertIn("skill:pdf-table-parser", source.skills)
            self.assertIn("skill:pdf-table-parser", source.contracts)
            self.assertTrue(source.skill_core_links("skill:financial-kpi-extractor"))
            self.assertTrue(source.operational_edges)

    def test_loads_raw_text_with_unicode_line_separators(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            registry_path = workspace / "graph" / "registry.jsonl"
            rows = registry_path.read_text(encoding="utf-8").split("\n")
            payload = json.loads(rows[0])
            raw_text = "before\u0085middle\u2028after"
            payload["raw_text"] = raw_text
            rows[0] = json.dumps(payload, ensure_ascii=False)
            registry_path.write_text("\n".join(rows), encoding="utf-8")

            source = load_wiki_source(Workspace(workspace))

            self.assertEqual(source.skills[payload["id"]].raw_text, raw_text)

    def test_rejects_unknown_graph_fields(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            graph_path = workspace / "graph" / "graph.json"
            payload = json.loads(graph_path.read_text(encoding="utf-8"))
            payload["unused"] = True
            graph_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "canonical fields"):
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

    def test_rejects_noncanonical_ready_status(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            status_path = workspace / "status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["unused"] = True
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
