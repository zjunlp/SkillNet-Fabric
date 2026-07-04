from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from skillfabric.router.bundle import RouterBundleConfig, build_router_bundle
from skillfabric.storage import Workspace
from skillfabric.wiki.explorer.skill_package import SkillPackage
from skillfabric.wiki.explorer.validation import route_from_skill_package, validate_skill_package
from skillfabric.wiki.materializer import build_wiki
from skillfabric.wiki.models import WikiBuildConfig
from skillfabric.wiki.query_wiki import materialize_query_wiki
from tests.unit.wiki_helpers import build_fixture_workspace


class SkillPackageTests(unittest.TestCase):
    def test_valid_package_converts_to_route_result(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_path = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace_path)
            build_wiki(WikiBuildConfig(workspace=workspace_path, use_llm_summaries=False))
            workspace = Workspace(workspace_path)
            bundle = build_router_bundle(
                RouterBundleConfig(workspace=workspace.root, query="extract financial KPIs from a PDF report")
            )
            query_wiki = materialize_query_wiki(workspace, bundle, trace_dir=workspace.runs_dir / "pkg-valid")
            package = SkillPackage.from_dict(
                {
                    "selected_skills": [
                        {
                            "skill_id": "skill:pdf-table-parser",
                            "role": "Parse PDF tables.",
                            "evidence": [{"path": "skills/pdf-table-parser.md", "reason": "routing fit"}],
                        },
                        {
                            "skill_id": "skill:financial-kpi-extractor",
                            "role": "Extract KPI values.",
                            "evidence": [
                                {"path": "skills/financial-kpi-extractor.md", "reason": "routing fit"}
                            ],
                        },
                    ],
                    "required_edges": [
                        {
                            "before": "skill:pdf-table-parser",
                            "after": "skill:financial-kpi-extractor",
                            "relation_type": "depend_on",
                            "evidence_path": "edges/bridge_edges.jsonl",
                            "reason": "Parsed tables feed KPI extraction.",
                        }
                    ],
                    "coverage_notes": [],
                    "rationale": "Parser feeds extractor.",
                }
            )

            validation = validate_skill_package(package, query_wiki.root)
            route = route_from_skill_package(
                validation.valid_package,
                bundle,
                query="extract financial KPIs from a PDF report",
                trace_id="pkg-valid",
                trace_dir=query_wiki.root.parent,
                warnings=validation.warnings,
            )

            self.assertTrue(validation.valid, validation.errors)
            self.assertEqual(route.selected_skill_ids, ["skill:pdf-table-parser", "skill:financial-kpi-extractor"])
            self.assertTrue(route.required_edges)
            self.assertEqual(route.provenance, "claude_code")

    def test_invalid_external_skill_and_path_traversal_are_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_path = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace_path)
            build_wiki(WikiBuildConfig(workspace=workspace_path, use_llm_summaries=False))
            workspace = Workspace(workspace_path)
            bundle = build_router_bundle(
                RouterBundleConfig(workspace=workspace.root, query="extract financial KPIs from a PDF report")
            )
            query_wiki = materialize_query_wiki(workspace, bundle, trace_dir=workspace.runs_dir / "pkg-invalid")
            package = SkillPackage.from_dict(
                {
                    "selected_skills": [
                        {
                            "skill_id": "skill:not-in-manifest",
                            "role": "Invalid.",
                            "evidence": [{"path": "../outside.md", "reason": "bad path"}],
                        }
                    ],
                    "required_edges": [
                        {
                            "before": "skill:not-in-manifest",
                            "after": "skill:pdf-table-parser",
                            "evidence_path": "../edges.jsonl",
                        }
                    ],
                    "near_misses": [{"skill_id": "skill:not-in-manifest", "reason": "bad"}],
                }
            )

            validation = validate_skill_package(package, query_wiki.root)

            self.assertFalse(validation.valid)
            self.assertEqual(validation.valid_package.selected_skills, [])
            self.assertTrue(any("not in query_wiki manifest" in error for error in validation.errors))
            self.assertTrue(any("path escapes query_wiki" in error for error in validation.errors))

    def test_required_edge_accepts_skill_page_evidence(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_path = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace_path)
            build_wiki(WikiBuildConfig(workspace=workspace_path, use_llm_summaries=False))
            workspace = Workspace(workspace_path)
            bundle = build_router_bundle(
                RouterBundleConfig(workspace=workspace.root, query="extract financial KPIs from a PDF report")
            )
            query_wiki = materialize_query_wiki(workspace, bundle, trace_dir=workspace.runs_dir / "pkg-skill-edge")
            package = SkillPackage.from_dict(
                {
                    "selected_skills": [
                        {
                            "skill_id": "skill:pdf-table-parser",
                            "role": "Parse PDF tables.",
                            "evidence": [{"path": "skills/pdf-table-parser.md", "reason": "parser page"}],
                        },
                        {
                            "skill_id": "skill:financial-kpi-extractor",
                            "role": "Extract KPIs.",
                            "evidence": [
                                {"path": "skills/financial-kpi-extractor.md", "reason": "extractor page"}
                            ],
                        },
                    ],
                    "required_edges": [
                        {
                            "before": "skill:pdf-table-parser",
                            "after": "skill:financial-kpi-extractor",
                            "relation_type": "depend_on",
                            "evidence_path": "skills/financial-kpi-extractor.md",
                            "reason": "Extractor page says it uses parser output.",
                        }
                    ],
                }
            )

            validation = validate_skill_package(package, query_wiki.root)

            self.assertTrue(validation.valid, validation.errors)
            self.assertFalse(validation.errors)

    def test_state_compatibility_edge_converts_to_route_result(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_path = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace_path)
            build_wiki(WikiBuildConfig(workspace=workspace_path, use_llm_summaries=False))
            workspace = Workspace(workspace_path)
            bundle = build_router_bundle(
                RouterBundleConfig(workspace=workspace.root, query="extract financial KPIs from a PDF report")
            )
            query_wiki = materialize_query_wiki(workspace, bundle, trace_dir=workspace.runs_dir / "pkg-state-edge")
            package = SkillPackage.from_dict(
                {
                    "selected_skills": [
                        {
                            "skill_id": "skill:pdf-table-parser",
                            "role": "Parse PDF tables.",
                            "evidence": [{"path": "skills/pdf-table-parser.md", "reason": "parser page"}],
                        },
                        {
                            "skill_id": "skill:financial-kpi-extractor",
                            "role": "Extract KPIs.",
                            "evidence": [
                                {"path": "skills/financial-kpi-extractor.md", "reason": "extractor page"}
                            ],
                        },
                    ],
                    "required_edges": [
                        {
                            "before": "skill:pdf-table-parser",
                            "after": "skill:financial-kpi-extractor",
                            "relation_type": "state_compatibility",
                            "evidence_path": "skills/financial-kpi-extractor.md",
                            "reason": "Extractor consumes parser state.",
                        }
                    ],
                    "coverage_notes": [],
                    "rationale": "Parser state feeds extractor.",
                }
            )

            validation = validate_skill_package(package, query_wiki.root)
            route = route_from_skill_package(
                validation.valid_package,
                bundle,
                query="extract financial KPIs from a PDF report",
                trace_id="pkg-state-edge",
                trace_dir=query_wiki.root.parent,
                warnings=validation.warnings,
            )

            self.assertTrue(validation.valid, validation.errors)
            self.assertEqual(route.required_edges[0].edge_type, "state_compatibility")

    def test_selected_skill_without_valid_evidence_is_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_path = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace_path)
            build_wiki(WikiBuildConfig(workspace=workspace_path, use_llm_summaries=False))
            workspace = Workspace(workspace_path)
            bundle = build_router_bundle(
                RouterBundleConfig(workspace=workspace.root, query="extract financial KPIs from a PDF report")
            )
            query_wiki = materialize_query_wiki(workspace, bundle, trace_dir=workspace.runs_dir / "pkg-no-evidence")
            package = SkillPackage.from_dict(
                {
                    "selected_skills": [
                        {
                            "skill_id": "skill:pdf-table-parser",
                            "role": "No page evidence.",
                            "evidence": [],
                        },
                        {
                            "skill_id": "skill:financial-kpi-extractor",
                            "role": "All evidence is invalid.",
                            "evidence": [{"path": "skills/missing.md", "reason": "missing"}],
                        },
                    ],
                }
            )

            validation = validate_skill_package(package, query_wiki.root)

            self.assertFalse(validation.valid)
            self.assertEqual(validation.valid_package.selected_skills, [])
            self.assertTrue(any("selected skill has no valid evidence" in error for error in validation.errors))


if __name__ == "__main__":
    unittest.main()
