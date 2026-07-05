from __future__ import annotations

import inspect
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from skillfabric.router.bundle import RouterBundle, RouterSkillCandidate
from skillfabric.router.routing import RouterConfig, route_task
from skillfabric.router.selection import _select_fallback_candidates
from skillfabric.wiki.explorer.skill_package import SkillPackage
from skillfabric.wiki.explorer.validation import route_from_skill_package
from skillfabric.wiki.materializer import build_wiki
from skillfabric.wiki.models import WikiBuildConfig
from tests.unit.test_wiki_explorer import _StubSdkRuntime
from tests.unit.wiki_helpers import build_fixture_workspace


class RouterRouteTests(unittest.TestCase):
    def test_deterministic_fallback_uses_bundle_order_without_hidden_filters(self) -> None:
        candidates = [
            RouterSkillCandidate("skill:core-a", "core-a", 3.0, sources=["bm25"], score_breakdown={"bm25": 0.3}),
            RouterSkillCandidate("skill:core-b", "core-b", 1.4, sources=["lexical"], score_breakdown={"lexical": 0.3}),
            RouterSkillCandidate("skill:core-c", "core-c", 1.1, sources=["embedding"], score_breakdown={"embedding": 0.3}),
            RouterSkillCandidate("skill:weak-lexical", "weak-lexical", 0.3, sources=["lexical"], score_breakdown={"lexical": 0.1}),
            RouterSkillCandidate("skill:weak-ppr", "weak-ppr", 0.28, sources=["ppr:similar_to"], ppr_score=0.02),
            RouterSkillCandidate("skill:object", "object", 0.2, sources=["object:produces"]),
            RouterSkillCandidate("skill:interface", "interface", 0.15, sources=["interface:field"]),
        ]

        selected = _select_fallback_candidates(candidates, max_selected_skills=8)
        selected_ids = [item.skill_id for item in selected]

        self.assertEqual(
            selected_ids,
            [item.skill_id for item in candidates],
        )

    def test_strict_explorer_raises_on_partial_validation_errors(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))

            runtime = _StubSdkRuntime(
                {
                    "selected_skills": [
                        {
                            "skill_id": "skill:pdf-table-parser",
                            "role": "Needed to parse PDF tables.",
                            "evidence": [{"path": "skills/cards/pdf-table-parser.md", "reason": "valid"}],
                        },
                        {
                            "skill_id": "skill:not-real",
                            "role": "Invalid skill should make strict mode fail.",
                            "evidence": [{"path": "skills/cards/pdf-table-parser.md", "reason": "wrong"}],
                        },
                    ],
                    "rationale": "One valid, one invalid.",
                }
            )

            with self.assertRaises(ValueError):
                route_task(
                    RouterConfig(
                        workspace=workspace,
                        query="extract financial KPIs from a PDF report",
                        trace_id="strict-partial-invalid",
                        explorer_backend="claude-code",
                        strict_explorer=True,
                    ),
                    sdk_runtime=runtime,
                )

    def test_llm_route_filters_invalid_skills_and_writes_trace(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))
            runtime = _StubSdkRuntime(
                {
                    "selected_skills": [
                        {
                            "skill_id": "skill:pdf-table-parser",
                            "role": "Needed to parse PDF tables.",
                            "evidence": [
                                {
                                    "path": "skills/cards/pdf-table-parser.md",
                                    "reason": "PDF table parser page.",
                                }
                            ],
                        },
                        {
                            "skill_id": "skill:financial-kpi-extractor",
                            "role": "Needed to extract KPI values.",
                            "evidence": [
                                {
                                    "path": "skills/cards/financial-kpi-extractor.md",
                                    "reason": "KPI extractor page.",
                                }
                            ],
                        },
                        {
                            "skill_id": "skill:not-real",
                            "role": "Invalid skill must be dropped.",
                            "evidence": [],
                        },
                    ],
                    "required_edges": [],
                    "ordered_hints": [],
                    "near_misses": [{"skill_id": "skill:report-writer", "reason": "Not needed for extraction."}],
                    "rationale": "Use the parser before the KPI extractor.",
                }
            )

            result = route_task(
                RouterConfig(
                    workspace=workspace,
                    query="extract financial KPIs from a PDF report",
                    env_file=".env",
                    use_llm_router=True,
                    max_selected_skills=4,
                    workflow_confidence_threshold=0.9,
                    trace_id="route-test",
                    explorer_backend="claude-code",
                ),
                sdk_runtime=runtime,
            )

            payload = result.to_dict()
            selected_ids = [item["skill_id"] for item in payload["selected_skills"]]
            self.assertEqual(selected_ids, ["skill:pdf-table-parser", "skill:financial-kpi-extractor"])
            self.assertTrue(any("not in query_wiki manifest" in warning for warning in payload["warnings"]))
            self.assertEqual(payload["provenance"], "claude_code")
            self.assertIn("skills/cards/pdf-table-parser.md", payload["wiki_pages_read"])
            self.assertTrue((workspace / "runs" / "route-test" / "route.json").exists())
            self.assertTrue((workspace / "runs" / "route-test" / "query_wiki" / "manifest.json").exists())
            self.assertTrue((workspace / "runs" / "route-test" / "cc_explorer" / "skill_package.json").exists())
            self.assertTrue((workspace / "runs" / "route-test" / "cc_explorer" / "validation.json").exists())

            edge = next(item for item in payload["required_edges"] if item["after_skill"] == "skill:financial-kpi-extractor")
            self.assertEqual(edge["before_skill"], "skill:pdf-table-parser")
            self.assertEqual(edge["source"], "execution_index")

    def test_llm_failure_falls_back_to_ranked_bundle(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))

            class BrokenRuntime(_StubSdkRuntime):
                async def query(self, *, prompt: Any, options: Any) -> Any:
                    del prompt, options
                    raise RuntimeError("router unavailable")
                    yield

            result = route_task(
                RouterConfig(
                    workspace=workspace,
                    query="parse pdf tables",
                    use_llm_router=True,
                    max_selected_skills=3,
                    trace_id="route-fallback",
                    explorer_backend="claude-code",
                ),
                sdk_runtime=BrokenRuntime({}),
            )

            payload = result.to_dict()
            self.assertEqual(payload["provenance"], "deterministic_fallback")
            self.assertGreaterEqual(len(payload["selected_skills"]), 1)
            self.assertTrue(any("explorer" in warning for warning in payload["warnings"]))
            self.assertTrue((workspace / "runs" / "route-fallback" / "route.json").exists())
            removed_context_name = "route_" + "context.md"
            self.assertFalse((workspace / "runs" / "route-fallback" / removed_context_name).exists())
            self.assertTrue((workspace / "runs" / "route-fallback" / "query_wiki" / "manifest.json").exists())
            self.assertTrue((workspace / "runs" / "route-fallback" / "cc_explorer" / "error.json").exists())

    def test_route_backend_rejects_removed_mode(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))

            with self.assertRaises(ValueError):
                route_task(
                    RouterConfig(
                        workspace=workspace,
                        query="parse pdf tables",
                        trace_id="old-removed-mode",
                        explorer_backend="single" + "-shot",
                    )
                )

    def test_removed_route_completion_injection_is_absent(self) -> None:
        self.assertNotIn("completion" + "_fn", inspect.signature(route_task).parameters)
        config_parameters = inspect.signature(RouterConfig).parameters
        self.assertNotIn("max_context_chars", config_parameters)
        self.assertNotIn("max_page_chars", config_parameters)

    def test_route_reconciles_ordered_hint_sequence_against_reversed_llm_edges(self) -> None:
        with TemporaryDirectory() as tmp:
            query = (
                "Analyze artifacts/penguins.csv with statistical tests, generate at least 4 PNG figures, "
                "write report.docx, and create presentation.pptx."
            )
            selected = [
                RouterSkillCandidate("skill:xlsx", "xlsx", 1.0),
                RouterSkillCandidate("skill:data-visualization", "data-visualization", 1.0),
                RouterSkillCandidate("skill:docx", "docx", 1.0),
                RouterSkillCandidate("skill:pptx", "pptx", 1.0),
                RouterSkillCandidate("skill:data-storytelling", "data-storytelling", 1.0),
            ]
            bundle = RouterBundle(
                query=query,
                selected_skills=selected,
                communities=[],
                workflow_hints=[],
                wiki_pages=[],
            )
            package = SkillPackage.from_dict(
                {
                    "selected_skills": [
                        {"skill_id": "skill:xlsx", "role": "Analyze CSV.", "evidence": []},
                        {
                            "skill_id": "skill:data-visualization",
                            "role": "Generate PNG figures.",
                            "evidence": [],
                        },
                        {"skill_id": "skill:docx", "role": "Write report.docx.", "evidence": []},
                        {"skill_id": "skill:pptx", "role": "Create presentation.pptx.", "evidence": []},
                        {
                            "skill_id": "skill:data-storytelling",
                            "role": "Build the narrative before report/deck authoring.",
                            "evidence": [],
                        },
                    ],
                    "required_edges": [
                        {
                            "before": "skill:xlsx",
                            "after": "skill:data-visualization",
                            "relation_type": "depend_on",
                            "reason": "Analysis informs figures.",
                        },
                        {
                            "before": "skill:docx",
                            "after": "skill:data-storytelling",
                            "relation_type": "depend_on",
                            "reason": "The document should present analysis in a coherent story.",
                        },
                        {
                            "before": "skill:pptx",
                            "after": "skill:data-storytelling",
                            "relation_type": "depend_on",
                            "reason": "The slide deck needs a clear narrative arc.",
                        },
                    ],
                    "ordered_hints": [
                        {"skill_id": "skill:xlsx"},
                        {"skill_id": "skill:data-visualization"},
                        {"skill_id": "skill:data-storytelling"},
                        {"skill_id": "skill:docx"},
                        {"skill_id": "skill:pptx"},
                    ],
                    "rationale": "Use analysis, figures, narrative, report, then deck.",
                }
            )
            warnings: list[str] = []

            route = route_from_skill_package(
                package,
                bundle,
                query=query,
                trace_id="penguin-route-conflict",
                trace_dir=Path(tmp) / ".skillfabric" / "runs" / "penguin-route-conflict",
                max_selected_skills=10,
                warnings=warnings,
            )

            required_pairs = {(edge.before_skill, edge.after_skill) for edge in route.required_edges}
            hint_pairs = {(edge.before_skill, edge.after_skill) for edge in route.ordered_hints}
            self.assertIn(("skill:data-storytelling", "skill:docx"), hint_pairs)
            self.assertIn(("skill:docx", "skill:pptx"), hint_pairs)
            self.assertNotIn(("skill:docx", "skill:data-storytelling"), required_pairs)
            self.assertNotIn(("skill:pptx", "skill:data-storytelling"), required_pairs)
            self.assertNotIn(("skill:data-storytelling", "skill:docx"), required_pairs)
            self.assertNotIn(("skill:docx", "skill:pptx"), required_pairs)
            self.assertTrue(any("dropped conflicting LLM required edge" in warning for warning in warnings))

    def test_ordered_hints_remain_soft_route_hints(self) -> None:
        with TemporaryDirectory() as tmp:
            query = "Create a literature review and then add a trend timeline."
            selected = [
                RouterSkillCandidate("skill:literature-review", "literature-review", 2.0),
                RouterSkillCandidate("skill:trend-report", "trend-report", 1.8),
            ]
            bundle = RouterBundle(
                query=query,
                selected_skills=selected,
                communities=[],
                workflow_hints=[],
                wiki_pages=[],
            )
            package = SkillPackage.from_dict(
                {
                    "selected_skills": [
                        {
                            "skill_id": "skill:literature-review",
                            "role": "Build the paper corpus and thematic synthesis.",
                            "evidence": [],
                        },
                        {
                            "skill_id": "skill:trend-report",
                            "role": "Add recent-development and timeline analysis.",
                            "evidence": [],
                        },
                    ],
                    "required_edges": [],
                    "ordered_hints": [
                        {"skill_id": "skill:literature-review"},
                        {"skill_id": "skill:trend-report"},
                    ],
                    "rationale": "The trend report is useful after the review, but not a hard dependency.",
                }
            )

            route = route_from_skill_package(
                package,
                bundle,
                query=query,
                trace_id="soft-hints",
                trace_dir=Path(tmp) / ".skillfabric" / "runs" / "soft-hints",
                warnings=[],
            )

            self.assertEqual(route.required_edges, [])
            self.assertEqual(len(route.ordered_hints), 1)
            self.assertEqual(route.ordered_hints[0].before_skill, "skill:literature-review")
            self.assertEqual(route.ordered_hints[0].after_skill, "skill:trend-report")
            self.assertEqual(route.ordered_hints[0].edge_type, "hint")

    def test_route_result_omits_removed_task_coverage_fields(self) -> None:
        with TemporaryDirectory() as tmp:
            query = (
                "Analyze artifacts/penguins.csv with statistical tests, generate at least 4 PNG figures, "
                "write report.docx, and create presentation.pptx."
            )
            selected = [
                RouterSkillCandidate("skill:xlsx", "xlsx", 1.0),
                RouterSkillCandidate("skill:data-visualization", "data-visualization", 1.0),
                RouterSkillCandidate("skill:docx", "docx", 1.0),
                RouterSkillCandidate("skill:pptx", "pptx", 1.0),
            ]
            bundle = RouterBundle(
                query=query,
                selected_skills=selected,
                communities=[],
                workflow_hints=[],
                wiki_pages=[],
            )
            package = SkillPackage.from_dict(
                {
                    "selected_skills": [
                        {
                            "skill_id": "skill:data-visualization",
                            "role": "Generate figures.",
                            "evidence": [],
                        },
                        {"skill_id": "skill:docx", "role": "Write report.", "evidence": []},
                        {"skill_id": "skill:pptx", "role": "Create deck.", "evidence": []},
                    ],
                    "rationale": "Explorer covered deliverables but omitted the analysis skill.",
                }
            )
            warnings: list[str] = []

            route = route_from_skill_package(
                package,
                bundle,
                query=query,
                trace_id="penguin-route-coverage",
                trace_dir=Path(tmp) / ".skillfabric" / "runs" / "penguin-route-coverage",
                max_selected_skills=10,
                warnings=warnings,
            )

            selected_ids = [item.skill_id for item in route.selected_skills]
            self.assertNotIn("skill:xlsx", selected_ids)
            payload = route.to_dict()
            self.assertNotIn("task_understanding", payload)
            self.assertNotIn("coverage_diagnostics", payload)


if __name__ == "__main__":
    unittest.main()
