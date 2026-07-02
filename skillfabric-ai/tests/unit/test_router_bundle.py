from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from skillfabric.compiled_graph.interface.models import InterfaceField, SkillInterface
from skillfabric.compiled_graph.models import CommunityNode, Edge, GraphDocument
from skillfabric.indexing.bm25 import build_bm25_index
from skillfabric.indexing.embeddings import build_embedding_store
from skillfabric.registry.models import SkillNode
from skillfabric.router.bundle import RouterBundleConfig, build_router_bundle
from skillfabric.storage import Workspace
from tests.unit.fake_embeddings import FakeEmbeddingProvider


def _skill(skill_id: str, name: str, description: str, body: str) -> SkillNode:
    return SkillNode(
        id=skill_id,
        type="skill",
        name=name,
        description=description,
        source_path=f"/skills/{name}/SKILL.md",
        wiki_path="",
        content_hash=f"hash-{skill_id}",
        token_count=len(body.split()),
        canonical_skill_text_hash=f"text-hash-{skill_id}",
        raw_text=body,
    )


def _interface(
    skill_id: str,
    *,
    produces: list[str] | None = None,
    requires: list[str] | None = None,
    summary: str = "",
    when_to_use: str = "",
    execution_role: str = "helper",
    uses_tools: list[str] | None = None,
) -> dict[str, object]:
    return SkillInterface(
        skill_id=skill_id,
        content_hash=f"hash-{skill_id}",
        capability_summary=summary,
        when_to_use=when_to_use,
        execution_role=execution_role,
        produces=[InterfaceField(name=item, kind="artifact", confidence=0.95) for item in produces or []],
        requires=[InterfaceField(name=item, kind="artifact", confidence=0.95) for item in requires or []],
        uses_tools=[InterfaceField(name=item, kind="tool", confidence=0.95) for item in uses_tools or []],
    ).to_dict()


class RouterBundleTests(unittest.TestCase):
    def test_facet_preservation_keeps_lower_scored_specialist_candidate(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Workspace(Path(tmp) / ".skillfabric")
            workspace.ensure()
            skills = [
                _skill(
                    "skill:noisy-high-score",
                    "noisy-high-score",
                    "Repeated presentation pdf chart report analysis slides output.",
                    "presentation presentation pdf pdf chart chart report report analysis analysis slides slides.",
                ),
                _skill(
                    "skill:pptx",
                    "pptx",
                    "Create and validate PowerPoint files.",
                    "Inspect presentation layouts and export slides.",
                ),
                _skill(
                    "skill:academic-seminar-slides",
                    "academic-seminar-slides",
                    "Specialized academic research seminar slide workflow.",
                    "Use for PhD seminar presentations, critical literature reviews, references, argument structure, and scientific visuals.",
                ),
            ]
            graph = GraphDocument(
                schema_version="1.0",
                build_id="facet-pptx-test",
                nodes=skills,
                edges=[],
                stats={},
                config_digest="facet-pptx-test",
            )
            workspace.write_json(workspace.graph_dir / "graph.json", graph.to_dict())
            workspace.write_jsonl(
                workspace.registry_dir / "skills.jsonl",
                [skill.to_dict(include_raw_text=True) for skill in skills],
            )
            workspace.write_jsonl(
                workspace.interfaces_dir / "skill_interfaces.jsonl",
                [
                    _interface(
                        "skill:noisy-high-score",
                        produces=["analysis_output"],
                        summary="Generic presentation pdf chart report analysis output.",
                    ),
                    _interface(
                        "skill:pptx",
                        produces=["presentation_document"],
                        summary="Create and validate PowerPoint presentations.",
                    ),
                    _interface(
                        "skill:academic-seminar-slides",
                        produces=["presentation_document"],
                        summary="Create polished academic seminar slide decks with research narrative and scientific visuals.",
                        when_to_use="Use for PhD seminars, critical research reviews, scientific presentations, references, and academic argument structure.",
                        execution_role="transformer",
                    ),
                ],
            )
            build_bm25_index(skills, workspace.index_dir / "bm25.sqlite")
            for skill in skills:
                page = workspace.wiki_skills_dir / f"{skill.name}.md"
                page.parent.mkdir(parents=True, exist_ok=True)
                page.write_text(f"# {skill.name}\n", encoding="utf-8")

            bundle = build_router_bundle(
                RouterBundleConfig(
                    workspace=workspace.root,
                    query=(
                        "Create a polished academic seminar presentation for a PhD research review. "
                        "The deck needs a clear scientific argument, references, and rich visuals. "
                        "Output presentation.pdf."
                    ),
                    seed_limit=1,
                    expanded_limit=2,
                )
            )
            selected = {item.skill_id: item for item in bundle.selected_skills}

            self.assertIn("skill:academic-seminar-slides", selected)
            self.assertTrue(
                any(
                    source.startswith(("interface:", "object:", "execution:", "facet:interface_text"))
                    for source in selected["skill:academic-seminar-slides"].sources
                )
            )

    def test_object_and_interface_scores_softly_recall_deliverable_skills(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Workspace(Path(tmp) / ".skillfabric")
            workspace.ensure()
            skills = [
                _skill(
                    "skill:data-visualization",
                    "data-visualization",
                    "Create publication-quality PNG figures and charts from data.",
                    "Use pandas, matplotlib, and seaborn to generate PNG figures.",
                ),
                _skill(
                    "skill:xlsx",
                    "xlsx",
                    "Analyze CSV and spreadsheet tabular data.",
                    "Load CSV data, compute descriptive statistics, and write spreadsheet tables.",
                ),
                _skill(
                    "skill:docx",
                    "docx",
                    "Create and edit Word .docx documents.",
                    "Generate report.docx documents with embedded tables and figures.",
                ),
                _skill(
                    "skill:pptx",
                    "pptx",
                    "Create academic PowerPoint .pptx presentations.",
                    "Generate presentation.pptx slide decks from outlines and figures.",
                ),
                _skill(
                    "skill:media-processing",
                    "media-processing",
                    "Process audio and video media files.",
                    "Transcode media assets and apply filters.",
                ),
            ]
            graph = GraphDocument(
                schema_version="1.0",
                build_id="deliverable-test",
                nodes=skills,
                edges=[],
                stats={},
                config_digest="deliverable-test",
            )
            workspace.write_json(workspace.graph_dir / "graph.json", graph.to_dict())
            workspace.write_jsonl(
                workspace.registry_dir / "skills.jsonl",
                [skill.to_dict(include_raw_text=True) for skill in skills],
            )
            workspace.write_jsonl(
                workspace.interfaces_dir / "skill_interfaces.jsonl",
                [
                    _interface(
                        "skill:data-visualization",
                        produces=["image_asset", "png figures"],
                        requires=["csv_table"],
                        summary="Create PNG charts and figures from data.",
                    ),
                    _interface(
                        "skill:xlsx",
                        produces=["statistical_summary", "spreadsheet_table"],
                        requires=["csv_table"],
                        summary="Analyze CSV and spreadsheet tabular data.",
                        when_to_use="Use for statistical analysis over CSV datasets.",
                    ),
                    _interface(
                        "skill:docx",
                        produces=["docx_document", "report document"],
                        summary="Create Word docx reports.",
                    ),
                    _interface(
                        "skill:pptx",
                        produces=["presentation_document", "slide deck"],
                        summary="Create PowerPoint presentation decks.",
                    ),
                ],
            )
            workspace.write_jsonl(
                workspace.execution_dir / "canonical_objects.jsonl",
                [
                    {
                        "canonical_id": "canonical:png_figures",
                        "name": "png figures",
                        "type": "artifact",
                        "description": "Publication-quality PNG figure artifacts.",
                        "aliases": ["chart.png", "image_asset"],
                        "required_by": [],
                        "produced_by": ["skill:data-visualization"],
                        "reuse_count": 1,
                        "promoted": True,
                        "confidence": 0.95,
                        "provenance": "test",
                        "reason": "interface produces png figures",
                    },
                    {
                        "canonical_id": "canonical:docx_report",
                        "name": "docx report",
                        "type": "artifact",
                        "description": "Word report document.",
                        "aliases": ["report.docx", "docx_document"],
                        "required_by": [],
                        "produced_by": ["skill:docx"],
                        "reuse_count": 1,
                        "promoted": True,
                        "confidence": 0.95,
                        "provenance": "test",
                        "reason": "interface produces docx reports",
                    },
                    {
                        "canonical_id": "canonical:pptx_deck",
                        "name": "presentation deck",
                        "type": "artifact",
                        "description": "PowerPoint presentation deck.",
                        "aliases": ["presentation.pptx", "slide deck"],
                        "required_by": [],
                        "produced_by": ["skill:pptx"],
                        "reuse_count": 1,
                        "promoted": True,
                        "confidence": 0.95,
                        "provenance": "test",
                        "reason": "interface produces presentation decks",
                    },
                ],
            )
            workspace.write_jsonl(
                workspace.execution_dir / "execution_index.jsonl",
                [
                    {
                        "source_skill": "skill:xlsx",
                        "target_skill": "skill:data-visualization",
                        "relation_type": "artifact_compatibility",
                        "canonical_object": "statistical_summary",
                        "direction": "source_to_target",
                        "confidence": 0.96,
                        "evidence": [],
                        "projected_edge_type": "depend_on",
                        "reason": "statistics feed chart generation",
                        "metadata": {},
                    }
                ],
            )
            build_bm25_index(skills, workspace.index_dir / "bm25.sqlite")
            for skill in skills:
                page = workspace.wiki_skills_dir / f"{skill.name}.md"
                page.parent.mkdir(parents=True, exist_ok=True)
                page.write_text(f"# {skill.name}\n", encoding="utf-8")

            bundle = build_router_bundle(
                RouterBundleConfig(
                    workspace=workspace.root,
                    query=(
                        "Analyze artifacts/penguins.csv with statistical tests, generate at least "
                        "4 PNG figures, write report.docx, and create presentation.pptx."
                    ),
                    seed_limit=1,
                    expanded_limit=4,
                )
            )
            payload = bundle.to_dict()

            selected_ids = {item["skill_id"] for item in payload["selected_skills"]}
            self.assertEqual(
                selected_ids,
                {"skill:data-visualization", "skill:xlsx", "skill:docx", "skill:pptx"},
            )
            self.assertFalse(
                any(
                    source.startswith("coverage:")
                    for item in payload["selected_skills"]
                    for source in item["sources"]
                )
            )
            self.assertNotIn("task_understanding", payload)
            self.assertNotIn("coverage_diagnostics", payload)
            self.assertTrue(
                any(
                    source.startswith(("interface:", "object:", "execution:"))
                    for item in payload["selected_skills"]
                    for source in item["sources"]
                )
            )

    def test_query_bundle_uses_interface_text_without_task_coverage_parser(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Workspace(Path(tmp) / ".skillfabric")
            workspace.ensure()
            skills = [
                _skill(
                    "skill:analyzing-financial-statements",
                    "analyzing-financial-statements",
                    "Analyze company financial statements and KPI trends.",
                    "Extract key financial ratios and compare year-over-year trends.",
                ),
                _skill(
                    "skill:data-visualization",
                    "data-visualization",
                    "Create charts and PNG figures from analysis results.",
                    "Generate charts and executive-ready visualizations.",
                ),
            ]
            graph = GraphDocument(
                schema_version="1.0",
                build_id="coverage-filter-test",
                nodes=skills,
                edges=[],
                stats={},
                config_digest="coverage-filter-test",
            )
            workspace.write_json(workspace.graph_dir / "graph.json", graph.to_dict())
            workspace.write_jsonl(
                workspace.registry_dir / "skills.jsonl",
                [skill.to_dict(include_raw_text=True) for skill in skills],
            )
            workspace.write_jsonl(
                workspace.interfaces_dir / "skill_interfaces.jsonl",
                [
                    _interface(
                        "skill:analyzing-financial-statements",
                        produces=["financial_kpi_summary"],
                        requires=["financial_statement"],
                        summary="Analyze company financial statements and KPI trends.",
                        when_to_use="Use for financial statement analysis and year-over-year KPI comparisons.",
                    ),
                    _interface(
                        "skill:data-visualization",
                        produces=["image_asset"],
                        summary="Create charts and PNG figures from analysis results.",
                    ),
                ],
            )
            build_bm25_index(skills, workspace.index_dir / "bm25.sqlite")
            for skill in skills:
                page = workspace.wiki_skills_dir / f"{skill.name}.md"
                page.parent.mkdir(parents=True, exist_ok=True)
                page.write_text(f"# {skill.name}\n", encoding="utf-8")

            bundle = build_router_bundle(
                RouterBundleConfig(
                    workspace=workspace.root,
                    query=(
                        "Analyze a company's financial statements, extract key financial KPIs, "
                        "compare year-over-year trends, generate charts, and produce an executive summary report."
                    ),
                    seed_limit=1,
                    expanded_limit=2,
                )
            )
            payload = bundle.to_dict()

            selected = {item["skill_id"]: item for item in payload["selected_skills"]}
            self.assertNotIn("task_understanding", payload)
            self.assertNotIn("coverage_diagnostics", payload)
            self.assertFalse(
                any(
                    source.startswith("coverage:")
                    for item in payload["selected_skills"]
                    for source in item["sources"]
                )
            )
            self.assertIn("skill:analyzing-financial-statements", selected)
            self.assertTrue(
                any(
                    source.startswith(("interface:", "facet:interface_text"))
                    for source in selected["skill:analyzing-financial-statements"]["sources"]
                )
            )

    def test_query_bundle_uses_query_local_graph_and_wiki_context(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Workspace(Path(tmp) / ".skillfabric")
            workspace.ensure()
            parser = _skill(
                "skill:pdf-table-parser",
                "pdf-table-parser",
                "Extract tables from PDF files into CSV tables.",
                "Parse PDF reports and export CSV tables.",
            )
            kpi = _skill(
                "skill:financial-kpi-extractor",
                "financial-kpi-extractor",
                "Extract financial KPIs from CSV tables.",
                "Use CSV tables produced from PDF reports to extract KPIs.",
            )
            writer = _skill(
                "skill:report-writer",
                "report-writer",
                "Write markdown reports from KPI JSON.",
                "Compose a final markdown report from KPI outputs.",
            )
            community = CommunityNode(
                id="community:finance",
                type="community",
                name="Financial Document Skills",
                summary="Skills for parsing financial documents and writing reports.",
                member_count=3,
                representative_skill_ids=[parser.id, kpi.id],
            )
            graph = GraphDocument(
                schema_version="1.0",
                build_id="router-test",
                nodes=[parser, kpi, writer, community],
                edges=[
                    Edge(source=parser.id, target=kpi.id, type="depend_on", confidence=0.97, reason="KPI extraction consumes parsed tables."),
                    Edge(source=kpi.id, target=writer.id, type="depend_on", confidence=0.92, reason="Report writing consumes extracted KPI JSON."),
                    Edge(source=parser.id, target=community.id, type="member_of", confidence=1.0),
                    Edge(source=kpi.id, target=community.id, type="member_of", confidence=1.0),
                    Edge(source=writer.id, target=community.id, type="member_of", confidence=1.0),
                ],
                stats={},
                config_digest="router-test",
            )
            workspace.write_json(workspace.graph_dir / "graph.json", graph.to_dict())
            workspace.write_jsonl(
                workspace.registry_dir / "skills.jsonl",
                [skill.to_dict(include_raw_text=True) for skill in (parser, kpi, writer)],
            )
            build_bm25_index([parser, kpi, writer], workspace.index_dir / "bm25.sqlite")
            build_embedding_store(
                [parser, kpi, writer],
                workspace.index_dir / "embeddings.json",
                provider=FakeEmbeddingProvider(),
            )
            workspace.write_jsonl(
                workspace.execution_dir / "execution_index.jsonl",
                [
                    {
                        "source_skill": parser.id,
                        "target_skill": kpi.id,
                        "relation_type": "artifact_compatibility",
                        "canonical_object": "csv_table",
                        "direction": "source_to_target",
                        "confidence": 0.98,
                        "evidence": [],
                        "projected_edge_type": "depend_on",
                        "reason": "Parsed CSV tables feed KPI extraction.",
                        "metadata": {},
                    },
                    {
                        "source_skill": kpi.id,
                        "target_skill": writer.id,
                        "relation_type": "artifact_compatibility",
                        "canonical_object": "kpi_json",
                        "direction": "source_to_target",
                        "confidence": 0.8,
                        "evidence": [],
                        "projected_edge_type": "depend_on",
                        "reason": "Below default workflow threshold.",
                        "metadata": {},
                    },
                ],
            )
            for path in (
                workspace.wiki_dir / "index.md",
                workspace.wiki_skills_dir / "pdf-table-parser.md",
                workspace.wiki_skills_dir / "financial-kpi-extractor.md",
                workspace.wiki_communities_dir / "finance.md",
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# page\n", encoding="utf-8")
            debug_page = workspace.wiki_dir / "debug" / "raw_artifacts" / "csv-table.md"
            debug_page.parent.mkdir(parents=True, exist_ok=True)
            debug_page.write_text("# debug\n", encoding="utf-8")

            bundle = build_router_bundle(
                RouterBundleConfig(
                    workspace=workspace.root,
                    query="parse pdf tables",
                    seed_limit=1,
                    expanded_limit=3,
                    workflow_confidence_threshold=0.95,
                )
            )
            payload = bundle.to_dict()

            selected_ids = [item["skill_id"] for item in payload["selected_skills"]]
            self.assertIn(parser.id, selected_ids)
            self.assertIn(kpi.id, selected_ids)
            self.assertTrue(any(item["skill_id"] == parser.id and "lexical" in item["sources"] for item in payload["selected_skills"]))
            self.assertTrue(any(item["skill_id"] == kpi.id and "ppr:depend_on" in item["sources"] for item in payload["selected_skills"]))
            self.assertTrue(any(item["skill_id"] == kpi.id and item["ppr_score"] > 0 for item in payload["selected_skills"]))
            self.assertEqual([hint["canonical_object"] for hint in payload["workflow_hints"]], ["csv_table"])
            self.assertFalse(any(page.endswith("wiki/index.md") for page in payload["wiki_pages"]))
            self.assertTrue(any(page.endswith("wiki/skills/pdf-table-parser.md") for page in payload["wiki_pages"]))
            self.assertTrue(any(page.endswith("wiki/communities/finance.md") for page in payload["wiki_pages"]))
            self.assertFalse(any("/debug/" in page for page in payload["wiki_pages"]))

    def test_ppr_recalls_two_hop_skills_and_preserves_depend_on_direction(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Workspace(Path(tmp) / ".skillfabric")
            workspace.ensure()
            goal = _skill("skill:goal", "goal-runner", "Run the exact target task.", "needle_target run the requested task.")
            prereq = _skill("skill:prereq", "prerequisite-maker", "Prepare required state.", "Prepare state.")
            second = _skill("skill:second", "second-hop-helper", "Prepare prerequisite input.", "Support prerequisite.")
            community = CommunityNode(
                id="community:test",
                type="community",
                name="Test Community",
                summary="Membership should not propagate candidates.",
                member_count=3,
                representative_skill_ids=[goal.id],
            )
            graph = GraphDocument(
                schema_version="1.0",
                build_id="ppr-test",
                nodes=[goal, prereq, second, community],
                edges=[
                    Edge(source=goal.id, target=prereq.id, type="depend_on", confidence=1.0),
                    Edge(source=prereq.id, target=second.id, type="compose_with", confidence=1.0),
                    Edge(source=goal.id, target=community.id, type="member_of", confidence=1.0),
                ],
                stats={},
                config_digest="ppr-test",
            )
            workspace.write_json(workspace.graph_dir / "graph.json", graph.to_dict())
            workspace.write_jsonl(
                workspace.registry_dir / "skills.jsonl",
                [skill.to_dict(include_raw_text=True) for skill in (goal, prereq, second)],
            )
            build_bm25_index([goal, prereq, second], workspace.index_dir / "bm25.sqlite")

            bundle = build_router_bundle(
                RouterBundleConfig(
                    workspace=workspace.root,
                    query="needle_target",
                    seed_limit=1,
                    expanded_limit=3,
                    graph_expansion_mode="ppr",
                )
            )
            selected = {item.skill_id: item for item in bundle.selected_skills}

            self.assertIn(goal.id, selected)
            self.assertIn(prereq.id, selected)
            self.assertIn(second.id, selected)
            self.assertIn("ppr:depend_on", selected[prereq.id].sources)
            self.assertIn("ppr:compose_with", selected[second.id].sources)
            self.assertGreater(selected[prereq.id].ppr_score, selected[second.id].ppr_score)
            self.assertFalse(any(source.startswith("ppr:member_of") for item in selected.values() for source in item.sources))

    def test_one_hop_ablation_does_not_recall_two_hop_skill(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Workspace(Path(tmp) / ".skillfabric")
            workspace.ensure()
            goal = _skill("skill:goal", "goal-runner", "Run the exact target task.", "needle_target run the requested task.")
            prereq = _skill("skill:prereq", "prerequisite-maker", "Prepare required state.", "Prepare state.")
            second = _skill("skill:second", "second-hop-helper", "Prepare prerequisite input.", "Support prerequisite.")
            graph = GraphDocument(
                schema_version="1.0",
                build_id="one-hop-test",
                nodes=[goal, prereq, second],
                edges=[
                    Edge(source=goal.id, target=prereq.id, type="depend_on", confidence=1.0),
                    Edge(source=prereq.id, target=second.id, type="compose_with", confidence=1.0),
                ],
                stats={},
                config_digest="one-hop-test",
            )
            workspace.write_json(workspace.graph_dir / "graph.json", graph.to_dict())
            workspace.write_jsonl(
                workspace.registry_dir / "skills.jsonl",
                [skill.to_dict(include_raw_text=True) for skill in (goal, prereq, second)],
            )
            build_bm25_index([goal, prereq, second], workspace.index_dir / "bm25.sqlite")

            bundle = build_router_bundle(
                RouterBundleConfig(
                    workspace=workspace.root,
                    query="needle_target",
                    seed_limit=1,
                    expanded_limit=3,
                    graph_expansion_mode="one_hop",
                )
            )

            selected_ids = {item.skill_id for item in bundle.selected_skills}
            self.assertIn(goal.id, selected_ids)
            self.assertIn(prereq.id, selected_ids)
            self.assertNotIn(second.id, selected_ids)


if __name__ == "__main__":
    unittest.main()
