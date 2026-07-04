from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from skillfabric.cli import main as cli_main
from skillfabric.compiled_graph.builder import BuildConfig, build_graph
from skillfabric.compiled_graph.canonicalization.compiler import (
    DeterministicCanonicalizationProvider,
)
from skillfabric.compiled_graph.execution.validation import DeterministicExecutionFlowValidator
from skillfabric.compiled_graph.health import analyze_health
from skillfabric.compiled_graph.interface.extraction import DeterministicInterfaceExtractor
from skillfabric.compiled_graph.models import CommunityNode, Edge, GraphDocument
from skillfabric.compiled_graph.relations.validation import StaticPairValidator
from skillfabric.indexing.canonical import canonical_skill_text
from skillfabric.registry.models import SkillNode
from skillfabric.registry.parser import parse_skill_file
from skillfabric.registry.scanner import scan_skill_root
from tests.unit.fake_embeddings import FakeEmbeddingProvider

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_SKILLS = ROOT / "fixtures" / "skills"


def fake_litellm_embedding(**kwargs):
    inputs = kwargs.get("input", [])
    if isinstance(inputs, str):
        inputs = [inputs]
    return {
        "data": [
            {"index": index, "embedding": [1.0, 0.0] if index % 2 == 0 else [0.0, 1.0]}
            for index, _text in enumerate(inputs)
        ]
    }


class StaticCommunityRefinementProvider:
    model_id = "static-community-refiner"

    def refine(self, payload):
        return {
            "name": "Refined Skill Cluster",
            "summary": "Refined community summary.",
            "task_patterns": ["route related skills"],
            "representative_skill_ids": [item["id"] for item in payload["members"][:2]],
        }


def _health_skill(skill_id: str, name: str, description: str) -> SkillNode:
    return SkillNode(
        id=skill_id,
        type="skill",
        name=name,
        description=description,
        source_path=f"/skills/{name}/SKILL.md",
        wiki_path=f"skills/{name}.md",
        content_hash=f"hash-{name}",
        token_count=len(description.split()),
        canonical_skill_text_hash=f"canonical-{name}",
    )


def _health_community(
    community_id: str,
    name: str,
    summary: str,
    *,
    task_patterns: list[str] | None = None,
    member_count: int = 1,
) -> CommunityNode:
    return CommunityNode(
        id=community_id,
        type="community",
        name=name,
        summary=summary,
        member_count=member_count,
        task_patterns=task_patterns or [],
    )


class KGBuildTests(unittest.TestCase):
    def test_parser_uses_frontmatter_without_removed_metadata_field(self) -> None:
        skill = parse_skill_file(FIXTURE_SKILLS / "pdf-table-parser" / "SKILL.md")
        removed_field = "routing_" + "metadata"

        self.assertEqual(skill.id, "skill:pdf-table-parser")
        self.assertEqual(skill.name, "pdf-table-parser")
        self.assertIn("Extract tables", skill.description)
        self.assertIn("pdf-table-parser", canonical_skill_text(skill))
        self.assertFalse(hasattr(skill, removed_field))
        self.assertNotIn(removed_field, skill.to_dict(include_raw_text=True))
        self.assertTrue(skill.content_hash)
        self.assertGreater(skill.token_count, 10)

    def test_parser_falls_back_without_frontmatter(self) -> None:
        skill = parse_skill_file(FIXTURE_SKILLS / "no-frontmatter" / "SKILL.md")

        self.assertEqual(skill.name, "no-frontmatter")
        self.assertIn("Fallback description", skill.description)

    def test_scan_finds_all_skill_documents(self) -> None:
        skills = scan_skill_root(FIXTURE_SKILLS)

        self.assertGreaterEqual(len(skills), 8)
        self.assertEqual(skills, sorted(skills))

    def test_build_graph_outputs_expected_artifacts_and_edges(self) -> None:
        validator = StaticPairValidator(
            {
                ("skill:financial-kpi-extractor", "skill:pdf-table-parser"): {
                    "edge_type": "depend_on",
                    "direction": "A->B",
                    "confidence": 0.92,
                    "evidence": [
                        {
                            "skill": "skill:financial-kpi-extractor",
                            "line": 7,
                            "text": "Use this after `pdf-table-parser` has produced `.csv` tables.",
                        }
                    ],
                    "reason": "KPI extraction consumes CSV tables produced by PDF table parsing.",
                },
                ("skill:report-writer", "skill:financial-kpi-extractor"): {
                    "edge_type": "depend_on",
                    "direction": "A->B",
                    "confidence": 0.9,
                    "evidence": [
                        {
                            "skill": "skill:report-writer",
                            "line": 5,
                            "text": "Use KPI JSON and chart artifacts to compose a final `.md` report.",
                        }
                    ],
                    "reason": "Report writing consumes KPI JSON.",
                },
                ("skill:webshop-product-evaluator", "skill:webshop-product-search"): {
                    "edge_type": "depend_on",
                    "direction": "A->B",
                    "confidence": 0.91,
                    "evidence": [
                        {
                            "skill": "skill:webshop-product-evaluator",
                            "line": 5,
                            "text": "Use after `webshop-product-search` returns candidates.",
                        }
                    ],
                    "reason": "Evaluation consumes search candidates.",
                },
                ("skill:testing-python", "skill:analyze-ci"): {
                    "edge_type": "compose_with",
                    "direction": "undirected",
                    "confidence": 0.91,
                    "evidence": [
                        {
                            "skill": "skill:testing-python",
                            "line": 6,
                            "text": "This skill composes with `analyze-ci` when a CI job fails.",
                        }
                    ],
                    "reason": "CI analysis and focused pytest diagnosis are commonly chained.",
                },
            }
        )

        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            result = build_graph(
                BuildConfig(
                    skill_root=FIXTURE_SKILLS,
                    workspace=workspace,
                    similar_top_k=3,
                    candidate_top_k=6,
                    validator=validator,
                    canonicalization_provider=DeterministicCanonicalizationProvider(),
                    interface_extractor=DeterministicInterfaceExtractor(),
                    execution_validator=DeterministicExecutionFlowValidator(),
                    embedding_provider=FakeEmbeddingProvider(),
                    build_id="test-build",
                )
            )

            self.assertEqual(result.graph.schema_version, "1.0")
            self.assertEqual(result.stats["skill_count"], 8)
            self.assertTrue((workspace / "graph" / "registry.jsonl").exists())
            self.assertTrue((workspace / "graph" / "skill_sources.jsonl").exists())
            self.assertTrue((workspace / "graph" / "bm25.sqlite").exists())
            self.assertTrue((workspace / "graph" / "embeddings.json").exists())
            self.assertTrue((workspace / "graph" / "embedding_meta.jsonl").exists())
            self.assertTrue((workspace / "graph" / "graph.json").exists())
            self.assertTrue((workspace / "graph" / "communities.json").exists())
            self.assertTrue((workspace / "graph" / "edge_evidence.jsonl").exists())
            self.assertFalse((workspace / "graph" / "relation_validation_audit.jsonl").exists())
            self.assertTrue((workspace / "graph" / "relation_validation_summary.json").exists())
            self.assertTrue((workspace / "graph" / "graph_health_report.md").exists())
            self.assertTrue((workspace / "graph" / "contracts.jsonl").exists())
            self.assertTrue((workspace / "graph" / "interface_evidence.jsonl").exists())
            self.assertTrue((workspace / "graph" / "interface_health_report.md").exists())
            self.assertTrue((workspace / "graph" / "canonical_objects.jsonl").exists())
            self.assertTrue((workspace / "graph" / "canonical_aliases.jsonl").exists())
            self.assertFalse((workspace / "graph" / "canonicalization_evidence.jsonl").exists())
            self.assertTrue((workspace / "graph" / "canonicalization_health_report.md").exists())
            self.assertTrue((workspace / "cache" / "interface_cache.json").exists())
            self.assertTrue((workspace / "cache" / "canonicalization_cache.json").exists())
            self.assertTrue((workspace / "graph" / "execution_index.jsonl").exists())
            self.assertFalse((workspace / "execution_graph").exists())
            self.assertFalse((workspace / "interfaces").exists())
            self.assertFalse((workspace / "registry").exists())
            self.assertFalse((workspace / "index").exists())
            self.assertTrue((workspace / "graph" / "execution_evidence.jsonl").exists())
            self.assertFalse((workspace / "graph" / "execution_validation_audit.jsonl").exists())
            self.assertTrue((workspace / "graph" / "execution_validation_summary.json").exists())
            self.assertTrue((workspace / "graph" / "execution_health_report.md").exists())
            self.assertTrue((workspace / "graph" / "compiled.json").exists())
            self.assertTrue((workspace / "reports" / "build_summary.json").exists())
            self.assertEqual(result.stats["interface_count"], 8)
            self.assertEqual(len(result.interfaces), 8)
            self.assertGreater(result.stats["execution_candidate_count"], 0)
            self.assertGreater(result.stats["execution_accepted_flow_count"], 0)
            self.assertIn("relation_validation", result.stats)
            self.assertIn("execution_validation", result.stats)
            self.assertIn("stage_wall_time_seconds", result.stats)

            graph_data = json.loads((workspace / "graph" / "graph.json").read_text())
            node_types = {node["type"] for node in graph_data["nodes"]}
            edge_types = {edge["type"] for edge in graph_data["edges"]}
            node_keys = set().union(*(node.keys() for node in graph_data["nodes"]))
            self.assertEqual(node_types, {"skill", "community"})
            self.assertLessEqual(edge_types, {"similar_to", "member_of", "compose_with", "depend_on"})
            self.assertTrue(all("candidate_sources" not in edge for edge in graph_data["edges"]))
            self.assertTrue(all("raw_output" not in edge for edge in graph_data["edges"]))
            self.assertNotIn("inputs", node_keys)
            self.assertNotIn("outputs", node_keys)
            self.assertNotIn("artifact", node_types)
            self.assertNotIn("scenario", node_types)
            self.assertIn("depend_on", edge_types)
            self.assertIn("compose_with", edge_types)
            self.assertGreater(result.stats["execution_projected_edge_count"], 0)
            self.assertTrue(any(edge["provenance"] == "deterministic_accept" for edge in graph_data["edges"]))
            compiled_graph = json.loads((workspace / "graph" / "compiled.json").read_text())
            self.assertIn("core_graph", compiled_graph)
            self.assertIn("interfaces", compiled_graph)
            self.assertIn("canonicalization", compiled_graph)
            self.assertIn("execution_graph", compiled_graph)
            self.assertTrue(compiled_graph["canonicalization"]["objects"])
            self.assertTrue(compiled_graph["canonicalization"]["aliases"])
            execution_index = compiled_graph["execution_graph"]["execution_index"]
            self.assertEqual(len(execution_index), result.stats["execution_accepted_flow_count"])
            self.assertTrue(all(row["projected_edge_type"] in {"depend_on", "compose_with"} for row in execution_index))
            build_metrics = json.loads((workspace / "reports" / "build_summary.json").read_text())
            self.assertEqual(build_metrics["skill_count"], 8)
            self.assertIn("relation_validation", build_metrics)
            self.assertIn("execution_validation", build_metrics)
            self.assertIn("policy_digest", build_metrics["relation_validation"])
            self.assertIn("policy_digest", build_metrics["execution_validation"])
            self.assertIn("llm_usage", build_metrics)
            self.assertIn("embedding", build_metrics)
            self.assertEqual(build_metrics["embedding"]["model_id"], "test-fake-embedding")
            self.assertGreater(build_metrics["embedding"]["estimated_input_tokens"], 0)
            self.assertTrue(all(row["canonical_object"] for row in execution_index))
            self.assertNotIn("artifact_nodes", compiled_graph["execution_graph"])
            self.assertNotIn("scenario_nodes", compiled_graph["execution_graph"])
            self.assertNotIn("skill_artifact_edges", compiled_graph["execution_graph"])
            self.assertNotIn("skill_scenario_edges", compiled_graph["execution_graph"])
            status = json.loads((workspace / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["canonical_object_count"], result.stats["canonical_object_count"])
            self.assertEqual(status["execution_compatibility_count"], result.stats["execution_compatibility_count"])
            self.assertIn("community_clustering_algorithm", status)
            self.assertIn("community_projection_similar_to_count", status)
            self.assertIn("community_projection_compose_with_count", status)
            self.assertIn("community_projection_depend_on_ignored_count", status)
            self.assertIn("community_oversize_split_count", status)
            self.assertNotIn("community_assignment_provenance", status)
            self.assertNotIn("community_assignment_warning", status)
            self.assertNotIn("artifact_node_count", status)
            self.assertNotIn("scenario_node_count", status)
            self.assertNotIn("artifact_nodes", status["artifacts"])
            self.assertNotIn("scenario_nodes", status["artifacts"])
            self.assertNotIn("skill_artifact_edges", status["artifacts"])
            self.assertNotIn("skill_scenario_edges", status["artifacts"])
            edge_evidence = [
                json.loads(line)
                for line in (workspace / "graph" / "edge_evidence.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(edge_evidence)
            self.assertTrue(any(row["accepted"] for row in edge_evidence))
            self.assertTrue(any(row["candidate_evidence"] for row in edge_evidence))
            self.assertTrue(any("rejection_reason" in row for row in edge_evidence))

            neighbors = result.neighbors("skill:financial-kpi-extractor")
            neighbor_ids = {item["skill_id"] for item in neighbors}
            self.assertIn("skill:pdf-table-parser", neighbor_ids)

            stale_obsolete_artifact = workspace / "graph" / "artifact_nodes.jsonl"
            stale_obsolete_artifact.write_text("{}\n", encoding="utf-8")
            stale_predicate_inventory = workspace / "graph" / "predicate_inventory.json"
            stale_predicate_inventory.write_text("{}\n", encoding="utf-8")
            stale_workflow_compatibility = workspace / "graph" / "workflow_compatibility.jsonl"
            stale_workflow_compatibility.write_text("{}\n", encoding="utf-8")

            second = build_graph(
                BuildConfig(
                    skill_root=FIXTURE_SKILLS,
                    workspace=workspace,
                    similar_top_k=3,
                    candidate_top_k=6,
                    validator=validator,
                    canonicalization_provider=DeterministicCanonicalizationProvider(),
                    interface_extractor=DeterministicInterfaceExtractor(),
                    execution_validator=DeterministicExecutionFlowValidator(),
                    embedding_provider=FakeEmbeddingProvider(),
                    build_id="test-build-2",
                )
            )
            self.assertEqual(second.stats["skipped_unchanged"], 8)
            self.assertFalse(stale_obsolete_artifact.exists())
            self.assertFalse(stale_predicate_inventory.exists())
            self.assertFalse(stale_workflow_compatibility.exists())

    def test_build_graph_writes_refined_community_metadata(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            result = build_graph(
                BuildConfig(
                    skill_root=FIXTURE_SKILLS,
                    workspace=workspace,
                    similar_top_k=3,
                    candidate_top_k=6,
                    validator=StaticPairValidator({}),
                    canonicalization_provider=DeterministicCanonicalizationProvider(),
                    interface_extractor=DeterministicInterfaceExtractor(),
                    execution_validator=DeterministicExecutionFlowValidator(),
                    community_refinement_provider=StaticCommunityRefinementProvider(),
                    embedding_provider=FakeEmbeddingProvider(),
                    build_id="community-refined-build",
                )
            )

            community_payload = json.loads((workspace / "graph" / "communities.json").read_text(encoding="utf-8"))
            first = community_payload["communities"][0]

            self.assertTrue(result.communities)
            self.assertEqual(first["community"]["name"], "Refined Skill Cluster")
            self.assertEqual(first["community"]["summary_provenance"], "llm_refined")
            self.assertEqual(first["community"]["model_id"], "static-community-refiner")
            self.assertEqual(first["community"]["task_patterns"], ["route related skills"])
            self.assertEqual(first["summary_provenance"], "llm_refined")

    def test_health_report_detects_cycles_and_missing_evidence(self) -> None:
        graph = GraphDocument(
            schema_version="1.0",
            build_id="health",
            nodes=[],
            edges=[
                Edge(source="skill:a", target="skill:b", type="depend_on", confidence=0.9),
                Edge(source="skill:b", target="skill:a", type="depend_on", confidence=0.9),
                Edge(source="skill:a", target="skill:c", type="compose_with", confidence=0.8),
            ],
            stats={},
            config_digest="x",
        )

        report = analyze_health(graph, communities=[])

        self.assertTrue(report.depend_on_cycles)
        self.assertEqual(report.edges_missing_evidence, 3)

    def test_health_report_detects_community_text_outlier(self) -> None:
        skill = _health_skill(
            "skill:data-storytelling",
            "data-storytelling",
            "Analyze tabular data and explain statistical findings through charts and analysis narratives.",
        )
        brand = _health_community(
            "community:brand",
            "Brand and Content Styling",
            "Brand copy, visual tone, and social campaign calendars.",
            task_patterns=["draft marketing posts"],
        )
        analytics = _health_community(
            "community:data",
            "Data Analytics",
            "Statistical analysis, spreadsheet data, charts, and analytical reporting.",
            task_patterns=["analyze tabular data"],
        )
        graph = GraphDocument(
            schema_version="1.0",
            build_id="community-text-health",
            nodes=[skill, brand, analytics],
            edges=[
                Edge(
                    source="skill:data-storytelling",
                    target="community:brand",
                    type="member_of",
                    confidence=1.0,
                )
            ],
            stats={},
            config_digest="x",
        )

        report = analyze_health(graph, communities=[brand, analytics])

        self.assertEqual(len(report.community_text_outliers), 1)
        self.assertEqual(report.community_text_outliers[0].skill_id, "skill:data-storytelling")
        self.assertEqual(report.community_text_outliers[0].suggested_community_id, "community:data")

    def test_health_report_ignores_generic_creator_token_for_community_text(self) -> None:
        skill = _health_skill(
            "skill:skill-creator",
            "skill-creator",
            "Create reusable skill documentation and agent instructions for prompt workflows.",
        )
        assigned = _health_community(
            "community:prompt",
            "Content and Prompt Operations",
            "Prompt engineering, reusable documentation, agent instructions, and conflict resolution.",
            task_patterns=["write reusable prompts"],
        )
        media = _health_community(
            "community:media",
            "Media Processing",
            "Slack gif creator workflows, video media production, animation, and color grading.",
            task_patterns=["generate gifs"],
        )
        graph = GraphDocument(
            schema_version="1.0",
            build_id="community-text-health",
            nodes=[skill, assigned, media],
            edges=[
                Edge(
                    source="skill:skill-creator",
                    target="community:prompt",
                    type="member_of",
                    confidence=1.0,
                )
            ],
            stats={},
            config_digest="x",
        )

        report = analyze_health(graph, communities=[assigned, media])

        self.assertFalse(report.community_text_outliers)

    def test_health_report_ignores_generic_review_and_sharing_tokens(self) -> None:
        skill = _health_skill(
            "skill:code-review-excellence",
            "code-review-excellence",
            "Review code changes, pull requests, tests, design risks, and engineering feedback quality.",
        )
        assigned = _health_community(
            "community:workflow",
            "Workflow Planning",
            "Task tracking, requirements shaping, prompt design, and process coordination.",
            task_patterns=["coordinate review process"],
        )
        presentation = _health_community(
            "community:presentation",
            "Presentation Artifacts",
            "Slide decks and publication-style visual deliverables packaged for sharing and review.",
            task_patterns=["package information for sharing"],
        )
        graph = GraphDocument(
            schema_version="1.0",
            build_id="community-text-health",
            nodes=[skill, assigned, presentation],
            edges=[
                Edge(
                    source="skill:code-review-excellence",
                    target="community:workflow",
                    type="member_of",
                    confidence=1.0,
                )
            ],
            stats={},
            config_digest="x",
        )

        report = analyze_health(graph, communities=[assigned, presentation])

        self.assertFalse(report.community_text_outliers)

    def test_health_report_detects_weak_cross_community_compose_edges(self) -> None:
        visual = _health_community(
            "community:visual",
            "Visual Artifact Studio",
            "Image, canvas, and presentation artifact workflows.",
        )
        data = _health_community(
            "community:data",
            "Data Analytics",
            "Statistics, spreadsheets, charts, and data reporting.",
        )
        graph = GraphDocument(
            schema_version="1.0",
            build_id="weak-cross-community-edge-health",
            nodes=[
                _health_skill("skill:canvas-design", "canvas-design", "Design visual canvas artifacts."),
                _health_skill("skill:statistical-analysis", "statistical-analysis", "Run statistical analysis."),
                visual,
                data,
            ],
            edges=[
                Edge("skill:canvas-design", "community:visual", "member_of", confidence=1.0),
                Edge("skill:statistical-analysis", "community:data", "member_of", confidence=1.0),
                Edge(
                    "skill:canvas-design",
                    "skill:statistical-analysis",
                    "compose_with",
                    confidence=0.86,
                    provenance="llm_validated",
                ),
                Edge(
                    "skill:statistical-analysis",
                    "skill:canvas-design",
                    "depend_on",
                    confidence=0.86,
                    provenance="llm_validated",
                ),
            ],
            stats={},
            config_digest="x",
        )

        report = analyze_health(graph, communities=[visual, data])

        self.assertEqual(len(report.weak_cross_community_compose_edges), 1)
        self.assertEqual(report.weak_cross_community_compose_edges[0].source, "skill:canvas-design")
        self.assertEqual(report.weak_cross_community_compose_edges[0].target, "skill:statistical-analysis")

    def test_health_report_detects_low_cohesion_large_community(self) -> None:
        large = _health_community(
            "community:mixed",
            "Mixed Web Data Integration",
            "A broad mix of APIs, browser automation, media processing, and reporting.",
            member_count=15,
        )
        large.cohesion_score = 0.16
        focused = _health_community(
            "community:docs",
            "Document Studio",
            "Document and presentation authoring.",
            member_count=6,
        )
        focused.cohesion_score = 0.6
        small_sparse = _health_community(
            "community:media",
            "Media Production",
            "Media conversion utilities.",
            member_count=3,
        )
        small_sparse.cohesion_score = 0.1

        report = analyze_health(
            GraphDocument(
                schema_version="1.0",
                build_id="low-cohesion-large-community",
                nodes=[large, focused, small_sparse],
                edges=[],
                stats={},
                config_digest="x",
            ),
            communities=[large, focused, small_sparse],
        )

        self.assertEqual(len(report.low_cohesion_large_communities), 1)
        self.assertEqual(report.low_cohesion_large_communities[0].community_id, "community:mixed")

    def test_public_cli_build_writes_core_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                cli_main(
                    [
                        "build",
                        "--skill-root",
                        str(FIXTURE_SKILLS),
                        "--workspace",
                        str(workspace),
                        "--skip-llm-validation",
                        "--embedding-provider",
                        "disabled",
                    ]
                )

            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["skill_count"], 8)
            self.assertTrue((workspace / "graph" / "registry.jsonl").exists())
            self.assertTrue((workspace / "graph" / "compiled.json").exists())
            self.assertTrue((workspace / "status.json").exists())
            self.assertTrue((workspace / "wiki" / "index.md").exists())
            self.assertIn("wiki", payload["artifacts"])

    def test_build_uses_litellm_validator_by_default_and_reuses_cache(self) -> None:
        calls: list[dict[str, object]] = []
        fake_litellm = types.SimpleNamespace()

        def fake_completion(**kwargs):
            calls.append(kwargs)
            messages = kwargs.get("messages", [])
            user_content = messages[-1].get("content", "") if messages else ""
            if "execution-aware SkillContract" in user_content:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "capability_summary": "Interface extracted for cache testing.",
                                        "uses_tools": [],
                                    }
                                )
                            }
                        }
                    ]
                }
            if "Validate whether an execution-level flow exists" in user_content:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "accepted": False,
                                        "flow_type": "none",
                                        "projected_edge_type": "none",
                                        "confidence": 0.0,
                                        "evidence": [],
                                        "reason": "No execution flow.",
                                    }
                                )
                            }
                        }
                    ]
                }
            if "Canonicalize a precluster of SkillContract" in user_content:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "canonical_objects": [],
                                        "assignments": [],
                                        "rejected_terms": [],
                                    }
                                )
                            }
                        }
                    ]
                }
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "edge_type": "none",
                                    "direction": "none",
                                    "confidence": 0.0,
                                    "evidence": [],
                                    "reason": "No relation.",
                                }
                            )
                        }
                    }
                ]
            }

        fake_litellm.completion = fake_completion

        fake_litellm.embedding = fake_litellm_embedding
        original = sys.modules.get("litellm")
        sys.modules["litellm"] = fake_litellm
        try:
            with TemporaryDirectory() as tmp:
                workspace = Path(tmp) / ".skillfabric"
                env_path = Path(tmp) / ".env"
                env_path.write_text(
                    "\n".join(
                        [
                            "BASE_URL=https://example.test/api",
                            "API_KEY=sk-test",
                            "MODEL=openai/test-model",
                            "EMBEDDING_MODEL=openai/test-embedding",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )

                build_graph(
                    BuildConfig(
                        skill_root=FIXTURE_SKILLS,
                        workspace=workspace,
                        similar_top_k=1,
                        candidate_top_k=2,
                        embedding_provider=FakeEmbeddingProvider(),
                        llm_env_path=env_path,
                        build_id="llm-build-1",
                    )
                )
                first_call_count = len(calls)
                self.assertGreater(first_call_count, 0)

                build_graph(
                    BuildConfig(
                        skill_root=FIXTURE_SKILLS,
                        workspace=workspace,
                        similar_top_k=1,
                        candidate_top_k=2,
                        embedding_provider=FakeEmbeddingProvider(),
                        llm_env_path=env_path,
                        build_id="llm-build-2",
                    )
                )
                self.assertEqual(len(calls), first_call_count)
        finally:
            if original is None:
                sys.modules.pop("litellm", None)
            else:
                sys.modules["litellm"] = original

    def test_cli_build_accepts_env_file_for_default_litellm(self) -> None:
        fake_litellm = types.SimpleNamespace()

        def fake_completion(**kwargs):
            messages = kwargs.get("messages", [])
            user_content = messages[-1].get("content", "") if messages else ""
            if "execution-aware SkillContract" in user_content:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "capability_summary": "Interface extracted for CLI test.",
                                        "uses_tools": [],
                                    }
                                )
                            }
                        }
                    ]
                }
            if "Validate whether an execution-level flow exists" in user_content:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "accepted": False,
                                        "flow_type": "none",
                                        "projected_edge_type": "none",
                                        "confidence": 0.0,
                                        "evidence": [],
                                        "reason": "No execution flow.",
                                    }
                                )
                            }
                        }
                    ]
                }
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "edge_type": "none",
                                    "direction": "none",
                                    "confidence": 0.0,
                                    "evidence": [],
                                    "reason": "No relation.",
                                }
                            )
                        }
                    }
                ]
            }

        fake_litellm.completion = fake_completion

        fake_litellm.embedding = fake_litellm_embedding
        original = sys.modules.get("litellm")
        sys.modules["litellm"] = fake_litellm
        try:
            with TemporaryDirectory() as tmp:
                workspace = Path(tmp) / ".skillfabric"
                env_path = Path(tmp) / ".env"
                env_path.write_text(
                    "BASE_URL=https://example.test/api\n"
                    "API_KEY=sk-test\n"
                    "MODEL=openai/test-model\n"
                    "EMBEDDING_MODEL=openai/test-embedding\n",
                    encoding="utf-8",
                )
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    cli_main(
                        [
                            "build",
                            "--skill-root",
                            str(FIXTURE_SKILLS),
                            "--workspace",
                            str(workspace),
                            "--similar-top-k",
                            "1",
                            "--candidate-top-k",
                            "1",
                            "--env-file",
                            str(env_path),
                            "--skip-wiki",
                        ]
                    )
                payload = json.loads(stdout.getvalue())
                self.assertEqual(payload["skill_count"], 8)
                self.assertIn("graph", payload["artifacts"])
        finally:
            if original is None:
                sys.modules.pop("litellm", None)
            else:
                sys.modules["litellm"] = original

    def test_cli_build_without_env_file_fails_fast_when_llm_enabled(self) -> None:
        with TemporaryDirectory() as tmp:
            missing_env = Path(tmp) / "missing.env"
            cleared_llm_env = {
                "API_KEY": "",
                "BASE_URL": "",
                "MODEL": "",
                "EMBEDDING_MODEL": "",
                "OPENAI_API_BASE": "",
                "OPENAI_BASE_URL": "",
                "OPENAI_API_KEY": "",
            }
            with patch.dict(os.environ, cleared_llm_env, clear=False):
                with self.assertRaisesRegex(SystemExit, "missing API configuration"):
                    cli_main(
                        [
                            "build",
                            "--skill-root",
                            str(FIXTURE_SKILLS),
                            "--workspace",
                            str(Path(tmp) / ".skillfabric"),
                            "--env-file",
                            str(missing_env),
                        ]
                    )


if __name__ == "__main__":
    unittest.main()
