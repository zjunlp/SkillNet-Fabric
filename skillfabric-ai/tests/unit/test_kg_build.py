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
from skillfabric.compiled_graph.builder import (
    BuildConfig,
    build_graph,
)
from skillfabric.compiled_graph.canonicalization.compiler import (
    DeterministicCanonicalizationProvider,
)
from skillfabric.compiled_graph.execution.validation import DeterministicExecutionFlowValidator
from skillfabric.compiled_graph.health import analyze_health
from skillfabric.compiled_graph.interface.extraction import DeterministicInterfaceExtractor
from skillfabric.compiled_graph.models import Edge, GraphDocument
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
            self.assertFalse((workspace / "graph" / "communities.json").exists())
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
            self.assertEqual(node_types, {"skill"})
            self.assertLessEqual(edge_types, {"similar_to", "compose_with", "depend_on"})
            self.assertNotIn("member_of", edge_types)
            self.assertTrue(all("candidate_sources" not in edge for edge in graph_data["edges"]))
            self.assertTrue(all("raw_output" not in edge for edge in graph_data["edges"]))
            self.assertNotIn("inputs", node_keys)
            self.assertNotIn("outputs", node_keys)
            self.assertNotIn("artifact", node_types)
            self.assertNotIn("scenario", node_types)
            self.assertIn("depend_on", edge_types)
            self.assertGreater(result.stats["execution_projected_edge_count"], 0)
            self.assertTrue(any(edge["provenance"] == "execution_projected" for edge in graph_data["edges"]))
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
            self.assertNotIn("community_count", status)
            self.assertNotIn("community_refinement_model_id", status)
            self.assertNotIn("community_clustering_algorithm", status)
            self.assertNotIn("community_projection_similar_to_count", status)
            self.assertNotIn("community_projection_compose_with_count", status)
            self.assertNotIn("community_projection_depend_on_ignored_count", status)
            self.assertNotIn("community_oversize_split_count", status)
            self.assertNotIn("community_assignment_provenance", status)
            self.assertNotIn("community_assignment_warning", status)
            self.assertNotIn("communities", status["artifacts"])
            self.assertNotIn("community_refinement_cache", status["artifacts"])
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
            self.assertEqual(edge_evidence, [])
            self.assertEqual(result.stats["compose_depend_candidate_count"], 0)
            self.assertEqual(result.stats["compose_depend_edge_count"], 0)
            self.assertEqual(result.stats["relation_validation"]["validator_calls"], 0)

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
        report = analyze_health(graph)

        self.assertTrue(report.depend_on_cycles)
        self.assertEqual(report.edges_missing_evidence, 3)

    def test_graph_document_rejects_removed_or_invalid_schema_fields(self) -> None:
        skill_payload = _health_skill("skill:a", "alpha", "A skill.").to_dict()

        with self.assertRaisesRegex(ValueError, "unsupported node type"):
            GraphDocument.from_dict(
                {
                    "schema_version": "1.0",
                    "build_id": "bad-node",
                    "nodes": [{**skill_payload, "type": "community"}],
                    "edges": [],
                    "stats": {},
                    "config_digest": "x",
                }
            )

        with self.assertRaisesRegex(ValueError, "missing edge type"):
            Edge.from_dict({"source": "skill:a", "target": "skill:b", "confidence": 0.5})

        with self.assertRaisesRegex(ValueError, "unsupported edge type"):
            Edge.from_dict(
                {"source": "skill:a", "target": "skill:b", "type": "member_of", "confidence": 0.5}
            )

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
