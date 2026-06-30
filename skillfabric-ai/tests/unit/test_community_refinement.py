from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from skillfabric.compiled_graph.communities.assignment import assign_final_communities
from skillfabric.compiled_graph.communities.clustering import cluster_communities
from skillfabric.compiled_graph.communities.providers import LiteLLMCommunityRefinementProvider
from skillfabric.compiled_graph.communities.refinement import refine_communities
from skillfabric.compiled_graph.interface.models import InterfaceField, SkillInterface
from skillfabric.compiled_graph.models import Edge
from skillfabric.llm import LLMConfig
from tests.unit.relation_helpers import make_skill


class CountingRefinementProvider:
    model_id = "community-refinement-test-model"

    def __init__(self) -> None:
        self.calls = 0
        self.payloads: list[dict] = []

    def refine(self, payload):
        self.calls += 1
        self.payloads.append(payload)
        member_ids = [item["id"] for item in payload["members"]]
        return {
            "name": "Refined Routing Boundary",
            "summary": "A refined graph-derived routing community.",
            "task_patterns": ["route graph-derived skills"],
            "representative_skill_ids": member_ids[:1],
        }

    def assign(self, _payload):
        raise AssertionError("global community assignment must not be called")


class CommunityRefinementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_litellm = sys.modules.get("litellm")

    def tearDown(self) -> None:
        if self.original_litellm is None:
            sys.modules.pop("litellm", None)
        else:
            sys.modules["litellm"] = self.original_litellm

    def test_litellm_refinement_prompt_uses_graph_first_routing_boundaries(self) -> None:
        calls = self._install_fake_litellm(
            {
                "name": "Document Formats",
                "summary": "Office artifact generation.",
                "task_patterns": ["create office files"],
                "representative_skill_ids": ["skill:docx"],
            }
        )
        provider = LiteLLMCommunityRefinementProvider(
            LLMConfig(api_base="https://example.test/api", api_key="sk-test")
        )

        provider.refine(
            {
                "community": {"id": "community:docs", "member_count": 1},
                "members": [{"id": "skill:docx", "name": "docx"}],
                "top_internal_edges": [],
                "selected_boundary_edges": [],
                "projection_edge_evidence": [],
                "cohesion_score": 1.0,
            }
        )

        payload = json.loads(calls[0]["messages"][1]["content"])
        prompt_text = json.dumps(calls[0]["messages"], ensure_ascii=False)
        self.assertEqual(payload["prompt_id"], "community_refinement_graph_routing_boundaries")
        for field in ("todo", "input", "output", "workflow", "rules", "constraints", "payload"):
            self.assertIn(field, payload)
        self.assertIn("Do not change community membership", prompt_text)
        self.assertIn("graph-topology", prompt_text)
        self.assertIn("routing boundary", prompt_text)
        self.assertIn("alternatives", prompt_text)
        self.assertIn("workflow stages", prompt_text)
        self.assertIn("validated relation edges", prompt_text)
        self.assertIn("projection_edge_evidence", prompt_text)

    def test_refinement_payload_includes_interfaces_and_boundary_edges(self) -> None:
        skills = [
            make_skill("skill:pdf", "pdf", "Parse PDFs."),
            make_skill("skill:xlsx", "xlsx", "Write spreadsheets."),
            make_skill("skill:slides", "slides", "Create slides."),
        ]
        internal = Edge("skill:pdf", "skill:xlsx", "similar_to", confidence=0.9, weight=0.9)
        boundary = Edge("skill:slides", "skill:xlsx", "depend_on", confidence=0.99, weight=0.99)
        communities, _member_edges, membership, _stats = cluster_communities(
            skills,
            [internal],
            [boundary],
        )
        provider = CountingRefinementProvider()
        interfaces = {
            "skill:pdf": SkillInterface(
                skill_id="skill:pdf",
                content_hash="hash-pdf",
                capability_summary="Extract table artifacts from PDF files.",
                execution_role="producer",
                produces=[InterfaceField(name="tables", kind="artifact", confidence=0.9)],
            )
        }

        refine_communities(
            communities,
            skills,
            [internal, boundary],
            membership,
            provider=provider,
            interfaces=interfaces,
        )

        payload = next(item for item in provider.payloads if any(member["id"] == "skill:pdf" for member in item["members"]))
        self.assertEqual(payload["members"][0]["capability_summary"], "Extract table artifacts from PDF files.")
        self.assertTrue(payload["top_internal_edges"])
        self.assertTrue(payload["selected_boundary_edges"])
        self.assertEqual(payload["projection_edge_evidence"][0]["edge_type"], "similar_to")
        self.assertEqual(payload["projection_edge_evidence"][0]["membership_role"], "strong")

    def test_assign_final_communities_clusters_then_refines_without_global_assignment(self) -> None:
        skills = [
            make_skill("skill:pdf", "pdf", "Parse PDFs."),
            make_skill("skill:xlsx", "xlsx", "Write spreadsheets."),
            make_skill("skill:audio", "audio", "Transcribe audio."),
        ]
        provider = CountingRefinementProvider()

        with TemporaryDirectory() as tmp:
            communities, member_edges, membership, stats = assign_final_communities(
                skills,
                provider=provider,
                refinement_cache_path=Path(tmp) / "community_refinement_cache.json",
                similar_edges=[Edge("skill:pdf", "skill:xlsx", "similar_to", confidence=0.9, weight=0.9)],
                relation_edges=[
                    Edge("skill:xlsx", "skill:audio", "depend_on", confidence=0.99, weight=0.99)
                ],
            )

        self.assertGreaterEqual(len(communities), 2)
        self.assertEqual(len(member_edges), len(skills))
        self.assertEqual(set(membership), {skill.id for skill in skills})
        self.assertEqual(provider.calls, len(communities))
        self.assertEqual(stats["community_refinement_model_id"], provider.model_id)
        self.assertIn(stats["community_clustering_algorithm"], {"leiden", "singletons"})
        self.assertNotIn("community_assignment_provenance", stats)
        self.assertNotIn("community_assignment_warning", stats)

    def _install_fake_litellm(self, payload: dict) -> list[dict]:
        calls: list[dict] = []
        fake_litellm = types.ModuleType("litellm")

        def fake_completion(**kwargs):
            calls.append(kwargs)
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(payload),
                        }
                    }
                ]
            }

        fake_litellm.completion = fake_completion
        sys.modules["litellm"] = fake_litellm
        return calls


if __name__ == "__main__":
    unittest.main()
