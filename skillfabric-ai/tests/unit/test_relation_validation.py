from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from skillfabric.compiled_graph.interface.models import InterfaceField, SkillInterface
from skillfabric.compiled_graph.relations.models import CandidatePair, RelationEvidence
from skillfabric.compiled_graph.relations.prompts import (
    build_compact_pair_validation_messages,
    build_pair_validation_messages,
)
from skillfabric.compiled_graph.relations.validation import (
    LiteLLMPairValidator,
    StaticPairValidator,
    relation_validation_audit_rows,
    summarize_relation_validation_records,
    validate_relation_candidates,
)
from skillfabric.runtime.jobs import LLMJobOptions
from skillfabric.runtime.llm import LLMConfig
from tests.unit.relation_helpers import make_skill


class RelationValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_litellm = sys.modules.get("litellm")

    def tearDown(self) -> None:
        if self.original_litellm is None:
            sys.modules.pop("litellm", None)
        else:
            sys.modules["litellm"] = self.original_litellm

    def _install_fake_litellm(self, content: str | Exception) -> list[dict[str, object]]:
        calls: list[dict[str, object]] = []
        fake_litellm = types.SimpleNamespace()

        def fake_completion(**kwargs):
            calls.append(kwargs)
            if isinstance(content, Exception):
                raise content
            return {"choices": [{"message": {"content": content}}]}

        fake_litellm.completion = fake_completion
        sys.modules["litellm"] = fake_litellm
        return calls

    def _install_fake_litellm_sequence(self, contents: list[str]) -> list[dict[str, object]]:
        calls: list[dict[str, object]] = []
        fake_litellm = types.SimpleNamespace()

        def fake_completion(**kwargs):
            calls.append(kwargs)
            content = contents[min(len(calls) - 1, len(contents) - 1)]
            return {"choices": [{"message": {"content": content}}]}

        fake_litellm.completion = fake_completion
        sys.modules["litellm"] = fake_litellm
        return calls

    def test_prompt_contains_full_skill_candidate_evidence_and_direction_hint(self) -> None:
        skill_a = make_skill("skill:a", "alpha", "A produces CSV.\nFULL_A")
        skill_b = make_skill("skill:b", "beta", "B requires CSV.\nFULL_B")
        pair = CandidatePair(
            "skill:a",
            "skill:b",
            1.0,
            sources=["explicit_mention"],
            evidence=[RelationEvidence("explicit_mention", "skill:b", 1, "B requires alpha.", "mention")],
            direction_hint="B->A",
        )

        messages = build_pair_validation_messages(skill_a, skill_b, pair)
        prompt_text = json.dumps(messages, ensure_ascii=False)
        payload = json.loads(messages[1]["content"])

        self.assertIn("candidate_evidence", prompt_text)
        self.assertIn("direction_hint", prompt_text)
        self.assertIn("FULL_A", prompt_text)
        self.assertIn("FULL_B", prompt_text)
        self.assertEqual(payload["prompt_id"], "relation_validation_low_redundancy_v2")
        for field in ("todo", "input", "output", "workflow", "rules", "constraints"):
            self.assertIn(field, payload)
        self.assertIn("decision_workflow", payload)
        self.assertIn("low_redundancy_goal", payload["edge_semantics"])
        self.assertIn("false positive edges are worse", prompt_text)
        self.assertIn("distinct roles or stages", prompt_text)
        self.assertIn("redundant alternatives", prompt_text)

    def test_compact_prompt_uses_interfaces_and_evidence_without_full_skill_md(self) -> None:
        skill_a = make_skill("skill:a", "alpha", "A produces CSV.\nFULL_A_SHOULD_NOT_APPEAR")
        skill_b = make_skill("skill:b", "beta", "B consumes CSV.\nFULL_B_SHOULD_NOT_APPEAR")
        pair = CandidatePair(
            "skill:a",
            "skill:b",
            0.86,
            sources=["interface_compatibility"],
            evidence=[
                RelationEvidence(
                    "interface_compatibility",
                    "skill:a",
                    1,
                    "A produces CSV.",
                    "interface",
                    metadata={"field_role": "producer_field", "field_name": "csv"},
                ),
                RelationEvidence(
                    "interface_compatibility",
                    "skill:b",
                    1,
                    "B consumes CSV.",
                    "interface",
                    metadata={"field_role": "consumer_field", "field_name": "csv"},
                ),
            ],
            direction_hint="B->A",
        )
        interfaces = {
            "skill:a": SkillInterface(
                skill_id="skill:a",
                content_hash="hash-a",
                capability_summary="Produces CSV artifacts.",
                when_to_use="Use when CSV output is needed.",
            ),
            "skill:b": SkillInterface(
                skill_id="skill:b",
                content_hash="hash-b",
                capability_summary="Consumes CSV artifacts.",
                when_to_use="Use after CSV is available.",
            ),
        }

        messages = build_compact_pair_validation_messages(skill_a, skill_b, pair, interfaces=interfaces)
        prompt_text = json.dumps(messages, ensure_ascii=False)
        payload = json.loads(messages[1]["content"])

        self.assertIn("skill_interface", prompt_text)
        self.assertIn("candidate_evidence", prompt_text)
        self.assertNotIn("full_skill_md", prompt_text)
        self.assertNotIn("FULL_A_SHOULD_NOT_APPEAR", prompt_text)
        self.assertNotIn("FULL_B_SHOULD_NOT_APPEAR", prompt_text)
        self.assertEqual(payload["prompt_id"], "relation_validation_compact_interface_first")

    def test_similarity_only_relation_candidate_is_rejected_without_validator_call(self) -> None:
        calls = 0

        class CountingValidator:
            model_id = "counting-relation"

            def validate(self, skill_a, skill_b, pair, *, interfaces=None, execution_records=None):
                nonlocal calls
                calls += 1
                return {
                    "edge_type": "compose_with",
                    "direction": "undirected",
                    "confidence": 0.8,
                    "evidence": [{"skill": skill_a.id, "line": 1, "text": "weak"}],
                    "reason": "Weak relation.",
                }

        skill_a = make_skill("skill:a", "alpha", "A text.")
        skill_b = make_skill("skill:b", "beta", "B text.")
        pair = CandidatePair("skill:a", "skill:b", 0.62, sources=["similar_neighbor"])

        records = validate_relation_candidates([pair], [skill_a, skill_b], validator=CountingValidator())

        self.assertEqual(calls, 0)
        self.assertFalse(records[0].accepted)
        self.assertIn("deterministic low-confidence", records[0].rejection_reason)
        summary = summarize_relation_validation_records(records)
        self.assertEqual(summary["deterministic_reject"], 1)
        self.assertEqual(summary["validator_calls"], 0)
        audit = relation_validation_audit_rows(records)
        self.assertEqual(audit[0]["source"], "deterministic_reject")
        self.assertEqual(audit[0]["action"], "deterministic_reject")
        self.assertIn("deterministic low-confidence", audit[0]["reason"])
        self.assertIn("policy_digest", audit[0])

    def test_compact_relation_validation_escalation_records_both_prompt_tiers(self) -> None:
        calls = self._install_fake_litellm_sequence(
            [
                json.dumps(
                    {
                        "edge_type": "none",
                        "direction": "none",
                        "confidence": 0.0,
                        "evidence": [],
                        "reason": "Insufficient context; need full skill markdown.",
                        "needs_full_context": True,
                    }
                ),
                json.dumps(
                    {
                        "edge_type": "compose_with",
                        "direction": "undirected",
                        "confidence": 0.82,
                        "evidence": [{"skill": "skill:a", "line": 1, "text": "A works with beta."}],
                        "reason": "Full context shows the skills compose.",
                    }
                ),
            ]
        )
        skill_a = make_skill("skill:a", "alpha", "A works with beta.")
        skill_b = make_skill("skill:b", "beta", "B complements alpha.")
        pair = CandidatePair("skill:a", "skill:b", 0.85, sources=["explicit_mention"])

        records = validate_relation_candidates(
            [pair],
            [skill_a, skill_b],
            validator=LiteLLMPairValidator(LLMConfig(api_base="https://example.test/api", api_key="sk-test")),
        )

        self.assertEqual(len(calls), 2)
        summary = summarize_relation_validation_records(records)
        self.assertEqual(summary["llm_compact"], 1)
        self.assertEqual(summary["llm_full"], 1)
        self.assertEqual(summary["validator_calls"], 2)
        audit = relation_validation_audit_rows(records)
        self.assertEqual(audit[0]["action"], "llm_full")
        self.assertTrue(audit[0]["escalated_from_compact"])
        self.assertIn("Full context", audit[0]["reason"])

    def test_execution_flow_relation_uses_validator_instead_of_deterministic_accept(self) -> None:
        skill_a = make_skill("skill:consumer", "consumer", "Consumes parsed data.")
        skill_b = make_skill("skill:producer", "producer", "Produces parsed data.")
        pair = CandidatePair(
            "skill:consumer",
            "skill:producer",
            0.95,
            sources=["execution_flow"],
            evidence=[
                RelationEvidence("execution_flow", "skill:consumer", 1, "Consumes parsed data.", "artifact_flow"),
                RelationEvidence("execution_flow", "skill:producer", 1, "Produces parsed data.", "artifact_flow"),
            ],
            direction_hint="A->B",
        )
        calls = 0

        class CountingValidator:
            model_id = "counting-relation"

            def validate(self, skill_a, skill_b, pair, *, interfaces=None, execution_records=None):
                nonlocal calls
                calls += 1
                return {
                    "edge_type": "depend_on",
                    "direction": "A->B",
                    "confidence": 0.91,
                    "evidence": [
                        {"skill": "skill:consumer", "line": 1, "text": "Consumes parsed data."},
                        {"skill": "skill:producer", "line": 1, "text": "Produces parsed data."},
                    ],
                    "reason": "Producer output satisfies consumer input.",
                }

        records = validate_relation_candidates([pair], [skill_a, skill_b], validator=CountingValidator())

        self.assertEqual(calls, 1)
        self.assertTrue(records[0].accepted)
        self.assertIsNotNone(records[0].edge)
        self.assertEqual(records[0].edge.provenance, "llm_validated")
        self.assertEqual(records[0].edge.weight, 0.728)
        summary = summarize_relation_validation_records(records)
        self.assertEqual(summary["deterministic_accept"], 0)

    def test_prompt_contains_confidence_calibration_without_extra_edge_schema(self) -> None:
        skill_a = make_skill("skill:object-picker", "object-picker", "Take object from receptacle.")
        skill_b = make_skill("skill:object-heater", "object-heater", "Requires object in inventory.")
        pair = CandidatePair("skill:object-picker", "skill:object-heater", 0.95, sources=["execution_flow"])

        messages = build_pair_validation_messages(skill_a, skill_b, pair)
        prompt_text = json.dumps(messages, ensure_ascii=False)
        payload = json.loads(messages[1]["content"])

        self.assertIn("confidence_calibration", prompt_text)
        self.assertIn("0.95-1.0", prompt_text)
        self.assertIn("0.85-0.89", prompt_text)
        self.assertEqual(
            set(payload["output_schema"]),
            {"edge_type", "direction", "confidence", "evidence", "reason"},
        )

    def test_prompt_includes_compact_skill_interface_without_routing_labels(self) -> None:
        skill_a = make_skill("skill:goal-interpreter", "goal-interpreter", "Parse task and generate plan.")
        skill_b = make_skill("skill:object-picker", "object-picker", "Take object from receptacle.")
        pair = CandidatePair("skill:goal-interpreter", "skill:object-picker", 0.9, sources=["explicit_mention"])
        interfaces = {
            "skill:goal-interpreter": SkillInterface(
                skill_id="skill:goal-interpreter",
                content_hash="hash-a",
                capability_summary="Parse goals into a plan.",
                when_to_use="Use when a task needs a parsed plan.",
            ),
            "skill:object-picker": SkillInterface(
                skill_id="skill:object-picker",
                content_hash="hash-b",
                capability_summary="Pick up a known object.",
                produces=[InterfaceField(name="object_in_inventory", kind="world_state", confidence=0.9)],
            ),
        }

        messages = build_pair_validation_messages(skill_a, skill_b, pair, interfaces=interfaces)
        payload = json.loads(messages[1]["content"])

        skill_payload = payload["candidate"]["skill_a"]["skill_interface"]
        self.assertEqual(skill_payload["capability_summary"], "Parse goals into a plan.")
        self.assertEqual(skill_payload["when_to_use"], "Use when a task needs a parsed plan.")
        self.assertNotIn("granularity", skill_payload)
        self.assertNotIn("execution_role", skill_payload)
        self.assertNotIn("failure_modes", skill_payload)

    def test_accepts_conditional_depend_on_with_lower_confidence(self) -> None:
        skill_a = make_skill("skill:receptacle-finder", "receptacle-finder", "May require clean object before searching.")
        skill_b = make_skill("skill:clean-object", "clean-object", "Produces clean object state.")
        pair = CandidatePair("skill:receptacle-finder", "skill:clean-object", 0.95, sources=["execution_flow"], direction_hint="A->B")
        validator = StaticPairValidator(
            {
                ("skill:receptacle-finder", "skill:clean-object"): {
                    "edge_type": "depend_on",
                    "direction": "A->B",
                    "confidence": 0.88,
                    "evidence": [{"skill": "skill:receptacle-finder", "line": 1, "text": "May require clean object before searching."}],
                    "reason": "Conditional dependency: cleaning may be required before receptacle search.",
                }
            }
        )

        records = validate_relation_candidates([pair], [skill_a, skill_b], validator=validator)

        self.assertTrue(records[0].accepted)
        self.assertIsNotNone(records[0].edge)
        self.assertLess(records[0].edge.confidence, 0.95)

    def test_accepts_valid_depend_on_and_computes_weight(self) -> None:
        self._install_fake_litellm(
            json.dumps(
                {
                    "edge_type": "depend_on",
                    "direction": "A->B",
                    "confidence": 0.9,
                    "evidence": [{"skill": "skill:a", "line": 1, "text": "A requires B."}],
                    "reason": "A consumes B output.",
                }
            )
        )
        skill_a = make_skill("skill:a", "alpha", "A requires B.")
        skill_b = make_skill("skill:b", "beta", "B produces output.")
        pair = CandidatePair("skill:a", "skill:b", 1.0, sources=["explicit_mention"])

        records = validate_relation_candidates(
            [pair],
            [skill_a, skill_b],
            validator=LiteLLMPairValidator(LLMConfig(api_base="https://example.test/api", api_key="sk-test")),
        )

        self.assertTrue(records[0].accepted)
        self.assertIsNotNone(records[0].edge)
        self.assertEqual(records[0].edge.weight, 0.9)

    def test_rejects_depend_on_reason_that_substitutes_missing_third_skill(self) -> None:
        skill_a = make_skill(
            "skill:beads",
            "beads",
            "Tracks complex work with dependency graphs and persistent issue memory.",
        )
        skill_b = make_skill(
            "skill:research-grants",
            "research-grants",
            "Every research grant proposal MUST include figures using the scientific-schematics skill.",
        )
        pair = CandidatePair("skill:beads", "skill:research-grants", 0.75, sources=["interface_compatibility"])
        validator = StaticPairValidator(
            {
                ("skill:beads", "skill:research-grants"): {
                    "edge_type": "depend_on",
                    "direction": "A->B",
                    "confidence": 0.97,
                    "evidence": [
                        {
                            "skill": "skill:research-grants",
                            "line": 1,
                            "text": "Every research grant proposal MUST include figures using the scientific-schematics skill.",
                        }
                    ],
                    "reason": (
                        "research-grants depends on scientific-schematics. The other skill id was not provided "
                        "in the candidate, so use A->B as the only available directed notation."
                    ),
                }
            }
        )

        records = validate_relation_candidates([pair], [skill_a, skill_b], validator=validator)

        self.assertFalse(records[0].accepted)
        self.assertIsNone(records[0].edge)
        self.assertIn("third skill", records[0].rejection_reason)

    def test_accepts_pair_edge_when_single_evidence_line_names_other_skill(self) -> None:
        skill_a = make_skill("skill:docx", "docx", "Use markitdown to convert DOCX to markdown.")
        skill_b = make_skill("skill:markitdown", "markitdown", "Convert office documents to markdown.")
        pair = CandidatePair("skill:docx", "skill:markitdown", 1.0, sources=["explicit_mention"])
        validator = StaticPairValidator(
            {
                ("skill:docx", "skill:markitdown"): {
                    "edge_type": "compose_with",
                    "direction": "undirected",
                    "confidence": 0.91,
                    "evidence": [{"skill": "skill:docx", "line": 1, "text": "Use markitdown to convert DOCX to markdown."}],
                    "reason": "docx explicitly names markitdown for document conversion, so the pair composes.",
                }
            }
        )

        records = validate_relation_candidates([pair], [skill_a, skill_b], validator=validator)

        self.assertTrue(records[0].accepted, records[0].rejection_reason)
        self.assertIsNotNone(records[0].edge)

    def test_rejects_execution_flow_depend_on_direction_conflict(self) -> None:
        skill_a = make_skill("skill:a-producer", "producer", "A produces object state.")
        skill_b = make_skill("skill:b-consumer", "consumer", "B requires object state.")
        pair = CandidatePair(
            "skill:a-producer",
            "skill:b-consumer",
            0.95,
            sources=["execution_flow"],
            evidence=[
                RelationEvidence(
                    "execution_flow",
                    "skill:a-producer",
                    1,
                    "A produces object state required by B.",
                    "scenario_transition",
                )
            ],
            direction_hint="B->A",
        )
        validator = StaticPairValidator(
            {
                ("skill:a-producer", "skill:b-consumer"): {
                    "edge_type": "depend_on",
                    "direction": "A->B",
                    "confidence": 0.97,
                    "evidence": [{"skill": "skill:a-producer", "line": 1, "text": "A produces object state required by b-consumer."}],
                    "reason": "B depends on A, but direction was emitted incorrectly.",
                }
            }
        )

        records = validate_relation_candidates([pair], [skill_a, skill_b], validator=validator)

        self.assertFalse(records[0].accepted)
        self.assertIsNone(records[0].edge)
        self.assertIn("direction conflicts", records[0].rejection_reason)

    def test_static_validator_flips_direction_for_reverse_pair_key(self) -> None:
        skill_a = make_skill("skill:a", "alpha", "A prepares data.")
        skill_b = make_skill("skill:b", "beta", "B requires A.")
        pair = CandidatePair("skill:a", "skill:b", 1.0, sources=["explicit_mention"])
        validator = StaticPairValidator(
            {
                ("skill:b", "skill:a"): {
                    "edge_type": "depend_on",
                    "direction": "A->B",
                    "confidence": 0.9,
                    "evidence": [{"skill": "skill:b", "line": 1, "text": "B requires A."}],
                    "reason": "B consumes A output.",
                }
            }
        )

        records = validate_relation_candidates([pair], [skill_a, skill_b], validator=validator)

        self.assertTrue(records[0].accepted)
        self.assertIsNotNone(records[0].edge)
        self.assertEqual(records[0].edge.source, "skill:b")
        self.assertEqual(records[0].edge.target, "skill:a")

    def test_rejects_invalid_json_without_caching_retryable_failure(self) -> None:
        calls = self._install_fake_litellm("not json")
        skill_a = make_skill("skill:a", "alpha", "A text.")
        skill_b = make_skill("skill:b", "beta", "B text.")
        pair = CandidatePair("skill:a", "skill:b", 0.78, sources=["similar_neighbor"])

        with TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache.json"
            validator = LiteLLMPairValidator(LLMConfig(api_base="https://example.test/api", api_key="sk-test"))
            first = validate_relation_candidates([pair], [skill_a, skill_b], validator=validator, cache_path=cache_path)
            second = validate_relation_candidates([pair], [skill_a, skill_b], validator=validator, cache_path=cache_path)

        self.assertFalse(first[0].accepted)
        self.assertFalse(second[0].accepted)
        self.assertEqual(len(calls), 6)
        self.assertIn("json_parse_error", first[0].rejection_reason)

    def test_retryable_cached_error_is_revalidated_and_replaced(self) -> None:
        attempts = 0

        class RecoveringValidator:
            model_id = "recovering-relation"

            def validate(self, skill_a, skill_b, pair, *, interfaces=None, execution_records=None):
                nonlocal attempts
                attempts += 1
                return {
                    "edge_type": "depend_on",
                    "direction": "A->B",
                    "confidence": 0.9,
                    "evidence": [{"skill": "skill:a", "line": 1, "text": "A requires B."}],
                    "reason": "A consumes B output.",
                }

        skill_a = make_skill("skill:a", "alpha", "A requires B.")
        skill_b = make_skill("skill:b", "beta", "B produces output.")
        pair = CandidatePair("skill:a", "skill:b", 1.0, sources=["explicit_mention"])
        validator = RecoveringValidator()

        with TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache.json"
            validate_relation_candidates(
                [pair],
                [skill_a, skill_b],
                validator=validator,
                cache_path=cache_path,
                job_options=LLMJobOptions(concurrency=1, max_retries=0, progress_every=0),
            )
            stale_cache = json.loads(cache_path.read_text(encoding="utf-8"))
            key = next(iter(stale_cache))
            cache_path.write_text(
                json.dumps(
                    {
                        key: {
                            "edge_type": "none",
                            "direction": "none",
                            "confidence": 0.0,
                            "evidence": [],
                            "reason": "service unavailable",
                            "error_type": "api_error",
                        }
                    }
                ),
                encoding="utf-8",
            )

            records = validate_relation_candidates(
                [pair],
                [skill_a, skill_b],
                validator=validator,
                cache_path=cache_path,
                job_options=LLMJobOptions(concurrency=1, max_retries=0, progress_every=0),
            )
            cached = json.loads(cache_path.read_text(encoding="utf-8"))

        self.assertEqual(attempts, 2)
        self.assertTrue(records[0].accepted)
        self.assertEqual(len(cached), 1)
        self.assertNotIn("error_type", next(iter(cached.values())))

    def test_relation_validation_summary_counts_cache_hits(self) -> None:
        validator = StaticPairValidator(
            {
                ("skill:a", "skill:b"): {
                    "edge_type": "depend_on",
                    "direction": "A->B",
                    "confidence": 0.9,
                    "evidence": [{"skill": "skill:a", "line": 1, "text": "A requires B."}],
                    "reason": "A consumes B output.",
                }
            }
        )
        skill_a = make_skill("skill:a", "alpha", "A requires B.")
        skill_b = make_skill("skill:b", "beta", "B produces output.")
        pair = CandidatePair("skill:a", "skill:b", 1.0, sources=["explicit_mention"])

        with TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache.json"
            validate_relation_candidates([pair], [skill_a, skill_b], validator=validator, cache_path=cache_path)
            records = validate_relation_candidates([pair], [skill_a, skill_b], validator=validator, cache_path=cache_path)

        summary = summarize_relation_validation_records(records)
        self.assertEqual(summary["cache_hits"], 1)
        self.assertEqual(summary["validator_calls"], 0)

    def test_validation_retries_transient_error_payloads(self) -> None:
        attempts = 0

        class FlakyValidator:
            model_id = "flaky-relation"

            def validate(self, skill_a, skill_b, pair, *, interfaces=None, execution_records=None):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    return {
                        "edge_type": "none",
                        "direction": "none",
                        "confidence": 0.0,
                        "evidence": [],
                        "reason": "timeout",
                        "error_type": "api_error",
                    }
                return {
                    "edge_type": "depend_on",
                    "direction": "A->B",
                    "confidence": 0.9,
                    "evidence": [{"skill": "skill:a", "line": 1, "text": "A requires B."}],
                    "reason": "A consumes B output.",
                }

        skill_a = make_skill("skill:a", "alpha", "A requires B.")
        skill_b = make_skill("skill:b", "beta", "B produces output.")
        pair = CandidatePair("skill:a", "skill:b", 1.0, sources=["explicit_mention"])

        records = validate_relation_candidates(
            [pair],
            [skill_a, skill_b],
            validator=FlakyValidator(),
            job_options=LLMJobOptions(concurrency=1, max_retries=1, progress_every=0),
        )

        self.assertEqual(attempts, 2)
        self.assertTrue(records[0].accepted)


if __name__ == "__main__":
    unittest.main()
