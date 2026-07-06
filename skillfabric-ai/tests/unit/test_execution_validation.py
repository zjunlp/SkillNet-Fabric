from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from skillfabric.compiled_graph.execution.models import ExecutionEvidence, ExecutionFlowCandidate
from skillfabric.compiled_graph.execution.prompts import (
    build_compact_execution_validation_messages,
    build_execution_validation_messages,
)
from skillfabric.compiled_graph.execution.validation import (
    DeterministicExecutionFlowValidator,
    LiteLLMExecutionFlowValidator,
    execution_validation_audit_rows,
    summarize_execution_validation_records,
    validate_execution_flow_candidates,
)
from skillfabric.compiled_graph.interface.models import InterfaceField, SkillInterface
from skillfabric.runtime.jobs import LLMJobOptions
from skillfabric.runtime.llm import LLMConfig
from tests.unit.relation_helpers import make_skill


class ExecutionValidationTests(unittest.TestCase):
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

    def _install_fake_litellm_payload(self, payload: dict[str, object]) -> list[dict[str, object]]:
        calls: list[dict[str, object]] = []
        fake_litellm = types.SimpleNamespace()

        def fake_completion(**kwargs):
            calls.append(kwargs)
            return payload

        fake_litellm.completion = fake_completion
        sys.modules["litellm"] = fake_litellm
        return calls

    def _candidate(self) -> ExecutionFlowCandidate:
        return ExecutionFlowCandidate(
            source_skill="skill:producer",
            target_skill="skill:consumer",
            flow_type="artifact_flow",
            matched_node_id="artifact:csv",
            matched_name="csv",
            evidence=[
                ExecutionEvidence(skill="skill:producer", line=2, text="Produce csv."),
                ExecutionEvidence(skill="skill:consumer", line=3, text="Consume csv."),
            ],
        )

    def _skills(self):
        producer = make_skill("skill:producer", "producer", "Produce csv.")
        consumer = make_skill("skill:consumer", "consumer", "Consume csv.")
        return [producer, consumer]

    def _interfaces(self):
        return {
            "skill:producer": SkillInterface(
                skill_id="skill:producer",
                content_hash="hash-producer",
                capability_summary="Produce csv.",
                produces=[InterfaceField(name="csv", kind="artifact", confidence=0.9)],
            ),
            "skill:consumer": SkillInterface(
                skill_id="skill:consumer",
                content_hash="hash-consumer",
                capability_summary="Consume csv.",
                requires=[InterfaceField(name="csv", kind="artifact", confidence=0.9)],
            ),
        }

    def test_deterministic_validator_accepts_evidence_backed_candidate(self) -> None:
        records = validate_execution_flow_candidates(
            [self._candidate()],
            self._skills(),
            interfaces=self._interfaces(),
            validator=DeterministicExecutionFlowValidator(),
        )

        self.assertTrue(records[0].accepted)
        self.assertEqual(records[0].flow_edge.type, "artifact_flow")
        self.assertEqual(records[0].normalized["projected_edge_type"], "depend_on")

    def test_litellm_accepts_fenced_json_response(self) -> None:
        self._install_fake_litellm(
            "```json\n"
            + json.dumps(
                {
                    "accepted": True,
                    "flow_type": "artifact_flow",
                    "projected_edge_type": "depend_on",
                    "confidence": 0.91,
                    "evidence": [{"skill": "skill:producer", "line": 2, "text": "Produce csv."}],
                    "reason": "Consumer needs producer output.",
                }
            )
            + "\n```"
        )
        records = validate_execution_flow_candidates(
            [self._candidate()],
            self._skills(),
            interfaces=self._interfaces(),
            validator=LiteLLMExecutionFlowValidator(
                config=LLMConfig(api_base="https://example.test/api", api_key="sk-test")
            ),
        )

        self.assertTrue(records[0].accepted)
        self.assertEqual(records[0].flow_edge.metadata["artifact_id"], "artifact:csv")

    def test_prompt_defines_projection_direction(self) -> None:
        messages = build_execution_validation_messages(
            self._candidate(),
            self._skills()[0],
            self._skills()[1],
            interfaces=self._interfaces(),
        )
        prompt_text = json.dumps(messages, ensure_ascii=False)
        payload = json.loads(messages[1]["content"])

        self.assertEqual(payload["prompt_id"], "execution_validation_handoff_precision")
        for field in ("todo", "input", "output", "workflow", "rules", "constraints"):
            self.assertIn(field, payload)
        self.assertIn("decision_workflow", payload)
        self.assertIn("precision_goal", payload["direction_semantics"])
        self.assertIn("For projected_edge_type=depend_on, the target skill depends on the source skill", prompt_text)
        self.assertIn("source_skill produces or enables; target_skill consumes or requires", prompt_text)
        self.assertIn("belief_state or planning_state", prompt_text)
        self.assertIn("not a world-state producer", prompt_text)
        self.assertIn("same overall task", prompt_text)
        self.assertIn("generic data, text, output, result", prompt_text)

    def test_invalid_json_and_api_exception_are_rejected(self) -> None:
        self._install_fake_litellm("not json")
        invalid = validate_execution_flow_candidates(
            [self._candidate()],
            self._skills(),
            interfaces=self._interfaces(),
            validator=LiteLLMExecutionFlowValidator(
                config=LLMConfig(api_base="https://example.test/api", api_key="sk-test")
            ),
        )
        self.assertFalse(invalid[0].accepted)
        self.assertIn("json_parse_error", invalid[0].rejection_reason)

        self._install_fake_litellm(RuntimeError("network down"))
        failed = validate_execution_flow_candidates(
            [self._candidate()],
            self._skills(),
            interfaces=self._interfaces(),
            validator=LiteLLMExecutionFlowValidator(
                config=LLMConfig(api_base="https://example.test/api", api_key="sk-test")
            ),
        )
        self.assertFalse(failed[0].accepted)
        self.assertIn("api_error", failed[0].rejection_reason)

    def test_schema_invalid_confidence_is_rejected_without_crashing(self) -> None:
        self._install_fake_litellm(
            json.dumps(
                {
                    "accepted": True,
                    "flow_type": "artifact_flow",
                    "projected_edge_type": "depend_on",
                    "confidence": "high",
                    "evidence": [{"skill": "skill:producer", "line": "bad", "text": "Produce csv."}],
                    "reason": "Bad schema.",
                }
            )
        )

        records = validate_execution_flow_candidates(
            [self._candidate()],
            self._skills(),
            interfaces=self._interfaces(),
            validator=LiteLLMExecutionFlowValidator(
                config=LLMConfig(api_base="https://example.test/api", api_key="sk-test")
            ),
        )

        self.assertFalse(records[0].accepted)
        self.assertIn("schema_error", records[0].rejection_reason)

    def test_schema_invalid_falsy_confidence_is_rejected(self) -> None:
        self._install_fake_litellm(
            json.dumps(
                {
                    "accepted": True,
                    "flow_type": "artifact_flow",
                    "projected_edge_type": "depend_on",
                    "confidence": [],
                    "evidence": [{"skill": "skill:producer", "line": 2, "text": "Produce csv."}],
                    "reason": "Bad schema.",
                }
            )
        )

        records = validate_execution_flow_candidates(
            [self._candidate()],
            self._skills(),
            interfaces=self._interfaces(),
            validator=LiteLLMExecutionFlowValidator(
                config=LLMConfig(api_base="https://example.test/api", api_key="sk-test")
            ),
        )

        self.assertFalse(records[0].accepted)
        self.assertIn("schema_error", records[0].rejection_reason)

    def test_litellm_accepts_responses_api_output_shape(self) -> None:
        self._install_fake_litellm_payload(
            {
                "output": [
                    {
                        "content": [
                            {
                                "text": json.dumps(
                                    {
                                        "accepted": True,
                                        "flow_type": "artifact_flow",
                                        "projected_edge_type": "depend_on",
                                        "confidence": 0.91,
                                        "evidence": [
                                            {
                                                "skill": "skill:producer",
                                                "line": 2,
                                                "text": "Produce csv.",
                                            }
                                        ],
                                        "reason": "Consumer needs producer output.",
                                    }
                                )
                            }
                        ]
                    }
                ]
            }
        )

        records = validate_execution_flow_candidates(
            [self._candidate()],
            self._skills(),
            interfaces=self._interfaces(),
            validator=LiteLLMExecutionFlowValidator(
                config=LLMConfig(api_base="https://example.test/api", api_key="sk-test")
            ),
        )

        self.assertTrue(records[0].accepted)
        self.assertEqual(records[0].flow_edge.type, "artifact_flow")

    def test_validation_cache_prevents_repeated_litellm_calls(self) -> None:
        calls = self._install_fake_litellm(
            json.dumps(
                {
                    "accepted": False,
                    "flow_type": "none",
                    "projected_edge_type": "none",
                    "confidence": 0.0,
                    "evidence": [],
                    "reason": "No flow.",
                }
            )
        )
        validator = LiteLLMExecutionFlowValidator(
            config=LLMConfig(api_base="https://example.test/api", api_key="sk-test")
        )
        with TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "execution_cache.json"
            validate_execution_flow_candidates(
                [self._candidate()],
                self._skills(),
                interfaces=self._interfaces(),
                validator=validator,
                cache_path=cache_path,
            )
            validate_execution_flow_candidates(
                [self._candidate()],
                self._skills(),
                interfaces=self._interfaces(),
                validator=validator,
                cache_path=cache_path,
            )
            cached_records = validate_execution_flow_candidates(
                [self._candidate()],
                self._skills(),
                interfaces=self._interfaces(),
                validator=validator,
                cache_path=cache_path,
            )

        self.assertEqual(len(calls), 1)
        summary = summarize_execution_validation_records(cached_records)
        self.assertEqual(summary["cache_hits"], 1)
        self.assertEqual(summary["validator_calls"], 0)

    def test_validation_retries_transient_error_payloads(self) -> None:
        attempts = 0

        class FlakyValidator:
            model_id = "flaky-execution"

            def validate(self, candidate, source_skill, target_skill, *, interfaces):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    return {
                        "accepted": False,
                        "flow_type": "none",
                        "projected_edge_type": "none",
                        "confidence": 0.0,
                        "evidence": [],
                        "reason": "timeout",
                        "error_type": "api_error",
                    }
                return {
                    "accepted": True,
                    "flow_type": "artifact_flow",
                    "projected_edge_type": "depend_on",
                    "confidence": 0.91,
                    "evidence": [{"skill": "skill:producer", "line": 2, "text": "Produce csv."}],
                    "reason": "Consumer needs producer output.",
                }

        records = validate_execution_flow_candidates(
            [self._candidate()],
            self._skills(),
            interfaces=self._interfaces(),
            validator=FlakyValidator(),
            job_options=LLMJobOptions(concurrency=1, max_retries=1, progress_every=0),
        )

        self.assertEqual(attempts, 2)
        self.assertTrue(records[0].accepted)

    def test_retryable_cached_error_is_revalidated_and_replaced(self) -> None:
        attempts = 0

        class RecoveringValidator:
            model_id = "recovering-execution"

            def validate(self, candidate, source_skill, target_skill, *, interfaces):
                nonlocal attempts
                attempts += 1
                return {
                    "accepted": True,
                    "flow_type": "artifact_flow",
                    "projected_edge_type": "depend_on",
                    "confidence": 0.91,
                    "evidence": [{"skill": "skill:producer", "line": 2, "text": "Produce csv."}],
                    "reason": "Consumer needs producer output.",
                }

        with TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "execution_cache.json"
            skills = self._skills()
            validator = RecoveringValidator()
            validate_execution_flow_candidates(
                [self._candidate()],
                skills,
                interfaces=self._interfaces(),
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
                            "accepted": False,
                            "flow_type": "none",
                            "projected_edge_type": "none",
                            "confidence": 0.0,
                            "evidence": [],
                            "reason": "service unavailable",
                            "error_type": "api_error",
                        }
                    }
                ),
                encoding="utf-8",
            )

            records = validate_execution_flow_candidates(
                [self._candidate()],
                skills,
                interfaces=self._interfaces(),
                validator=validator,
                cache_path=cache_path,
                job_options=LLMJobOptions(concurrency=1, max_retries=0, progress_every=0),
            )
            cached = json.loads(cache_path.read_text(encoding="utf-8"))

        self.assertEqual(attempts, 2)
        self.assertTrue(records[0].accepted)
        self.assertEqual(len(cached), 1)
        self.assertNotIn("error_type", next(iter(cached.values())))

    def test_prompt_contains_full_skill_md_interface_and_candidate_evidence(self) -> None:
        messages = build_execution_validation_messages(
            self._candidate(),
            self._skills()[0],
            self._skills()[1],
            interfaces=self._interfaces(),
        )
        prompt = json.dumps(messages, ensure_ascii=False)

        self.assertIn("full_skill_md", prompt)
        self.assertIn("skill_interface", prompt)
        self.assertIn("Produce csv.", prompt)

    def test_compact_prompt_uses_interface_and_candidate_evidence_without_full_skill_md(self) -> None:
        producer = make_skill("skill:producer", "producer", "Produce csv.\nFULL_SOURCE_SHOULD_NOT_APPEAR")
        consumer = make_skill("skill:consumer", "consumer", "Consume csv.\nFULL_TARGET_SHOULD_NOT_APPEAR")

        messages = build_compact_execution_validation_messages(
            self._candidate(),
            producer,
            consumer,
            interfaces=self._interfaces(),
        )
        prompt = json.dumps(messages, ensure_ascii=False)
        payload = json.loads(messages[1]["content"])

        self.assertIn("skill_interface", prompt)
        self.assertIn("candidate", prompt)
        self.assertNotIn("full_skill_md", prompt)
        self.assertNotIn("FULL_SOURCE_SHOULD_NOT_APPEAR", prompt)
        self.assertNotIn("FULL_TARGET_SHOULD_NOT_APPEAR", prompt)
        self.assertEqual(payload["prompt_id"], "execution_validation_compact_interface_first")

    def test_high_confidence_execution_candidate_still_uses_llm_validator(self) -> None:
        calls = self._install_fake_litellm(
            json.dumps(
                {
                    "accepted": True,
                    "flow_type": "artifact_flow",
                    "projected_edge_type": "depend_on",
                    "confidence": 0.91,
                    "evidence": [{"skill": "skill:producer", "line": 2, "text": "Produce csv."}],
                    "reason": "LLM confirms the consumer needs producer output.",
                }
            )
        )
        candidate = self._candidate()
        candidate.metadata["canonical_object_id"] = "artifact:csv"

        records = validate_execution_flow_candidates(
            [candidate],
            self._skills(),
            interfaces=self._interfaces(),
            validator=LiteLLMExecutionFlowValidator(
                config=LLMConfig(api_base="https://example.test/api", api_key="sk-test")
            ),
        )

        self.assertEqual(len(calls), 1)
        self.assertTrue(records[0].accepted)
        self.assertEqual(records[0].normalized["reason"], "LLM confirms the consumer needs producer output.")
        summary = summarize_execution_validation_records(records)
        self.assertEqual(summary["deterministic_accept"], 0)
        self.assertEqual(summary["llm_compact"], 1)
        self.assertEqual(summary["validator_calls"], 1)
        audit = execution_validation_audit_rows(records)
        self.assertEqual(audit[0]["source"], "llm")
        self.assertEqual(audit[0]["action"], "llm_compact")
        self.assertIn("LLM confirms", audit[0]["reason"])
        self.assertIn("policy_digest", audit[0])

    def test_compact_execution_validation_escalation_records_both_prompt_tiers(self) -> None:
        calls = self._install_fake_litellm_sequence(
            [
                json.dumps(
                    {
                        "accepted": False,
                        "flow_type": "none",
                        "projected_edge_type": "none",
                        "confidence": 0.0,
                        "evidence": [],
                        "reason": "Insufficient context; need full skill markdown.",
                        "needs_full_context": True,
                    }
                ),
                json.dumps(
                    {
                        "accepted": True,
                        "flow_type": "artifact_flow",
                        "projected_edge_type": "depend_on",
                        "confidence": 0.91,
                        "evidence": [{"skill": "skill:producer", "line": 2, "text": "Produce csv."}],
                        "reason": "Full context confirms the consumer needs producer output.",
                    }
                ),
            ]
        )

        records = validate_execution_flow_candidates(
            [self._candidate()],
            self._skills(),
            interfaces=self._interfaces(),
            validator=LiteLLMExecutionFlowValidator(
                config=LLMConfig(api_base="https://example.test/api", api_key="sk-test")
            ),
        )

        self.assertEqual(len(calls), 2)
        summary = summarize_execution_validation_records(records)
        self.assertEqual(summary["llm_compact"], 1)
        self.assertEqual(summary["llm_full"], 1)
        self.assertEqual(summary["validator_calls"], 2)
        audit = execution_validation_audit_rows(records)
        self.assertEqual(audit[0]["action"], "llm_full")
        self.assertTrue(audit[0]["escalated_from_compact"])
        self.assertIn("Full context", audit[0]["reason"])


if __name__ == "__main__":
    unittest.main()
