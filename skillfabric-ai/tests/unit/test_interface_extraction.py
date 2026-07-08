from __future__ import annotations

import json
import sys
import types
import unittest

from skillfabric.compiled_graph.interface.extraction import (
    DeterministicInterfaceExtractor,
    LiteLLMInterfaceExtractor,
    extract_skill_interfaces,
)
from skillfabric.compiled_graph.interface.prompts import build_interface_extraction_messages
from skillfabric.runtime.llm import LLMConfig
from tests.unit.relation_helpers import make_skill


class InterfaceExtractionTests(unittest.TestCase):
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

    def _install_fake_litellm_payload(self, payload: dict[str, object]) -> list[dict[str, object]]:
        calls: list[dict[str, object]] = []
        fake_litellm = types.SimpleNamespace()

        def fake_completion(**kwargs):
            calls.append(kwargs)
            return payload

        fake_litellm.completion = fake_completion
        sys.modules["litellm"] = fake_litellm
        return calls

    def test_valid_litellm_json_generates_interface(self) -> None:
        self._install_fake_litellm(
            json.dumps(
                {
                    "capability_summary": "Extract PDF tables into CSV.",
                    "when_to_use": "Use when PDF tables need structured CSV output.",
                    "produces": [
                        {
                            "name": "csv tables",
                            "description": "Structured table CSV output.",
                            "kind": "artifact",
                            "confidence": 0.9,
                            "evidence": [{"skill": "skill:pdf", "line": 2, "text": "Write CSV tables."}],
                        }
                    ],
                    "uses_tools": [],
                }
            )
        )
        skill = make_skill("skill:pdf", "pdf", "Write CSV tables.")
        extractor = LiteLLMInterfaceExtractor(LLMConfig(api_base="https://example.test/api", api_key="sk-test"))

        records = extract_skill_interfaces([skill], extractor=extractor)

        self.assertTrue(records[0].accepted)
        self.assertEqual(records[0].interface.capability_summary, "Extract PDF tables into CSV.")
        self.assertEqual(records[0].interface.when_to_use, "Use when PDF tables need structured CSV output.")
        self.assertEqual(records[0].interface.produces[0].name, "csv tables")
        self.assertNotIn("inferred", records[0].interface.produces[0].to_dict())
        self.assertEqual(records[0].interface.provenance, "llm_extracted")

    def test_legacy_inferred_field_is_ignored(self) -> None:
        self._install_fake_litellm(
            json.dumps(
                {
                    "capability_summary": "Extract PDF tables into CSV.",
                    "produces": [
                        {
                            "name": "csv tables",
                            "kind": "artifact",
                            "confidence": 0.9,
                            "inferred": True,
                            "evidence": [],
                        }
                    ],
                }
            )
        )
        skill = make_skill("skill:pdf", "pdf", "Write CSV tables.")
        extractor = LiteLLMInterfaceExtractor(LLMConfig(api_base="https://example.test/api", api_key="sk-test"))

        records = extract_skill_interfaces([skill], extractor=extractor)

        self.assertTrue(records[0].accepted)
        self.assertNotIn("inferred", records[0].interface.produces[0].to_dict())

    def test_deterministic_fallback_omits_routing_classification_fields(self) -> None:
        skills = [
            make_skill("skill:alfworld-goal-interpreter", "alfworld-goal-interpreter", "Parse the task and generate a sequential plan."),
            make_skill("skill:alfworld-object-picker", "alfworld-object-picker", "Execute take object from receptacle."),
            make_skill("skill:alfworld-heat-object-with-appliance", "alfworld-heat-object-with-appliance", "Locate, take, heat, and place the object."),
            make_skill("skill:alfworld-object-state-inspector", "alfworld-object-state-inspector", "Inspect receptacle contents and report state."),
        ]

        records = extract_skill_interfaces(skills, extractor=DeterministicInterfaceExtractor())
        by_id = {record.skill_id: record.interface for record in records}

        for interface in by_id.values():
            payload = interface.to_dict()
            self.assertNotIn("granularity", payload)
            self.assertNotIn("execution_role", payload)
            self.assertNotIn("failure_modes", payload)

    def test_fenced_json_is_parsed(self) -> None:
        self._install_fake_litellm(
            "```json\n"
            + json.dumps({"capability_summary": "Run tests.", "uses_tools": [{"name": "pytest", "kind": "tool"}]})
            + "\n```"
        )
        skill = make_skill("skill:test", "test", "Run pytest.")

        records = extract_skill_interfaces(
            [skill],
            extractor=LiteLLMInterfaceExtractor(LLMConfig(api_base="https://example.test/api", api_key="sk-test")),
        )

        self.assertTrue(records[0].accepted)
        self.assertEqual(records[0].interface.uses_tools[0].name, "pytest")

    def test_responses_api_output_shape_is_parsed(self) -> None:
        self._install_fake_litellm_payload(
            {
                "output": [
                    {
                        "content": [
                            {
                                "text": json.dumps(
                                    {
                                        "capability_summary": "Run tests.",
                                        "uses_tools": [{"name": "pytest", "kind": "tool"}],
                                    }
                                )
                            }
                        ]
                    }
                ]
            }
        )
        skill = make_skill("skill:test", "test", "Run pytest.")

        records = extract_skill_interfaces(
            [skill],
            extractor=LiteLLMInterfaceExtractor(LLMConfig(api_base="https://example.test/api", api_key="sk-test")),
        )

        self.assertTrue(records[0].accepted)
        self.assertEqual(records[0].interface.uses_tools[0].name, "pytest")

    def test_invalid_json_uses_fallback_and_records_rejection(self) -> None:
        self._install_fake_litellm("not json")
        skill = make_skill("skill:pdf", "pdf", "Write CSV.", artifacts=["csv"])

        records = extract_skill_interfaces(
            [skill],
            extractor=LiteLLMInterfaceExtractor(LLMConfig(api_base="https://example.test/api", api_key="sk-test")),
        )

        self.assertFalse(records[0].accepted)
        self.assertIn("json_parse_error", records[0].rejection_reason)
        self.assertEqual(records[0].interface.provenance, "deterministic_fallback")
        self.assertEqual(records[0].interface.requires, [])
        self.assertEqual(records[0].interface.produces, [])
        self.assertEqual(records[0].interface.uses_tools, [])

    def test_schema_invalid_json_uses_fallback_and_records_rejection(self) -> None:
        self._install_fake_litellm(json.dumps({"edge_type": "none", "confidence": 0.0}))
        skill = make_skill("skill:pdf", "pdf", "Write CSV.", artifacts=["csv"])

        records = extract_skill_interfaces(
            [skill],
            extractor=LiteLLMInterfaceExtractor(LLMConfig(api_base="https://example.test/api", api_key="sk-test")),
        )

        self.assertFalse(records[0].accepted)
        self.assertIn("schema_error", records[0].rejection_reason)
        self.assertEqual(records[0].interface.provenance, "deterministic_fallback")

    def test_evidence_only_json_is_not_a_valid_interface(self) -> None:
        self._install_fake_litellm(json.dumps({"edge_type": "none", "evidence": []}))
        skill = make_skill("skill:pdf", "pdf", "Write CSV.", artifacts=["csv"])

        records = extract_skill_interfaces(
            [skill],
            extractor=LiteLLMInterfaceExtractor(LLMConfig(api_base="https://example.test/api", api_key="sk-test")),
        )

        self.assertFalse(records[0].accepted)
        self.assertIn("schema_error", records[0].rejection_reason)

    def test_schema_invalid_list_field_uses_fallback_and_records_rejection(self) -> None:
        self._install_fake_litellm(json.dumps({"capability_summary": "Run tests.", "uses_tools": "pytest"}))
        skill = make_skill("skill:test", "test", "Run pytest.", tools=["pytest"])

        records = extract_skill_interfaces(
            [skill],
            extractor=LiteLLMInterfaceExtractor(LLMConfig(api_base="https://example.test/api", api_key="sk-test")),
        )

        self.assertFalse(records[0].accepted)
        self.assertIn("schema_error", records[0].rejection_reason)
        self.assertEqual(records[0].interface.uses_tools, [])

    def test_recoverable_shape_drift_is_normalized_without_fallback(self) -> None:
        self._install_fake_litellm(
            json.dumps(
                {
                    "capability_summary": "Clean an object.",
                    "when_to_use": ["Use when the task asks to clean or wash an object."],
                    "produces": [
                        {
                            "name": "clean_object_state",
                            "kind": "world_state",
                            "confidence": 0.93,
                            "evidence": ["line 12: The object is now clean."],
                        }
                    ],
                }
            )
        )
        skill = make_skill("skill:clean", "clean", "The object is now clean.")

        records = extract_skill_interfaces(
            [skill],
            extractor=LiteLLMInterfaceExtractor(LLMConfig(api_base="https://example.test/api", api_key="sk-test")),
        )

        self.assertTrue(records[0].accepted)
        self.assertEqual(records[0].interface.provenance, "llm_extracted")
        self.assertEqual(records[0].interface.when_to_use, "Use when the task asks to clean or wash an object.")
        self.assertEqual(records[0].interface.produces[0].name, "clean_object_state")
        self.assertEqual(records[0].interface.produces[0].evidence, [])

    def test_invalid_field_kind_uses_fallback_and_records_rejection(self) -> None:
        self._install_fake_litellm(
            json.dumps(
                {
                    "capability_summary": "Track remembered object location.",
                    "produces": [
                        {
                            "name": "object_permanence_state",
                            "kind": "state",
                            "confidence": 0.9,
                        }
                    ],
                }
            )
        )
        skill = make_skill("skill:planner", "planner", "Track remembered object location.")

        records = extract_skill_interfaces(
            [skill],
            extractor=LiteLLMInterfaceExtractor(LLMConfig(api_base="https://example.test/api", api_key="sk-test")),
        )

        self.assertFalse(records[0].accepted)
        self.assertIn("schema_error", records[0].rejection_reason)
        self.assertEqual(records[0].interface.provenance, "deterministic_fallback")

    def test_tool_kind_is_not_kept_in_requires_or_produces(self) -> None:
        self._install_fake_litellm(
            json.dumps(
                {
                    "capability_summary": "Run tests.",
                    "requires": [{"name": "pytest", "kind": "tool", "confidence": 0.9}],
                    "produces": [{"name": "pytest report", "kind": "tool", "confidence": 0.9}],
                    "uses_tools": [{"name": "pytest", "kind": "tool", "confidence": 0.9}],
                }
            )
        )
        skill = make_skill("skill:test", "test", "Run pytest.")

        records = extract_skill_interfaces(
            [skill],
            extractor=LiteLLMInterfaceExtractor(LLMConfig(api_base="https://example.test/api", api_key="sk-test")),
        )

        self.assertTrue(records[0].accepted)
        self.assertEqual(records[0].interface.requires, [])
        self.assertEqual(records[0].interface.produces, [])
        self.assertEqual(records[0].interface.uses_tools[0].kind, "tool")

    def test_schema_invalid_field_confidence_uses_fallback_and_records_rejection(self) -> None:
        self._install_fake_litellm(
            json.dumps(
                {
                    "capability_summary": "Run tests.",
                    "uses_tools": [{"name": "pytest", "kind": "tool", "confidence": "high"}],
                }
            )
        )
        skill = make_skill("skill:test", "test", "Run pytest.", tools=["pytest"])

        records = extract_skill_interfaces(
            [skill],
            extractor=LiteLLMInterfaceExtractor(LLMConfig(api_base="https://example.test/api", api_key="sk-test")),
        )

        self.assertFalse(records[0].accepted)
        self.assertIn("schema_error", records[0].rejection_reason)

    def test_schema_invalid_evidence_line_uses_fallback_and_records_rejection(self) -> None:
        self._install_fake_litellm(
            json.dumps(
                {
                    "capability_summary": "Run tests.",
                    "evidence": [{"skill": "skill:test", "line": "bad", "text": "Run pytest."}],
                }
            )
        )
        skill = make_skill("skill:test", "test", "Run pytest.", tools=["pytest"])

        records = extract_skill_interfaces(
            [skill],
            extractor=LiteLLMInterfaceExtractor(LLMConfig(api_base="https://example.test/api", api_key="sk-test")),
        )

        self.assertFalse(records[0].accepted)
        self.assertIn("schema_error", records[0].rejection_reason)

    def test_api_exception_uses_fallback_and_records_rejection(self) -> None:
        self._install_fake_litellm(RuntimeError("network down"))
        skill = make_skill("skill:pdf", "pdf", "Write CSV.", artifacts=["csv"])

        records = extract_skill_interfaces(
            [skill],
            extractor=LiteLLMInterfaceExtractor(LLMConfig(api_base="https://example.test/api", api_key="sk-test")),
        )

        self.assertFalse(records[0].accepted)
        self.assertIn("api_error", records[0].rejection_reason)
        self.assertEqual(records[0].interface.produces, [])

    def test_deterministic_fallback_is_empty_and_does_not_call_litellm(self) -> None:
        calls = self._install_fake_litellm(RuntimeError("should not be called"))
        skill = make_skill(
            "skill:pdf",
            "pdf",
            "Use python to write CSV.",
        )

        records = extract_skill_interfaces([skill], extractor=DeterministicInterfaceExtractor())

        self.assertEqual(calls, [])
        self.assertTrue(records[0].accepted)
        self.assertEqual(records[0].interface.capability_summary, skill.description)
        self.assertEqual(records[0].interface.when_to_use, skill.description)
        self.assertEqual(records[0].interface.requires, [])
        self.assertEqual(records[0].interface.produces, [])
        self.assertEqual(records[0].interface.uses_tools, [])

    def test_deterministic_fallback_does_not_infer_obvious_inputs_and_outputs(self) -> None:
        skill = make_skill(
            "skill:kpi",
            "kpi",
            "Use after parser has produced CSV tables.\nLoad CSV tables and output kpi.json.",
        )

        records = extract_skill_interfaces([skill], extractor=DeterministicInterfaceExtractor())

        self.assertEqual(records[0].interface.requires, [])
        self.assertEqual(records[0].interface.produces, [])

    def test_deterministic_fallback_does_not_treat_generation_verbs_as_outputs(self) -> None:
        skill = make_skill(
            "skill:clinical-report",
            "clinical-report",
            "Generate professional clinical decision support documents as publication-ready PDF files.",
        )

        records = extract_skill_interfaces([skill], extractor=DeterministicInterfaceExtractor())

        self.assertEqual(records[0].interface.requires, [])
        self.assertEqual(records[0].interface.produces, [])

    def test_deterministic_fallback_does_not_extract_hyphenated_substring_artifacts(self) -> None:
        skill = make_skill(
            "skill:survival-analysis",
            "survival-analysis",
            "Survival analysis with Kaplan-Meier curves and log-rank tests.",
        )

        records = extract_skill_interfaces([skill], extractor=DeterministicInterfaceExtractor())
        field_names = {field.name for field in records[0].interface.requires + records[0].interface.produces}

        self.assertNotIn("log", field_names)

    def test_deterministic_fallback_does_not_invent_fields_from_ambiguous_text(self) -> None:
        skill = make_skill(
            "skill:ambiguous-communication",
            "ambiguous-communication",
            "\n".join(
                [
                    "Transform source material into a clear communication plan.",
                    "Use when the task needs audience context, structure, and verification notes.",
                    "| Input | Output | Notes |",
                    "```markdown",
                ]
            ),
        )

        records = extract_skill_interfaces([skill], extractor=DeterministicInterfaceExtractor())
        interface = records[0].interface

        self.assertEqual(interface.requires, [])
        self.assertEqual(interface.produces, [])

    def test_prompt_contains_full_skill_md(self) -> None:
        skill = make_skill("skill:pdf", "pdf", "Line one.\nFULL_SKILL_LINE")

        messages = build_interface_extraction_messages(skill)
        prompt = json.dumps(messages, ensure_ascii=False)
        payload = json.loads(messages[1]["content"])

        self.assertEqual(payload["prompt_id"], "skill_contract_core_interface_v3")
        for field in (
            "role",
            "objective",
            "source_contract",
            "contract_semantics",
            "quality_standard",
            "process",
            "format_requirements",
            "examples",
        ):
            self.assertIn(field, payload)
        self.assertEqual(
            set(payload["output_schema"]),
            {"capability_summary", "when_to_use", "requires", "produces", "uses_tools"},
        )
        self.assertNotIn("granularity", prompt)
        self.assertNotIn("execution_role", prompt)
        self.assertNotIn("failure_modes", prompt)
        self.assertNotIn("failure signals", prompt)
        self.assertNotIn("failure mode", prompt)
        self.assertNotIn("verification signals", prompt)
        self.assertNotIn("scope boundaries", prompt)
        self.assertNotIn("non-goals", prompt)
        self.assertIn("FULL_SKILL_LINE", prompt)
        self.assertIn("full_skill_md", prompt)
        self.assertIn("requires", prompt)
        self.assertIn("produces", prompt)
        self.assertIn("SkillContract", prompt)
        self.assertIn("reusable operational capability", prompt)
        self.assertIn("stable snake_case", prompt)
        self.assertIn("routing_value", prompt)
        self.assertIn("environment_diagnosis", prompt)
        self.assertIn("world_state", prompt)
        self.assertIn("belief_state", prompt)
        self.assertIn("planning_state", prompt)
        self.assertIn("Each evidence item is an object", prompt)
        self.assertNotIn("preconditions", prompt)
        self.assertNotIn("postconditions", prompt)


if __name__ == "__main__":
    unittest.main()
