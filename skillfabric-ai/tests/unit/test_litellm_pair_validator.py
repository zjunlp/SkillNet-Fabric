from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from skillfabric.compiled_graph.relations.models import CandidatePair
from skillfabric.compiled_graph.relations.prompts import build_pair_validation_messages
from skillfabric.compiled_graph.relations.validation import (
    LiteLLMPairValidator,
    validate_relation_candidates,
)
from skillfabric.llm import LLMConfig
from skillfabric.registry.models import SkillNode


def _skill(
    skill_id: str,
    name: str,
    raw_text: str,
) -> SkillNode:
    return SkillNode(
        id=skill_id,
        type="skill",
        name=name,
        description=f"{name} description",
        source_path=f"/tmp/{name}/SKILL.md",
        wiki_path=f"/tmp/wiki/{name}.md",
        content_hash=f"hash-{name}",
        token_count=len(raw_text.split()),
        canonical_skill_text_hash=f"canonical-{name}",
        raw_text=raw_text,
    )


class LiteLLMPairValidatorTests(unittest.TestCase):
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

    def test_accepts_compose_with_json_response(self) -> None:
        self._install_fake_litellm(
            json.dumps(
                {
                    "edge_type": "compose_with",
                    "direction": "undirected",
                    "confidence": 0.81,
                    "evidence": [{"skill": "skill:a", "line": 1, "text": "A composes with B."}],
                    "reason": "The skills are complementary.",
                }
            )
        )
        skill_a = _skill("skill:a", "alpha", "A composes with B.")
        skill_b = _skill("skill:b", "beta", "B handles the follow-up.")
        pair = CandidatePair("skill:a", "skill:b", 1.0, ["explicit_mention"])

        results = validate_relation_candidates(
            [pair],
            [skill_a, skill_b],
            validator=LiteLLMPairValidator(
                config=LLMConfig(api_base="https://example.test/api", api_key="sk-test")
            ),
        )

        self.assertTrue(results[0].accepted)
        self.assertIsNotNone(results[0].edge)
        self.assertEqual(results[0].edge.type, "compose_with")

    def test_accepts_depend_on_fenced_json_response_and_direction(self) -> None:
        self._install_fake_litellm(
            "```json\n"
            + json.dumps(
                {
                    "edge_type": "depend_on",
                    "direction": "B->A",
                    "confidence": 0.91,
                    "evidence": [{"skill": "skill:b", "line": 2, "text": "B requires A output."}],
                    "reason": "B consumes A output.",
                }
            )
            + "\n```"
        )
        skill_a = _skill("skill:a", "extractor", "A produces CSV.")
        skill_b = _skill("skill:b", "reporter", "B requires A output.")
        pair = CandidatePair("skill:a", "skill:b", 0.9, ["similar_neighbor"])

        results = validate_relation_candidates(
            [pair],
            [skill_a, skill_b],
            validator=LiteLLMPairValidator(
                config=LLMConfig(api_base="https://example.test/api", api_key="sk-test")
            ),
        )

        self.assertTrue(results[0].accepted)
        self.assertEqual(results[0].edge.source, "skill:b")
        self.assertEqual(results[0].edge.target, "skill:a")
        self.assertEqual(results[0].edge.type, "depend_on")

    def test_rejects_invalid_json_response_without_edge(self) -> None:
        self._install_fake_litellm("not json")
        skill_a = _skill("skill:a", "alpha", "A text.")
        skill_b = _skill("skill:b", "beta", "B text.")
        pair = CandidatePair("skill:a", "skill:b", 0.78, ["similar_neighbor"])

        results = validate_relation_candidates(
            [pair],
            [skill_a, skill_b],
            validator=LiteLLMPairValidator(
                config=LLMConfig(api_base="https://example.test/api", api_key="sk-test")
            ),
        )

        self.assertFalse(results[0].accepted)
        self.assertIsNone(results[0].edge)
        self.assertEqual(results[0].raw_output["edge_type"], "none")
        self.assertEqual(results[0].raw_output["error_type"], "json_parse_error")

    def test_rejects_api_exception_without_edge(self) -> None:
        self._install_fake_litellm(RuntimeError("network down"))
        skill_a = _skill("skill:a", "alpha", "A text.")
        skill_b = _skill("skill:b", "beta", "B text.")
        pair = CandidatePair("skill:a", "skill:b", 0.78, ["similar_neighbor"])

        results = validate_relation_candidates(
            [pair],
            [skill_a, skill_b],
            validator=LiteLLMPairValidator(
                config=LLMConfig(api_base="https://example.test/api", api_key="sk-test")
            ),
        )

        self.assertFalse(results[0].accepted)
        self.assertIsNone(results[0].edge)
        self.assertEqual(results[0].raw_output["error_type"], "api_error")
        self.assertIn("network down", results[0].raw_output["reason"])

    def test_rejects_schema_invalid_confidence_without_crashing(self) -> None:
        self._install_fake_litellm(
            json.dumps(
                {
                    "edge_type": "depend_on",
                    "direction": "A->B",
                    "confidence": "high",
                    "evidence": [{"skill": "skill:a", "line": "bad", "text": "A produces output."}],
                    "reason": "Bad schema.",
                }
            )
        )
        skill_a = _skill("skill:a", "alpha", "A produces output.")
        skill_b = _skill("skill:b", "beta", "B consumes output.")
        pair = CandidatePair("skill:a", "skill:b", 0.9, ["similar_neighbor"])

        results = validate_relation_candidates(
            [pair],
            [skill_a, skill_b],
            validator=LiteLLMPairValidator(
                config=LLMConfig(api_base="https://example.test/api", api_key="sk-test")
            ),
        )

        self.assertFalse(results[0].accepted)
        self.assertIn("schema_error", results[0].rejection_reason)

    def test_rejects_schema_invalid_falsy_confidence(self) -> None:
        self._install_fake_litellm(
            json.dumps(
                {
                    "edge_type": "compose_with",
                    "direction": "undirected",
                    "confidence": [],
                    "evidence": [{"skill": "skill:a", "line": 1, "text": "A composes with B."}],
                    "reason": "Bad schema.",
                }
            )
        )
        skill_a = _skill("skill:a", "alpha", "A composes with B.")
        skill_b = _skill("skill:b", "beta", "B works with A.")
        pair = CandidatePair("skill:a", "skill:b", 0.9, ["similar_neighbor"])

        results = validate_relation_candidates(
            [pair],
            [skill_a, skill_b],
            validator=LiteLLMPairValidator(
                config=LLMConfig(api_base="https://example.test/api", api_key="sk-test")
            ),
        )

        self.assertFalse(results[0].accepted)
        self.assertIn("schema_error", results[0].rejection_reason)

    def test_prompt_includes_full_skill_md_by_default(self) -> None:
        raw_text = "\n".join(
            [
                "Line 1 mentions csv output.",
                "Line 2 is useful evidence for beta.",
                "FULL_TEXT_LINE_SHOULD_APPEAR",
            ]
        )
        skill_a = _skill(
            "skill:a",
            "alpha",
            raw_text,
        )
        skill_b = _skill("skill:b", "beta", "Beta consumes alpha output.")
        messages = build_pair_validation_messages(
            skill_a,
            skill_b,
            CandidatePair("skill:a", "skill:b", 0.75, ["interface_compatibility"]),
        )
        prompt_text = json.dumps(messages, ensure_ascii=False)

        self.assertIn("Line 1 mentions csv output.", prompt_text)
        self.assertIn("FULL_TEXT_LINE_SHOULD_APPEAR", prompt_text)
        self.assertIn("full_skill_md", prompt_text)
        self.assertIn("A depend_on B means skill A requires skill B to run first", prompt_text)
        self.assertIn("producer -> consumer compatibility implies consumer depend_on producer", prompt_text)
        self.assertIn("belief_state or planning_state is not a world-state producer", prompt_text)
        self.assertIn("object_permanence_state", prompt_text)

    def test_validation_cache_prevents_repeated_litellm_calls(self) -> None:
        calls = self._install_fake_litellm(
            json.dumps(
                {
                    "edge_type": "none",
                    "direction": "none",
                    "confidence": 0.0,
                    "evidence": [],
                    "reason": "No relation.",
                }
            )
        )
        skill_a = _skill("skill:a", "alpha", "A text.")
        skill_b = _skill("skill:b", "beta", "B text.")
        pair = CandidatePair("skill:a", "skill:b", 0.78, ["similar_neighbor"])

        with TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache.json"
            validator = LiteLLMPairValidator(
                config=LLMConfig(api_base="https://example.test/api", api_key="sk-test")
            )
            validate_relation_candidates([pair], [skill_a, skill_b], validator=validator, cache_path=cache_path)
            validate_relation_candidates([pair], [skill_a, skill_b], validator=validator, cache_path=cache_path)

        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
