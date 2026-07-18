from __future__ import annotations

import json
from dataclasses import replace

import pytest

import skillfabric.compiled_graph.semantic.projection as projection_module
from skillfabric.compiled_graph.models import EvidenceRef
from skillfabric.compiled_graph.semantic.models import CandidateHit, CandidatePair
from skillfabric.compiled_graph.semantic.projection import (
    DependencyCycleError,
    LiteLLMCycleAdjudicator,
    project_relation_decisions,
)
from skillfabric.compiled_graph.semantic.prompts import build_cycle_adjudication_messages
from skillfabric.compiled_graph.semantic.validation import validate_candidate_pairs
from skillfabric.runtime.llm import LLMConfig
from tests.unit.relation_helpers import make_skill
from tests.unit.semantic_fixtures import StaticCycleAdjudicator, StaticRelationJudge
from tests.unit.semantic_helpers import (
    dependency_payload,
    semantic_pair,
    semantic_skills_and_contracts,
)


def _dependency_decision():
    skills, contracts = semantic_skills_and_contracts()
    return validate_candidate_pairs(
        [semantic_pair()],
        skills,
        contracts,
        judge=StaticRelationJudge(
            model_id="relation-test-model",
            responses={semantic_pair().key: dependency_payload()},
        ),
    )[0]


def test_projection_writes_one_evidence_grounded_edge_without_weight() -> None:
    skills, _contracts = semantic_skills_and_contracts()

    result = project_relation_decisions([_dependency_decision()], skills)

    assert len(result.edges) == 1
    edge = result.edges[0]
    assert (edge.source, edge.target, edge.type) == (
        "skill:producer",
        "skill:consumer",
        "depend_on",
    )
    assert set(edge.to_dict()) == {
        "source",
        "target",
        "type",
        "confidence",
        "evidence",
        "reason",
    }


def test_duplicate_pair_decisions_are_rejected() -> None:
    skills, _contracts = semantic_skills_and_contracts()
    decision = _dependency_decision()

    with pytest.raises(ValueError, match="one decision"):
        project_relation_decisions([decision, decision], skills)


def test_dependency_cycle_requires_adjudication() -> None:
    skills, decisions = _cyclic_decisions()

    with pytest.raises(DependencyCycleError, match="requires adjudication"):
        project_relation_decisions(decisions, skills)


def test_cycle_adjudicator_can_reclassify_one_edge() -> None:
    skills, decisions = _cyclic_decisions()
    first = decisions[0]
    replacement = replace(
        first,
        relation="compose_with",
        reason="The capabilities compose but neither is a hard prerequisite.",
    )
    adjudicator = StaticCycleAdjudicator(
        replacements={first.candidate.key: replacement},
    )

    result = project_relation_decisions(
        decisions,
        skills,
        cycle_adjudicator=adjudicator,
    )

    assert result.cycle_review_count == 1
    assert {edge.type for edge in result.edges} == {"depend_on", "compose_with"}


def test_adjudicator_that_preserves_cycle_fails_closed() -> None:
    skills, decisions = _cyclic_decisions()
    adjudicator = StaticCycleAdjudicator(
        replacements={decision.candidate.key: decision for decision in decisions},
    )

    with pytest.raises(DependencyCycleError, match="unresolved"):
        project_relation_decisions(
            decisions,
            skills,
            cycle_adjudicator=adjudicator,
        )


def test_llm_cycle_adjudicator_uses_pair_local_output(monkeypatch) -> None:
    skills, decisions = _cyclic_decisions()
    raw_decisions = [
        {
            "pair_index": index,
            "relation": "none",
            "direction": "symmetric",
            "confidence": 0.95,
            "reason": "The apparent dependency is not required.",
            "evidence": {"skill_a_lines": [], "skill_b_lines": []},
        }
        for index in range(len(decisions))
    ]
    monkeypatch.setattr(
        projection_module,
        "litellm_completion",
        lambda **_kwargs: {"content": json.dumps({"decisions": raw_decisions})},
    )
    adjudicator = LiteLLMCycleAdjudicator(
        LLMConfig(
            api_base="https://example.test/v1",
            api_key="test-key",
            model="openai/responses/gpt-5.6-luna",
        )
    )

    replacements = adjudicator.adjudicate(
        tuple(decisions),
        {skill.id: skill for skill in skills},
    )

    assert [decision.candidate.key for decision in replacements] == [
        decision.candidate.key for decision in decisions
    ]
    assert all(decision.relation == "none" for decision in replacements)
    assert all(decision.evidence == () for decision in replacements)


def test_cycle_prompt_excludes_blank_source_lines() -> None:
    skills, decisions = _cyclic_decisions()
    skills[1].raw_text = "Skill B source.\n\nAdditional evidence."

    user_prompt = build_cycle_adjudication_messages(
        tuple(decisions),
        {skill.id: skill for skill in skills},
    )[1]["content"]
    serialized_sources = user_prompt.split("<skill_sources>\n", 1)[1].split(
        "\n</skill_sources>", 1
    )[0]
    sources = json.loads(serialized_sources)

    assert sources["skill:b"] == [
        {"line": 1, "text": "Skill B source."},
        {"line": 3, "text": "Additional evidence."},
    ]


def test_cycle_prompt_limits_evidence_to_listed_source_lines() -> None:
    skills, decisions = _cyclic_decisions()

    user_prompt = build_cycle_adjudication_messages(
        tuple(decisions),
        {skill.id: skill for skill in skills},
    )[1]["content"]

    assert "Cite only line numbers explicitly listed" in user_prompt


def _cyclic_decisions():
    skills = [
        make_skill("skill:a", "a", "Skill A source."),
        make_skill("skill:b", "b", "Skill B source."),
        make_skill("skill:c", "c", "Skill C source."),
    ]
    decisions = []
    for source, target in (("skill:a", "skill:b"), ("skill:b", "skill:c"), ("skill:c", "skill:a")):
        pair_key = tuple(sorted((source, target)))
        pair = CandidatePair(
            skill_a=pair_key[0],
            skill_b=pair_key[1],
            hits=(
                CandidateHit(
                    channel="handoff",
                    query_skill=target,
                    matched_skill=source,
                    rank=1,
                ),
            ),
        )
        decisions.append(
            replace(
                _dependency_decision(),
                candidate=pair,
                relation="depend_on",
                source_skill=source,
                target_skill=target,
                evidence=(
                    EvidenceRef(skill=source, line=1, text=f"Skill {source[-1].upper()} source."),
                    EvidenceRef(skill=target, line=1, text=f"Skill {target[-1].upper()} source."),
                ),
            )
        )
    return skills, decisions
