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


@pytest.mark.parametrize(
    "replacement",
    [
        lambda original: replace(
            original,
            source_skill=original.target_skill,
            target_skill=original.source_skill,
        ),
        lambda original: replace(original, evidence=tuple(reversed(original.evidence))),
        lambda original: replace(
            original,
            relation="similar_to",
            source_skill=original.candidate.skill_a,
            target_skill=original.candidate.skill_b,
        ),
    ],
)
def test_cycle_adjudicator_cannot_rewrite_validated_relation_fields(replacement) -> None:
    skills, decisions = _cyclic_decisions()
    first = decisions[0]
    adjudicator = StaticCycleAdjudicator(
        replacements={first.candidate.key: replacement(first)},
    )

    with pytest.raises(DependencyCycleError, match="monotonically weaken"):
        project_relation_decisions(
            decisions,
            skills,
            cycle_adjudicator=adjudicator,
        )


def test_llm_cycle_adjudicator_applies_monotonic_actions(monkeypatch) -> None:
    skills, decisions = _cyclic_decisions()
    raw_decisions = [
        {
            "pair_index": 0,
            "action": "keep",
            "confidence": 0.91,
            "reason": "The hard handoff remains source-grounded.",
        },
        {
            "pair_index": 1,
            "action": "downgrade_to_compose",
            "confidence": 0.87,
            "reason": "The stages are adjacent but the handoff is optional.",
        },
        {
            "pair_index": 2,
            "action": "remove",
            "confidence": 0.96,
            "reason": "The apparent dependency is unsupported.",
        },
    ]
    adjudicator = _llm_cycle_adjudicator(monkeypatch, raw_decisions)

    replacements = adjudicator.adjudicate(
        tuple(decisions),
        {skill.id: skill for skill in skills},
    )

    kept, downgraded, removed = replacements
    assert kept.candidate == decisions[0].candidate
    assert (kept.relation, kept.source_skill, kept.target_skill, kept.evidence) == (
        "depend_on",
        decisions[0].source_skill,
        decisions[0].target_skill,
        decisions[0].evidence,
    )
    assert (kept.confidence, kept.reason) == (0.91, raw_decisions[0]["reason"])
    assert downgraded.candidate == decisions[1].candidate
    assert (
        downgraded.relation,
        downgraded.source_skill,
        downgraded.target_skill,
        downgraded.evidence,
    ) == (
        "compose_with",
        decisions[1].source_skill,
        decisions[1].target_skill,
        decisions[1].evidence,
    )
    assert (downgraded.confidence, downgraded.reason) == (
        0.87,
        raw_decisions[1]["reason"],
    )
    assert removed.candidate == decisions[2].candidate
    assert (removed.relation, removed.source_skill, removed.target_skill, removed.evidence) == (
        "none",
        decisions[2].candidate.skill_a,
        decisions[2].candidate.skill_b,
        (),
    )
    assert (removed.confidence, removed.reason) == (0.96, raw_decisions[2]["reason"])


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda rows: [{**rows[0], "unexpected": True}, *rows[1:]], "unexpected keys"),
        (lambda rows: [{**rows[0], "action": "reverse"}, *rows[1:]], "action must be"),
        (lambda rows: [{**rows[0], "confidence": True}, *rows[1:]], "confidence must be"),
        (lambda rows: [{**rows[0], "confidence": 1.1}, *rows[1:]], "between 0 and 1"),
        (lambda rows: [{**rows[0], "reason": " "}, *rows[1:]], "reason must be"),
        (lambda rows: [rows[0], {**rows[1], "pair_index": 0}, rows[2]], "duplicate"),
        (lambda rows: rows[:-1], "replace every cycle pair"),
        (lambda rows: [{**rows[0], "pair_index": 3}, *rows[1:]], "unknown pair_index"),
    ],
)
def test_llm_cycle_adjudicator_rejects_invalid_actions(monkeypatch, mutator, message) -> None:
    skills, decisions = _cyclic_decisions()
    rows = [
        {
            "pair_index": index,
            "action": "keep",
            "confidence": 0.9,
            "reason": "The dependency remains supported.",
        }
        for index in range(len(decisions))
    ]
    adjudicator = _llm_cycle_adjudicator(monkeypatch, mutator(rows))

    with pytest.raises(DependencyCycleError, match=message):
        adjudicator.adjudicate(
            tuple(decisions),
            {skill.id: skill for skill in skills},
        )


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


def test_cycle_prompt_uses_action_only_output() -> None:
    skills, decisions = _cyclic_decisions()

    user_prompt = build_cycle_adjudication_messages(
        tuple(decisions),
        {skill.id: skill for skill in skills},
    )[1]["content"]
    serialized_schema = user_prompt.split("<output_schema>\n", 1)[1].split(
        "\n</output_schema>", 1
    )[0]
    schema = json.loads(serialized_schema)

    assert schema == {
        "decisions": [
            {
                "pair_index": 0,
                "action": "keep|downgrade_to_compose|remove",
                "confidence": 0.0,
                "reason": "concise evidence-grounded explanation",
            }
        ]
    }
    assert "Do not generate skill ids, directions, relations, or evidence" in user_prompt


def _llm_cycle_adjudicator(monkeypatch, raw_decisions) -> LiteLLMCycleAdjudicator:
    monkeypatch.setattr(
        projection_module,
        "litellm_completion",
        lambda **_kwargs: {"content": json.dumps({"decisions": raw_decisions})},
    )
    return LiteLLMCycleAdjudicator(
        LLMConfig(
            api_base="https://example.test/v1",
            api_key="test-key",
            model="openai/responses/gpt-5.6-luna",
        )
    )


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
