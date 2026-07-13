from __future__ import annotations

from dataclasses import replace

import pytest

from skillfabric.compiled_graph.models import EvidenceRef
from skillfabric.compiled_graph.semantic.models import CandidateHit, CandidatePair
from skillfabric.compiled_graph.semantic.projection import (
    DependencyCycleError,
    project_relation_decisions,
)
from skillfabric.compiled_graph.semantic.validation import validate_candidate_pairs
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
        "skill:consumer",
        "skill:producer",
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
        source_skill=min(first.candidate.key),
        target_skill=max(first.candidate.key),
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
