from __future__ import annotations

from typing import Any

from skillfabric.compiled_graph.contracts.models import SkillContract
from skillfabric.compiled_graph.models import EvidenceRef
from skillfabric.compiled_graph.semantic.models import CandidateHit, CandidatePair
from tests.unit.relation_helpers import make_skill


def semantic_skills_and_contracts():
    producer = make_skill(
        "skill:producer",
        "producer",
        "Produces a normalized table.\nUse the parser command.",
    )
    consumer = make_skill(
        "skill:consumer",
        "consumer",
        "Requires the normalized table.\nWrites the final report.",
    )
    contracts = {
        producer.id: _contract(
            producer,
            capability="Produce a normalized table.",
            produces=[("normalized_table", 1, "Produces a normalized table.")],
        ),
        consumer.id: _contract(
            consumer,
            capability="Write a report from a normalized table.",
            requires=[("normalized_table", 1, "Requires the normalized table.")],
        ),
    }
    return [producer, consumer], contracts


def semantic_pair() -> CandidatePair:
    return CandidatePair(
        skill_a="skill:consumer",
        skill_b="skill:producer",
        hits=(
            CandidateHit(
                channel="handoff",
                query_skill="skill:producer",
                matched_skill="skill:consumer",
                rank=1,
                query_field="produces:normalized_table",
                matched_field="requires:normalized_table",
                evidence=(
                    EvidenceRef(
                        skill="skill:producer",
                        line=1,
                        text="Produces a normalized table.",
                    ),
                    EvidenceRef(
                        skill="skill:consumer",
                        line=1,
                        text="Requires the normalized table.",
                    ),
                ),
            ),
        ),
    )


def dependency_payload(*, confidence: float = 0.91) -> dict[str, Any]:
    return {
        "relation": "depend_on",
        "source_skill": "skill:consumer",
        "target_skill": "skill:producer",
        "confidence": confidence,
        "reason": "The consumer requires the normalized table produced by the producer.",
        "evidence": [
            {"skill": "skill:consumer", "line": 1},
            {"skill": "skill:producer", "line": 1},
        ],
    }


def _contract(
    skill,
    *,
    capability: str,
    requires: list[tuple[str, int, str]] | None = None,
    produces: list[tuple[str, int, str]] | None = None,
) -> SkillContract:
    def fields(values: list[tuple[str, int, str]]) -> list[dict[str, object]]:
        return [
            {
                "name": name,
                "description": f"Concrete field {name}.",
                "evidence": [{"line": line}],
            }
            for name, line, _text in values
        ]

    return SkillContract.from_extraction(
        skill,
        {
            "capability": capability,
            "when_to_use": f"Use {skill.name} for the documented task.",
            "requires": fields(requires or []),
            "produces": fields(produces or []),
            "tools": [],
            "evidence": [{"line": 1}],
        },
    )
