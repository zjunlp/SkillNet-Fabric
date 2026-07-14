from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from skillfabric.compiled_graph.contracts.models import SkillContract
from skillfabric.compiled_graph.semantic.models import CandidatePair, RelationDecision
from skillfabric.registry.models import SkillNode


@dataclass
class StaticContractExtractor:
    model_id: str
    responses: dict[str, dict[str, Any]]

    def extract(self, skill: SkillNode) -> dict[str, Any]:
        return dict(self.responses[skill.id])


@dataclass
class StaticRelationJudge:
    model_id: str
    responses: dict[tuple[str, str], dict[str, Any]]
    calls: list[tuple[tuple[str, str], ...]] = field(default_factory=list)

    def judge(
        self,
        pairs: tuple[CandidatePair, ...],
        skills: dict[str, SkillNode],
        contracts: dict[str, SkillContract],
    ) -> dict[str, Any]:
        del skills, contracts
        self.calls.append(tuple(pair.key for pair in pairs))
        return {"decisions": [dict(self.responses[pair.key]) for pair in pairs]}


@dataclass
class StaticCycleAdjudicator:
    replacements: dict[tuple[str, str], RelationDecision]

    def adjudicate(
        self,
        decisions: tuple[RelationDecision, ...],
        skills: dict[str, SkillNode],
    ) -> list[RelationDecision]:
        del skills
        return [self.replacements.get(decision.candidate.key, decision) for decision in decisions]


@dataclass
class FixtureContractExtractor:
    model_id: str = "fixture-contract-model"
    calls: list[str] = field(default_factory=list)

    def extract(self, skill: SkillNode) -> dict[str, Any]:
        self.calls.append(skill.id)
        requires: list[tuple[str, str]] = []
        produces: list[tuple[str, str]] = []
        if skill.id == "skill:pdf-table-parser":
            produces = [("normalized_csv_table", "produces `.csv` files")]
        elif skill.id == "skill:financial-kpi-extractor":
            requires = [("normalized_csv_table", "after `pdf-table-parser`")]
            produces = [("financial_kpi_json", "output `kpi.json`")]
        elif skill.id == "skill:report-writer":
            requires = [("financial_kpi_json", "Use KPI JSON")]
            produces = [("markdown_report", "final `.md` report")]
        elif skill.id == "skill:webshop-product-search":
            produces = [("candidate_products", "search results feed downstream")]
        elif skill.id == "skill:webshop-product-evaluator":
            requires = [("candidate_products", "after `webshop-product-search`")]
        evidence_line, _ = _find_line(skill, skill.description)
        return {
            "capability": skill.description,
            "when_to_use": skill.description,
            "requires": [_field(skill, name, needle) for name, needle in requires],
            "produces": [_field(skill, name, needle) for name, needle in produces],
            "tools": [],
            "evidence": [{"line": evidence_line}],
        }


@dataclass
class FixtureRelationJudge:
    model_id: str = "fixture-relation-model"
    calls: list[tuple[str, str]] = field(default_factory=list)

    def judge(
        self,
        pairs: tuple[CandidatePair, ...],
        skills: dict[str, SkillNode],
        contracts: dict[str, SkillContract],
    ) -> dict[str, Any]:
        del contracts
        decisions = []
        for pair in pairs:
            self.calls.append(pair.key)
            relation = _KNOWN_RELATIONS.get(pair.key)
            if relation is None:
                decisions.append(
                    {
                        "relation": "none",
                        "source_skill": pair.skill_a,
                        "target_skill": pair.skill_b,
                        "confidence": 0.96,
                        "reason": "The fixture sources do not establish an operational relation.",
                        "evidence": [],
                    }
                )
                continue
            relation_type, source, target, source_needle, target_needle = relation
            source_line, _ = _find_line(skills[source], source_needle)
            target_line, _ = _find_line(skills[target], target_needle)
            decisions.append(
                {
                    "relation": relation_type,
                    "source_skill": source,
                    "target_skill": target,
                    "confidence": 0.94,
                    "reason": "The fixture sources explicitly support this operational relation.",
                    "evidence": [
                        {"skill": source, "line": source_line},
                        {"skill": target, "line": target_line},
                    ],
                }
            )
        return {"decisions": decisions}


_KNOWN_RELATIONS = {
    ("skill:financial-kpi-extractor", "skill:pdf-table-parser"): (
        "depend_on",
        "skill:pdf-table-parser",
        "skill:financial-kpi-extractor",
        "produces `.csv` files",
        "after `pdf-table-parser`",
    ),
    ("skill:financial-kpi-extractor", "skill:report-writer"): (
        "depend_on",
        "skill:financial-kpi-extractor",
        "skill:report-writer",
        "output `kpi.json`",
        "Use KPI JSON",
    ),
    ("skill:webshop-product-evaluator", "skill:webshop-product-search"): (
        "depend_on",
        "skill:webshop-product-search",
        "skill:webshop-product-evaluator",
        "search results feed downstream",
        "after `webshop-product-search`",
    ),
    ("skill:analyze-ci", "skill:testing-python"): (
        "compose_with",
        "skill:analyze-ci",
        "skill:testing-python",
        "Inspect GitHub Actions logs",
        "composes with `analyze-ci`",
    ),
}


def _field(skill: SkillNode, name: str, needle: str) -> dict[str, Any]:
    line, _ = _find_line(skill, needle)
    return {
        "name": name,
        "description": f"Concrete reusable {name} value.",
        "evidence": [{"line": line}],
    }


def _find_line(
    skill: SkillNode,
    needle: str,
) -> tuple[int, str]:
    for line_number, line in enumerate(skill.raw_text.splitlines(), start=1):
        if needle.lower() in line.lower():
            return line_number, line
    raise AssertionError(f"fixture evidence not found for {skill.id}: {needle}")
