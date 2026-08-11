from __future__ import annotations

import hashlib
import math
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from skillfabric.compiled_graph.builder import BuildConfig, _BuildDependencies, build_graph
from skillfabric.compiled_graph.contracts.models import SkillContract
from skillfabric.compiled_graph.models import EvidenceRef
from skillfabric.compiled_graph.semantic.models import CandidateHit, CandidatePair
from skillfabric.registry.models import SkillNode
from skillfabric.wiki.materializer import build_wiki
from skillfabric.wiki.models import WikiBuildConfig


class FakeEmbeddingProvider:
    """Provide stable local embeddings without network access."""

    model_id = "test-fake-embedding"

    def __init__(self, dimension: int = 96) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in re.findall(r"[a-z0-9][a-z0-9_.+-]*", text.lower()):
            digest = hashlib.sha256(token.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            vector[index] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return vector if norm == 0 else [value / norm for value in vector]

    def embed_query(self, query: str) -> list[float]:
        return self.embed(query)


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
        evidence_line = _line_number(skill, skill.description)
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
        for pair_index, pair in enumerate(pairs):
            self.calls.append(pair.key)
            relation = _KNOWN_RELATIONS.get(pair.key)
            if relation is None:
                decisions.append(
                    {
                        "pair_index": pair_index,
                        "relation": "none",
                        "direction": "symmetric",
                        "confidence": 0.96,
                        "reason": "The sources do not establish an operational relation.",
                        "evidence": {"skill_a_lines": [], "skill_b_lines": []},
                    }
                )
                continue
            relation_type, source, target, source_text, target_text = relation
            lines = {
                source: [_line_number(skills[source], source_text)],
                target: [_line_number(skills[target], target_text)],
            }
            decisions.append(
                {
                    "pair_index": pair_index,
                    "relation": relation_type,
                    "direction": (
                        "skill_a_to_skill_b" if source == pair.skill_a else "skill_b_to_skill_a"
                    ),
                    "confidence": 0.94,
                    "reason": "The sources explicitly support this operational relation.",
                    "evidence": {
                        "skill_a_lines": lines[pair.skill_a],
                        "skill_b_lines": lines[pair.skill_b],
                    },
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


_SKILL_SOURCES = {
    "analyze-ci": (
        "Analyze failed CI logs and summarize root causes.",
        "Inspect GitHub Actions logs and explain root causes.",
    ),
    "financial-kpi-extractor": (
        "Extract financial KPI values from CSV tables.",
        "Use after `pdf-table-parser`, then output `kpi.json`.",
    ),
    "pdf-table-parser": (
        "Extract tables from PDF files and save structured CSV output.",
        "Parse PDF pages. This skill produces `.csv` files for downstream analysis.",
    ),
    "report-writer": (
        "Write concise Markdown reports from JSON metrics.",
        "Use KPI JSON to compose a final `.md` report.",
    ),
    "testing-python": (
        "Run and diagnose Python test failures.",
        "Use pytest; this composes with `analyze-ci` when CI fails.",
    ),
    "webshop-product-evaluator": (
        "Evaluate product candidates against constraints.",
        "Use after `webshop-product-search` returns candidates.",
    ),
    "webshop-product-search": (
        "Search product listings from shopping criteria.",
        "Return candidate products; search results feed downstream product evaluation.",
    ),
}

_TEMP_ROOT = TemporaryDirectory(prefix="skillfabric-tests-")
FIXTURE_SKILLS = Path(_TEMP_ROOT.name) / "skills"
_WORKSPACE_CACHE = Path(_TEMP_ROOT.name) / "workspace"


def write_skill(
    root: Path,
    name: str,
    description: str,
    body: str,
) -> Path:
    """Write one native skill under a temporary test root."""

    path = root / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n\n{body}\n",
        encoding="utf-8",
    )
    return path


def build_fixture_workspace(workspace: Path) -> None:
    """Copy a cached workspace built only from deterministic local fakes."""

    _ensure_fixture_workspace()
    if workspace.exists():
        shutil.rmtree(workspace)
    shutil.copytree(_WORKSPACE_CACHE, workspace)


def make_skill(skill_id: str, name: str, raw_text: str) -> SkillNode:
    return SkillNode(
        id=skill_id,
        type="skill",
        name=name,
        description=f"{name} description",
        content_hash=f"hash-{name}",
        raw_text=raw_text,
    )


def semantic_skills_and_contracts() -> tuple[list[SkillNode], dict[str, SkillContract]]:
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
            produces=["normalized_table"],
        ),
        consumer.id: _contract(
            consumer,
            capability="Write a report from a normalized table.",
            requires=["normalized_table"],
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
        "pair_index": 0,
        "relation": "depend_on",
        "direction": "skill_b_to_skill_a",
        "confidence": confidence,
        "reason": "The producer supplies the table consumed by the consumer.",
        "evidence": {"skill_a_lines": [1], "skill_b_lines": [1]},
    }


def none_payload(*, pair_index: int = 0, confidence: float = 0.99) -> dict[str, Any]:
    return {
        "pair_index": pair_index,
        "relation": "none",
        "direction": "symmetric",
        "confidence": confidence,
        "reason": "The skills are unrelated.",
        "evidence": {"skill_a_lines": [], "skill_b_lines": []},
    }


def _ensure_fixture_workspace() -> None:
    if _WORKSPACE_CACHE.exists():
        return
    build_graph(
        BuildConfig(skill_root=FIXTURE_SKILLS, workspace=_WORKSPACE_CACHE),
        dependencies=_BuildDependencies(
            contract_extractor=FixtureContractExtractor(),
            relation_judge=FixtureRelationJudge(),
            embedding_provider=FakeEmbeddingProvider(),
            build_id="test-build",
        ),
    )
    build_wiki(WikiBuildConfig(workspace=_WORKSPACE_CACHE))


def _field(skill: SkillNode, name: str, text: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": f"Concrete reusable {name} value.",
        "evidence": [{"line": _line_number(skill, text)}],
    }


def _line_number(skill: SkillNode, text: str) -> int:
    for line_number, line in enumerate(skill.raw_text.splitlines(), start=1):
        if text.lower() in line.lower():
            return line_number
    raise AssertionError(f"evidence not found for {skill.id}: {text}")


def _contract(
    skill: SkillNode,
    *,
    capability: str,
    requires: list[str] | None = None,
    produces: list[str] | None = None,
) -> SkillContract:
    def fields(values: list[str]) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "description": f"Concrete field {name}.",
                "evidence": [{"line": 1}],
            }
            for name in values
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


for _name, (_description, _body) in _SKILL_SOURCES.items():
    write_skill(FIXTURE_SKILLS, _name, _description, _body)
