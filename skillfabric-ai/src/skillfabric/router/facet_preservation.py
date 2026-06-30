"""Facet-aware candidate preservation for query-local router bundles."""

from __future__ import annotations

import math
import re
from pathlib import Path

from skillfabric.compiled_graph.interface.models import SkillInterface
from skillfabric.indexing.canonical import canonical_skill_text
from skillfabric.registry.models import SkillNode
from skillfabric.router.models import RouterSkillCandidate
from skillfabric.task_understanding import TaskUnderstanding
from skillfabric.wiki.pages import slug

_STOP_TERMS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "being",
    "build",
    "by",
    "contain",
    "create",
    "deliver",
    "each",
    "file",
    "files",
    "for",
    "from",
    "generate",
    "have",
    "i",
    "in",
    "input",
    "is",
    "it",
    "make",
    "name",
    "of",
    "on",
    "or",
    "output",
    "please",
    "produce",
    "result",
    "results",
    "same",
    "should",
    "start",
    "task",
    "their",
    "the",
    "to",
    "use",
    "using",
    "want",
    "with",
}

_INTERFACE_LOW_VALUE_TERMS = {
    "artifact",
    "assistant",
    "count",
    "data",
    "display",
    "element",
    "green",
    "helper",
    "json",
    "mp4",
    "red",
    "yellow",
}


def preserve_facet_candidates(
    selected: list[RouterSkillCandidate],
    *,
    seed_scores: dict[str, RouterSkillCandidate],
    skills: dict[str, SkillNode],
    interfaces: dict[str, SkillInterface],
    understanding: TaskUnderstanding,
    hard_include_ids: set[str],
    expanded_limit: int,
    wiki_skills_dir: Path,
    warnings: list[str],
) -> list[RouterSkillCandidate]:
    """Preserve representative candidates for task facets before query_wiki materialization."""

    if expanded_limit <= 0:
        return []
    has_wiki_pages = wiki_skills_dir.exists() and any(wiki_skills_dir.glob("*.md"))
    expanded_ids = {item.skill_id for item in selected}
    candidate_pool = _candidate_pool(selected, seed_scores, skills)
    available_ids = {
        skill_id
        for skill_id, candidate in candidate_pool.items()
        if _has_candidate_evidence(candidate)
        or skill_id in expanded_ids
        or skill_id in hard_include_ids
    }
    selected_ids: list[str] = []
    limit = max(expanded_limit, len(hard_include_ids & available_ids))

    for skill_id in _rank_ids(hard_include_ids & available_ids, candidate_pool):
        _append_selected(selected_ids, skill_id, limit)
    global_budget = max(_global_budget(expanded_limit), len(selected_ids))
    for skill_id in _rank_ids(expanded_ids, candidate_pool):
        if len(selected_ids) >= min(global_budget, limit):
            break
        _append_selected(selected_ids, skill_id, limit)
    for skill_id, source in _coverage_facet_ids(
        understanding,
        skills=skills,
        candidate_pool=candidate_pool,
        expanded_limit=expanded_limit,
        wiki_skills_dir=wiki_skills_dir,
        require_wiki_page=has_wiki_pages,
        warnings=warnings,
    ):
        _preserve_with_source(selected_ids, candidate_pool, skill_id, source, limit)
        available_ids.add(skill_id)
    for skill_id, source in _domain_facet_ids(
        understanding,
        skills=skills,
        interfaces=interfaces,
        candidate_pool=candidate_pool,
        wiki_skills_dir=wiki_skills_dir,
        require_wiki_page=has_wiki_pages,
        warnings=warnings,
    ):
        _preserve_with_source(selected_ids, candidate_pool, skill_id, source, limit)
        available_ids.add(skill_id)
    for skill_id, source in _interface_text_facet_ids(
        understanding,
        skills=skills,
        interfaces=interfaces,
        candidate_pool=candidate_pool,
        wiki_skills_dir=wiki_skills_dir,
        require_wiki_page=has_wiki_pages,
        warnings=warnings,
    ):
        _preserve_with_source(selected_ids, candidate_pool, skill_id, source, limit)
        available_ids.add(skill_id)

    if len(selected_ids) < limit:
        for skill_id in _rank_ids(expanded_ids & available_ids, candidate_pool):
            _append_selected(selected_ids, skill_id, limit)
            if len(selected_ids) >= limit:
                break

    preserved = selected_ids[:limit]
    return sorted(
        (candidate_pool[skill_id] for skill_id in preserved),
        key=lambda item: (-item.score, item.graph_depth, item.skill_id),
    )


def _candidate_pool(
    selected: list[RouterSkillCandidate],
    seed_scores: dict[str, RouterSkillCandidate],
    skills: dict[str, SkillNode],
) -> dict[str, RouterSkillCandidate]:
    output = {item.skill_id: _copy_candidate(item) for item in selected}
    for skill_id, item in seed_scores.items():
        if skill_id in skills:
            output.setdefault(skill_id, _copy_candidate(item))
    for skill_id, skill in skills.items():
        output.setdefault(
            skill_id,
            RouterSkillCandidate(
                skill_id=skill_id,
                name=skill.name,
                score=0.0,
                sources=[],
                graph_depth=99,
                reason="facet candidate",
            ),
        )
    return output


def _copy_candidate(item: RouterSkillCandidate) -> RouterSkillCandidate:
    return RouterSkillCandidate(
        skill_id=item.skill_id,
        name=item.name,
        score=item.score,
        sources=list(item.sources),
        graph_depth=item.graph_depth,
        reason=item.reason,
        seed_score=item.seed_score,
        ppr_score=item.ppr_score,
        score_breakdown=dict(item.score_breakdown),
    )


def _coverage_facet_ids(
    understanding: TaskUnderstanding,
    *,
    skills: dict[str, SkillNode],
    candidate_pool: dict[str, RouterSkillCandidate],
    expanded_limit: int,
    wiki_skills_dir: Path,
    require_wiki_page: bool,
    warnings: list[str],
) -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    for requirement in understanding.coverage_requirements:
        added = 0
        budget = _coverage_budget(requirement.kind, expanded_limit)
        for skill_id in _dedupe([*requirement.preferred_skill_ids, *requirement.acceptable_skill_ids]):
            if skill_id not in skills or skill_id not in candidate_pool:
                continue
            if require_wiki_page and not _has_wiki_page(skill_id, wiki_skills_dir):
                _warn_missing_wiki(skill_id, warnings)
                continue
            output.append((skill_id, f"coverage:{requirement.id}"))
            added += 1
            if added >= budget:
                break
    return output


def _coverage_budget(kind: str, expanded_limit: int) -> int:
    if expanded_limit < 12:
        return 2
    if kind == "intent":
        return min(16, max(4, expanded_limit // 3))
    return min(4, max(2, expanded_limit // 12))


def _domain_facet_ids(
    understanding: TaskUnderstanding,
    *,
    skills: dict[str, SkillNode],
    interfaces: dict[str, SkillInterface],
    candidate_pool: dict[str, RouterSkillCandidate],
    wiki_skills_dir: Path,
    require_wiki_page: bool,
    warnings: list[str],
) -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    for hint in understanding.domain_hints:
        terms = _terms(hint)
        if not terms:
            continue
        ranked = _rank_by_terms(
            terms,
            skills=skills,
            interfaces=interfaces,
            candidate_pool=candidate_pool,
            wiki_skills_dir=wiki_skills_dir,
            require_wiki_page=require_wiki_page,
            warnings=warnings,
            min_matches=1,
        )
        for skill_id, _score in ranked[:2]:
            output.append((skill_id, f"facet:domain:{hint}"))
    return output


def _interface_text_facet_ids(
    understanding: TaskUnderstanding,
    *,
    skills: dict[str, SkillNode],
    interfaces: dict[str, SkillInterface],
    candidate_pool: dict[str, RouterSkillCandidate],
    wiki_skills_dir: Path,
    require_wiki_page: bool,
    warnings: list[str],
) -> list[tuple[str, str]]:
    query_terms = _terms(
        " ".join(
            [
                understanding.query,
                " ".join(understanding.analysis_intents),
                " ".join(understanding.domain_hints),
                " ".join(item.label for item in understanding.required_deliverables),
            ]
        )
    ) - _INTERFACE_LOW_VALUE_TERMS
    ranked = _rank_by_terms(
        query_terms,
        skills=skills,
        interfaces=interfaces,
        candidate_pool=candidate_pool,
        wiki_skills_dir=wiki_skills_dir,
        require_wiki_page=require_wiki_page,
        warnings=warnings,
        min_matches=2,
    )
    exact = _exact_interface_term_ids(
        query_terms,
        skills=skills,
        interfaces=interfaces,
        candidate_pool=candidate_pool,
        wiki_skills_dir=wiki_skills_dir,
        require_wiki_page=require_wiki_page,
        warnings=warnings,
    )
    output: list[tuple[str, str]] = [(skill_id, "facet:exact_interface_term") for skill_id in exact[:6]]
    seen = {skill_id for skill_id, _source in output}
    for skill_id, _score in ranked[:10]:
        if skill_id in seen:
            continue
        output.append((skill_id, "facet:interface_text"))
        seen.add(skill_id)
        if len(output) >= 10:
            break
    return output


def _exact_interface_term_ids(
    query_terms: set[str],
    *,
    skills: dict[str, SkillNode],
    interfaces: dict[str, SkillInterface],
    candidate_pool: dict[str, RouterSkillCandidate],
    wiki_skills_dir: Path,
    require_wiki_page: bool,
    warnings: list[str],
) -> list[str]:
    exact_terms = {
        term
        for term in query_terms
        if len(term) >= 6
        and term not in _STOP_TERMS
        and term not in _INTERFACE_LOW_VALUE_TERMS
        and term not in {"article", "attention", "course", "create", "directory", "output", "source", "student", "video"}
    }
    ranked: list[tuple[str, float]] = []
    for skill_id, skill in skills.items():
        if skill_id not in candidate_pool:
            continue
        if require_wiki_page and not _has_wiki_page(skill_id, wiki_skills_dir):
            _warn_missing_wiki(skill_id, warnings)
            continue
        text_terms = _terms(_facet_text(skill, interfaces.get(skill_id)))
        matches = exact_terms & text_terms
        name_matches = exact_terms & _terms(f"{skill_id} {skill.name}")
        if not matches:
            continue
        score = (2.0 * len(name_matches)) + len(matches)
        ranked.append((skill_id, score))
    return [
        skill_id
        for skill_id, _score in sorted(
            ranked,
            key=lambda item: (-item[1], -candidate_pool[item[0]].score, item[0]),
        )
    ]


def _rank_by_terms(
    terms: set[str],
    *,
    skills: dict[str, SkillNode],
    interfaces: dict[str, SkillInterface],
    candidate_pool: dict[str, RouterSkillCandidate],
    wiki_skills_dir: Path,
    require_wiki_page: bool,
    warnings: list[str],
    min_matches: int,
) -> list[tuple[str, float]]:
    ranked: list[tuple[str, float]] = []
    for skill_id, skill in skills.items():
        if skill_id not in candidate_pool:
            continue
        if require_wiki_page and not _has_wiki_page(skill_id, wiki_skills_dir):
            _warn_missing_wiki(skill_id, warnings)
            continue
        text_terms = _terms(_facet_text(skill, interfaces.get(skill_id)))
        matches = terms & text_terms
        if len(matches) < min_matches:
            continue
        score = len(matches) / math.sqrt(max(len(terms), 1))
        interface = interfaces.get(skill_id)
        if interface is not None and interface.execution_role in {"transformer", "actor", "planner", "verifier"}:
            score += 0.05
        ranked.append((skill_id, score))
    return sorted(
        ranked,
        key=lambda item: (
            -item[1],
            -candidate_pool[item[0]].score,
            candidate_pool[item[0]].graph_depth,
            item[0],
        ),
    )


def _facet_text(skill: SkillNode, interface: SkillInterface | None) -> str:
    parts = [skill.name, skill.description]
    if interface is not None:
        parts.extend(
            [
                interface.capability_summary,
                interface.when_to_use,
                interface.execution_role,
                interface.granularity,
                " ".join(field.name for field in interface.produces),
                " ".join(field.description for field in interface.produces),
                " ".join(field.name for field in interface.uses_tools),
                " ".join(field.description for field in interface.uses_tools),
            ]
        )
    parts.append(canonical_skill_text(skill)[:1200])
    return " ".join(parts)


def _global_budget(expanded_limit: int) -> int:
    if expanded_limit <= 1:
        return expanded_limit
    if expanded_limit < 8:
        return max(1, expanded_limit // 2)
    return min(expanded_limit, max(8, expanded_limit // 3))


def _append_selected(selected_ids: list[str], skill_id: str, limit: int) -> bool:
    if skill_id in selected_ids:
        return False
    if len(selected_ids) >= limit:
        return False
    selected_ids.append(skill_id)
    return True


def _preserve_with_source(
    selected_ids: list[str],
    candidate_pool: dict[str, RouterSkillCandidate],
    skill_id: str,
    source: str,
    limit: int,
) -> None:
    if skill_id in selected_ids:
        _add_source(candidate_pool[skill_id], source)
        return
    if _append_selected(selected_ids, skill_id, limit):
        _add_source(candidate_pool[skill_id], source)


def _rank_ids(skill_ids: set[str], candidates: dict[str, RouterSkillCandidate]) -> list[str]:
    return [
        item.skill_id
        for item in sorted(
            (candidates[skill_id] for skill_id in skill_ids if skill_id in candidates),
            key=lambda item: (-item.score, item.graph_depth, item.skill_id),
        )
    ]


def _add_source(candidate: RouterSkillCandidate, source: str) -> None:
    if source and source not in candidate.sources:
        candidate.sources.append(source)
    if source:
        candidate.score_breakdown[source] = max(candidate.score_breakdown.get(source, 0.0), 0.0)
    if source.startswith("facet:") and candidate.reason in {"", "facet candidate"}:
        candidate.reason = "preserved for task facet coverage"


def _has_candidate_evidence(candidate: RouterSkillCandidate) -> bool:
    if candidate.score > 0 or candidate.seed_score > 0 or candidate.ppr_score > 0:
        return True
    return any(
        source.startswith(("bm25", "lexical", "embedding", "graph:", "ppr:", "coverage:", "facet:"))
        for source in candidate.sources
    )


def _has_wiki_page(skill_id: str, wiki_skills_dir: Path) -> bool:
    return (wiki_skills_dir / f"{slug(skill_id)}.md").exists()


def _warn_missing_wiki(skill_id: str, warnings: list[str]) -> None:
    message = f"facet candidate skipped because wiki page is missing: {skill_id}"
    if message not in warnings:
        warnings.append(message)


def _terms(text: str) -> set[str]:
    output: set[str] = set()
    for raw_token in re.findall(r"[a-z0-9][a-z0-9_.+-]*", text.lower()):
        for token in _token_variants(raw_token):
            if len(token) < 3 or token in _STOP_TERMS:
                continue
            output.add(token)
    return output


def _token_variants(raw_token: str) -> set[str]:
    output: set[str] = set()
    split_tokens = [item for item in re.split(r"[_.+-]+", raw_token) if item]
    tokens = split_tokens if len(split_tokens) > 1 else [raw_token]
    for token in tokens:
        token = token.strip("._+-")
        if not token:
            continue
        token = _stem(token)
        output.add(token)
    return output


def _stem(token: str) -> str:
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _dedupe(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        output.append(value)
        seen.add(value)
    return output


__all__ = ["preserve_facet_candidates"]
