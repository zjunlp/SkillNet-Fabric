"""Extract explicit inter-skill mentions from skill documents."""

from __future__ import annotations

import re
from dataclasses import dataclass

from skillfabric.compiled_graph.relations.models import SkillMention
from skillfabric.registry.models import SkillNode


@dataclass(slots=True)
class _MentionPattern:
    skill_id: str
    mention_type: str
    pattern: re.Pattern[str]


def extract_skill_mentions(skills: list[SkillNode]) -> list[SkillMention]:
    """Return explicit mentions of known skills inside full SKILL.md text."""

    patterns = _build_patterns(skills)
    mentions: list[SkillMention] = []
    seen: set[tuple[str, str, int, str]] = set()
    for skill in skills:
        for line_number, line in enumerate(skill.raw_text.splitlines(), start=1):
            for pattern in patterns:
                if pattern.skill_id == skill.id:
                    continue
                if not pattern.pattern.search(line):
                    continue
                key = (skill.id, pattern.skill_id, line_number, pattern.mention_type)
                if key in seen:
                    continue
                seen.add(key)
                mentions.append(
                    SkillMention(
                        from_skill=skill.id,
                        to_skill=pattern.skill_id,
                        line=line_number,
                        text=line.strip(),
                        mention_type=pattern.mention_type,
                        direction_hint=_direction_hint(line),
                    )
                )
    return mentions


def _build_patterns(skills: list[SkillNode]) -> list[_MentionPattern]:
    patterns: list[_MentionPattern] = []
    seen: set[tuple[str, str, str]] = set()
    for skill in skills:
        targets = [
            ("id", skill.id),
            ("name", skill.name),
            ("alias", skill.name.replace("-", " ")),
        ]
        stripped = skill.id.removeprefix("skill:")
        if stripped != skill.id:
            targets.append(("id", stripped))
        targets.append(("wikilink", skill.name))
        for mention_type, value in targets:
            if not value:
                continue
            key = (skill.id, mention_type, value.lower())
            if key in seen:
                continue
            seen.add(key)
            if mention_type == "wikilink":
                pattern = re.compile(r"\[\[\s*" + re.escape(value) + r"\s*\]\]", re.IGNORECASE)
            else:
                pattern = re.compile(_term_pattern(value), re.IGNORECASE)
            patterns.append(_MentionPattern(skill.id, mention_type, pattern))
    patterns.sort(key=lambda item: (item.skill_id, item.mention_type))
    return patterns


def _term_pattern(value: str) -> str:
    escaped = re.escape(value)
    return rf"(?<![a-z0-9_.:+-]){escaped}(?![a-z0-9_.:+-])"


def _direction_hint(line: str) -> str:
    lower = line.lower()
    if any(phrase in lower for phrase in ("compose with", "composes with", "works with", "combine with")):
        return "undirected"
    if any(phrase in lower for phrase in ("use after", "after ", "requires", "depends on", "depend on")):
        return "A->B"
    return "none"
