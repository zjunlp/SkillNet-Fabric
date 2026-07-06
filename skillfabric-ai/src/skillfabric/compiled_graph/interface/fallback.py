"""Deterministic fallback interface inference."""

from __future__ import annotations

import re

from skillfabric.compiled_graph.interface.models import (
    InterfaceEvidence,
    InterfaceField,
    SkillInterface,
)
from skillfabric.registry.models import SkillNode

_TOOL_TERMS = (
    "pdfplumber",
    "pypdf",
    "reportlab",
    "pytest",
    "gh",
    "ffmpeg",
    "python",
    "pandas",
    "numpy",
    "openpyxl",
    "playwright",
)

_ARTIFACT_TERMS = {
    ".pdf": "pdf",
    "pdf": "pdf",
    ".csv": "csv",
    "csv": "csv",
    ".json": "json",
    "json": "json",
    ".md": "markdown",
    "markdown": "markdown",
    ".xlsx": "xlsx",
    "xlsx": "xlsx",
    ".docx": "docx",
    "docx": "docx",
    ".pptx": "pptx",
    "pptx": "pptx",
}


def _fallback_interface(skill: SkillNode, *, model_id: str = "deterministic-interface") -> SkillInterface:
    requires = _complete_fallback_io_fields(skill, role="input")
    produces = _complete_fallback_io_fields(skill, role="output")
    return SkillInterface(
        skill_id=skill.id,
        content_hash=skill.content_hash,
        capability_summary=skill.description,
        when_to_use=skill.description,
        requires=requires,
        produces=produces,
        uses_tools=_tool_fields_from_text(skill),
        evidence=[],
        provenance="deterministic_fallback",
        model_id=model_id,
    )


def _skill_text_for_classification(skill: SkillNode) -> str:
    return " ".join(
        [
            skill.id,
            skill.name,
            skill.description,
            skill.raw_text[:3000],
        ]
    ).lower()


def _has_any(text: str, values: tuple[str, ...]) -> bool:
    return any(value in text for value in values)


def _has_semantic_term(text: str, terms: tuple[str, ...]) -> bool:
    return any(_semantic_term_index(text, term) >= 0 for term in terms)


def _semantic_term_index(text: str, term: str) -> int:
    if not term:
        return -1
    if any(not character.isalnum() and character not in {"_"} for character in term):
        pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
    else:
        pattern = rf"(?<![a-z0-9_-]){re.escape(term)}(?![a-z0-9_-])"
    match = re.search(pattern, text)
    if match is None:
        return -1
    return match.start()


def _complete_fallback_io_fields(skill: SkillNode, *, role: str) -> list[InterfaceField]:
    return _fallback_io_fields(skill, role=role)


def _fallback_io_fields(skill: SkillNode, *, role: str) -> list[InterfaceField]:
    fields: dict[str, list[InterfaceEvidence]] = {}
    for value in _artifact_values_from_text(skill):
        evidence = [
            item
            for item in _artifact_line_evidence(skill, value)
            if _line_matches_io_role(item.text, role=role, value=value)
        ]
        if evidence:
            fields[value] = evidence[:3]
    return [
        InterfaceField(
            name=value,
            kind="artifact",
            confidence=0.6,
            evidence=evidence,
        )
        for value, evidence in sorted(fields.items())
    ]


def _tool_fields_from_text(skill: SkillNode) -> list[InterfaceField]:
    fields: list[InterfaceField] = []
    for value in _tools_from_text(skill):
        fields.append(
            InterfaceField(
                name=value,
                kind="tool",
                confidence=0.55,
                inferred=True,
                evidence=_tool_line_evidence(skill, value),
            )
        )
    return fields


def _tools_from_text(skill: SkillNode) -> list[str]:
    found: set[str] = set()
    for line in skill.raw_text.splitlines():
        lower = line.lower()
        for tool in _TOOL_TERMS:
            if re.search(rf"(?<![a-z0-9_-]){re.escape(tool)}(?![a-z0-9_-])", lower):
                found.add(tool)
    return sorted(found)


def _tool_line_evidence(skill: SkillNode, value: str) -> list[InterfaceEvidence]:
    evidence: list[InterfaceEvidence] = []
    seen: set[tuple[int, str]] = set()
    for line_number, text in enumerate(skill.raw_text.splitlines(), start=1):
        lower = text.lower()
        if re.search(rf"(?<![a-z0-9_-]){re.escape(value)}(?![a-z0-9_-])", lower):
            _append_interface_evidence(evidence, seen, line_number, text, skill.id)
        if len(evidence) >= 2:
            break
    return evidence


def _artifact_values_from_text(skill: SkillNode) -> list[str]:
    values: set[str] = set()
    for line in skill.raw_text.splitlines():
        lower = line.lower()
        for alias, normalized in _ARTIFACT_TERMS.items():
            if alias.startswith("."):
                matched = alias in lower
            else:
                matched = re.search(rf"(?<![a-z0-9_-]){re.escape(alias)}(?![a-z0-9_-])", lower) is not None
            if matched:
                values.add(normalized)
    return sorted(values)


def _line_matches_io_role(text: str, *, role: str, value: str) -> bool:
    line = text.lower()
    closest_hint = _closest_io_hint(line, value)
    if role == "input":
        if ("use after" in line or "depends on" in line) and _line_mentions_artifact(line, value):
            return True
        return closest_hint == "input"
    if closest_hint == "input":
        if not _value_after_any_hint(line, value, ("final",)):
            return False
    if closest_hint == "output":
        if "use after" in line:
            return False
        return True
    if _value_after_any_hint(line, value, ("final",)):
        return True
    return False


def _value_after_any_hint(line: str, value: str, hints: tuple[str, ...]) -> bool:
    value_index = _artifact_value_index(line, value)
    if value_index < 0:
        return False
    return any(0 <= _hint_index(line, hint) < value_index for hint in hints)


def _closest_io_hint(line: str, value: str) -> str:
    value_index = _artifact_value_index(line, value)
    if value_index < 0:
        return "none"
    hints = {
        "input": ("from", "use", "using", "consume", "consumes", "load", "read"),
        "output": (
            "produce",
            "produces",
            "save",
            "saves",
            "write",
            "writes",
            "output",
            "outputs",
            "return",
            "returns",
            "generate",
            "generates",
            "create",
            "creates",
            "draft",
            "drafts",
            "build",
            "builds",
            "export",
            "exports",
            "render",
            "renders",
        ),
    }
    closest_kind = "none"
    closest_index = -1
    for kind, values in hints.items():
        for hint in values:
            hint_index = _hint_index(line, hint)
            if 0 <= hint_index < value_index and hint_index > closest_index:
                closest_kind = kind
                closest_index = hint_index
    return closest_kind


def _hint_index(line: str, hint: str) -> int:
    match = re.search(rf"(?<![a-z0-9]){re.escape(hint)}(?![a-z0-9])", line)
    if match is None:
        return -1
    return match.start()


def _artifact_line_evidence(skill: SkillNode, value: str) -> list[InterfaceEvidence]:
    evidence: list[InterfaceEvidence] = []
    seen: set[tuple[int, str]] = set()
    for line_number, text in enumerate(skill.raw_text.splitlines(), start=1):
        if _line_mentions_artifact(text.lower(), value):
            _append_interface_evidence(evidence, seen, line_number, text, skill.id)
    return evidence


def _append_interface_evidence(
    evidence: list[InterfaceEvidence],
    seen: set[tuple[int, str]],
    line: int,
    text: str,
    skill_id: str,
) -> None:
    key = (line, text)
    if key not in seen:
        evidence.append(InterfaceEvidence(skill=skill_id, line=line, text=text))
        seen.add(key)


def _line_mentions_artifact(line: str, value: str) -> bool:
    return _artifact_value_index(line, value) >= 0


def _artifact_value_index(line: str, value: str) -> int:
    aliases = _artifact_aliases(value)
    positions = [_artifact_alias_index(line, alias) for alias in aliases]
    positions = [position for position in positions if position >= 0]
    return min(positions) if positions else -1


def _artifact_alias_index(line: str, alias: str) -> int:
    if alias.startswith("."):
        return line.find(alias)
    match = re.search(rf"(?<![a-z0-9_-]){re.escape(alias)}(?![a-z0-9_-])", line)
    if match is None:
        return -1
    return match.start()


def _artifact_aliases(value: str) -> tuple[str, ...]:
    normalized = value.lower()
    aliases = {normalized, f".{normalized}"}
    if normalized == "markdown":
        aliases.update({"md", ".md"})
    elif normalized and normalized[-1].isalnum() and not normalized.endswith("s"):
        aliases.add(f"{normalized}s")
    return tuple(sorted(aliases, key=len, reverse=True))
