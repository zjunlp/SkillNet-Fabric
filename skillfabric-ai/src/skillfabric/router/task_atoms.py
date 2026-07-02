"""Task decomposition atoms used as graph-grounded route query views."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TASK_ATOMS_SCHEMA_VERSION = "1.0"
TASK_ATOM_KINDS = {"action", "artifact", "constraint"}
_SKILL_ID_RE = re.compile(r"(?<![A-Za-z0-9_:-])skill:[A-Za-z0-9][A-Za-z0-9_.:-]*")
_FORBIDDEN_KEYS = {"skill_id", "skill_ids", "intent", "domain_hints", "deliverable"}
_TASK_ATOMS_KEYS = {"schema_version", "atoms"}
_TASK_ATOM_KEYS = {"id", "kind", "text", "evidence", "required", "depends_on"}


@dataclass(slots=True)
class TaskAtom:
    """One atomic task requirement extracted from the original user query."""

    id: str
    kind: str
    text: str
    evidence: str
    required: bool = True
    depends_on: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "text": self.text,
            "evidence": self.evidence,
            "required": bool(self.required),
            "depends_on": list(self.depends_on),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TaskAtom:
        raw_depends_on = payload.get("depends_on", [])
        depends_on = (
            [str(item).strip() for item in raw_depends_on if str(item).strip()]
            if isinstance(raw_depends_on, list)
            else []
        )
        return cls(
            id=str(payload.get("id", "")).strip(),
            kind=str(payload.get("kind", "")).strip(),
            text=str(payload.get("text", "")).strip(),
            evidence=str(payload.get("evidence", "")).strip(),
            required=bool(payload.get("required", True)),
            depends_on=depends_on,
        )


@dataclass(slots=True)
class TaskDecomposition:
    """Route-time decomposition of a user query into atomic requirements."""

    schema_version: str = TASK_ATOMS_SCHEMA_VERSION
    atoms: list[TaskAtom] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "atoms": [atom.to_dict() for atom in self.atoms],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TaskDecomposition:
        return cls(
            schema_version=str(payload.get("schema_version", TASK_ATOMS_SCHEMA_VERSION)),
            atoms=[
                TaskAtom.from_dict(item)
                for item in payload.get("atoms", [])
                if isinstance(item, dict)
            ],
        )

    @classmethod
    def empty(cls) -> TaskDecomposition:
        return cls(schema_version=TASK_ATOMS_SCHEMA_VERSION, atoms=[])


def task_atoms_json_schema() -> dict[str, Any]:
    """Return the strict JSON schema for LLM task atomization."""

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "atoms"],
        "properties": {
            "schema_version": {"type": "string", "const": TASK_ATOMS_SCHEMA_VERSION},
            "atoms": {
                "type": "array",
                "maxItems": 12,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "kind", "text", "evidence", "required", "depends_on"],
                    "properties": {
                        "id": {"type": "string"},
                        "kind": {"type": "string", "enum": sorted(TASK_ATOM_KINDS)},
                        "text": {"type": "string"},
                        "evidence": {"type": "string"},
                        "required": {"type": "boolean"},
                        "depends_on": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
        },
    }


def validate_task_decomposition(payload: TaskDecomposition | dict[str, Any], *, query: str = "") -> TaskDecomposition:
    """Validate and normalize a task decomposition payload."""

    if isinstance(payload, dict):
        _validate_raw_payload_shape(payload)
    decomposition = payload if isinstance(payload, TaskDecomposition) else TaskDecomposition.from_dict(payload)
    if decomposition.schema_version != TASK_ATOMS_SCHEMA_VERSION:
        raise ValueError(f"unsupported task atoms schema_version: {decomposition.schema_version!r}")
    if len(decomposition.atoms) > 12:
        raise ValueError("task atoms payload contains more than 12 atoms")
    seen_ids: set[str] = set()
    normalized: list[TaskAtom] = []
    for index, atom in enumerate(decomposition.atoms, start=1):
        _validate_no_forbidden_payload_keys(atom.to_dict())
        if not atom.id:
            raise ValueError("task atom id is required")
        if atom.id in seen_ids:
            raise ValueError(f"duplicate task atom id: {atom.id}")
        if atom.kind not in TASK_ATOM_KINDS:
            raise ValueError(f"unknown task atom kind for {atom.id}: {atom.kind}")
        if not atom.text:
            raise ValueError(f"task atom text is required: {atom.id}")
        if not atom.evidence:
            raise ValueError(f"task atom evidence is required: {atom.id}")
        if _SKILL_ID_RE.search(atom.text) or _SKILL_ID_RE.search(atom.evidence):
            raise ValueError(f"task atom must not mention skill ids: {atom.id}")
        if query and atom.evidence.lower() not in query.lower():
            raise ValueError(f"task atom evidence is not quoted from query: {atom.id}")
        seen_ids.add(atom.id)
        normalized.append(
            TaskAtom(
                id=atom.id or f"a{index}",
                kind=atom.kind,
                text=atom.text,
                evidence=atom.evidence,
                required=bool(atom.required),
                depends_on=list(atom.depends_on),
            )
        )
    for atom in normalized:
        for dependency in atom.depends_on:
            if dependency not in seen_ids:
                raise ValueError(f"task atom {atom.id} depends on unknown atom id: {dependency}")
            if dependency == atom.id:
                raise ValueError(f"task atom {atom.id} cannot depend on itself")
    return TaskDecomposition(schema_version=TASK_ATOMS_SCHEMA_VERSION, atoms=normalized)


def load_task_decomposition(path: str | Path, *, query: str = "") -> TaskDecomposition:
    """Load and validate task atoms from a JSON file."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("task atoms file must contain a JSON object")
    return validate_task_decomposition(payload, query=query)


def _validate_no_forbidden_payload_keys(payload: dict[str, Any]) -> None:
    forbidden = sorted(_FORBIDDEN_KEYS & set(payload))
    if forbidden:
        raise ValueError(f"task atom contains forbidden field(s): {', '.join(forbidden)}")


def _validate_raw_payload_shape(payload: dict[str, Any]) -> None:
    extra_top_level = sorted(set(payload) - _TASK_ATOMS_KEYS)
    if extra_top_level:
        raise ValueError(f"task atoms payload contains unknown field(s): {', '.join(extra_top_level)}")
    missing_top_level = sorted(_TASK_ATOMS_KEYS - set(payload))
    if missing_top_level:
        raise ValueError(f"task atoms payload missing required field(s): {', '.join(missing_top_level)}")
    atoms = payload.get("atoms")
    if not isinstance(atoms, list):
        raise ValueError("task atoms payload must contain an atoms list")
    if len(atoms) > 12:
        raise ValueError("task atoms payload contains more than 12 atoms")
    for index, raw_atom in enumerate(atoms, start=1):
        if not isinstance(raw_atom, dict):
            raise ValueError(f"task atom #{index} must be a JSON object")
        _validate_no_forbidden_payload_keys(raw_atom)
        missing_atom_fields = sorted(_TASK_ATOM_KEYS - set(raw_atom))
        if missing_atom_fields:
            raise ValueError(
                f"task atom {raw_atom.get('id', index)!r} missing required field(s): "
                f"{', '.join(missing_atom_fields)}"
            )
        extra_atom_fields = sorted(set(raw_atom) - _TASK_ATOM_KEYS)
        if extra_atom_fields:
            raise ValueError(
                f"task atom {raw_atom.get('id', index)!r} contains unknown field(s): "
                f"{', '.join(extra_atom_fields)}"
            )
