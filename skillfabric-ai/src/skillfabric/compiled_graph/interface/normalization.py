"""Schema normalization for extracted skill interfaces."""

from __future__ import annotations

from typing import Any

from skillfabric.compiled_graph.interface.models import (
    INTERFACE_FIELD_KINDS,
    InterfaceEvidence,
    InterfaceField,
    SkillInterface,
    normalize_interface_kind,
)
from skillfabric.compiled_graph.interface.providers import InterfaceSchemaError
from skillfabric.registry.models import SkillNode


def _interface_from_raw(skill: SkillNode, raw: dict[str, Any], *, model_id: str, provenance: str) -> SkillInterface:
    return SkillInterface(
        skill_id=skill.id,
        content_hash=skill.content_hash,
        capability_summary=_string_value(raw.get("capability_summary") or skill.description),
        when_to_use=_string_value(raw.get("when_to_use") or ""),
        requires=_fields_from_raw(skill, raw.get("requires", []), default_kind="data", allow_tool=False),
        produces=_fields_from_raw(skill, raw.get("produces", []), default_kind="data", allow_tool=False),
        uses_tools=_fields_from_raw(skill, raw.get("uses_tools", []), default_kind="tool", only_tool=True),
        evidence=_evidence_from_raw(skill, raw.get("evidence", [])),
        provenance=provenance,
        model_id=model_id,
    )


def _normalize_recoverable_interface_payload(raw: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(raw)
    for key in ("capability_summary", "when_to_use"):
        if key in normalized:
            normalized[key] = _string_value(normalized[key])
    for key in ("requires", "produces", "uses_tools"):
        payload = normalized.get(key)
        if not isinstance(payload, list):
            continue
        items: list[Any] = []
        for item in payload:
            if not isinstance(item, dict):
                items.append(item)
                continue
            copy = dict(item)
            copy["name"] = _string_value(copy.get("name", ""))
            copy["description"] = _string_value(copy.get("description", ""))
            copy["kind"] = _normalize_kind(_string_value(copy.get("kind", "")))
            copy["evidence"] = _recover_evidence_items(copy.get("evidence", []))
            items.append(copy)
        normalized[key] = items
    return normalized


def _string_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "; ".join(_string_value(item) for item in value if _string_value(item)).strip()
    if isinstance(value, dict):
        preferred = value.get("text") or value.get("summary") or value.get("description") or value.get("value")
        if preferred is not None:
            return _string_value(preferred)
        return "; ".join(f"{key}: {_string_value(item)}" for key, item in value.items() if _string_value(item)).strip()
    if value is None:
        return ""
    return str(value).strip()


def _recover_evidence_items(payload: Any) -> list[Any]:
    if not isinstance(payload, list):
        return []
    recovered: list[Any] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            int(item.get("line", 0))
        except (TypeError, ValueError):
            continue
        recovered.append(item)
    return recovered


def _validate_interface_payload(raw: dict[str, Any]) -> None:
    structural_fields = {
        "requires",
        "produces",
        "uses_tools",
    }
    list_fields = {
        *structural_fields,
        "evidence",
    }
    if "capability_summary" not in raw and not any(key in raw for key in structural_fields):
        raise InterfaceSchemaError("interface JSON does not contain SkillInterface fields")
    if "capability_summary" in raw and not isinstance(raw["capability_summary"], str):
        raise InterfaceSchemaError("capability_summary must be a string")
    if "when_to_use" in raw and not isinstance(raw["when_to_use"], str):
        raise InterfaceSchemaError("when_to_use must be a string")
    for key in list_fields:
        if key not in raw:
            continue
        if not isinstance(raw[key], list):
            raise InterfaceSchemaError(f"{key} must be a list")
        if any(not isinstance(item, dict) for item in raw[key]):
            raise InterfaceSchemaError(f"{key} items must be objects")
        if key == "evidence":
            _validate_evidence_items(key, raw[key])
        else:
            for item in raw[key]:
                _validate_field_kind(key, item.get("kind", ""))
                _validate_evidence_items(key, item.get("evidence", []))
    for key in list_fields - {"evidence"}:
        for item in raw.get(key, []):
            evidence = item.get("evidence", [])
            if evidence is not None and not isinstance(evidence, list):
                raise InterfaceSchemaError(f"{key}.evidence must be a list")
            if "confidence" in item and not _is_float(item.get("confidence")):
                raise InterfaceSchemaError(f"{key}.confidence must be numeric")


def _validate_evidence_items(field_name: str, evidence: Any) -> None:
    if evidence is None:
        return
    if not isinstance(evidence, list):
        raise InterfaceSchemaError(f"{field_name}.evidence must be a list")
    for item in evidence:
        if not isinstance(item, dict):
            raise InterfaceSchemaError(f"{field_name}.evidence items must be objects")
        if not _is_int(item.get("line", 0)):
            raise InterfaceSchemaError(f"{field_name}.evidence line must be numeric")


def _is_float(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _is_int(value: Any) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def _validate_field_kind(field_name: str, value: Any) -> None:
    kind = _normalize_kind(_string_value(value))
    if kind not in INTERFACE_FIELD_KINDS:
        raise InterfaceSchemaError(f"{field_name}.kind must be one of {sorted(INTERFACE_FIELD_KINDS)}")


def _fields_from_raw(
    skill: SkillNode,
    payload: Any,
    *,
    default_kind: str,
    allow_tool: bool = True,
    only_tool: bool = False,
) -> list[InterfaceField]:
    if not isinstance(payload, list):
        return []
    fields: list[InterfaceField] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        evidence = _evidence_from_raw(skill, item.get("evidence", []))
        kind = _normalize_kind(str(item.get("kind", default_kind) or default_kind))
        if only_tool and kind != "tool":
            kind = "tool"
        if not allow_tool and kind == "tool":
            continue
        fields.append(
            InterfaceField(
                name=name,
                description=str(item.get("description", "")),
                kind=kind,
                confidence=float(item.get("confidence", 0.0) or 0.0),
                evidence=evidence,
            )
        )
    return fields


def _normalize_kind(value: str) -> str:
    return normalize_interface_kind(value)


def _evidence_from_raw(skill: SkillNode, payload: Any) -> list[InterfaceEvidence]:
    if not isinstance(payload, list):
        return []
    evidence: list[InterfaceEvidence] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        evidence.append(
            InterfaceEvidence(
                skill=str(item.get("skill", skill.id)),
                line=int(item.get("line", 0)),
                text=str(item.get("text", "")),
            )
        )
    return evidence


def _error_payload(error_type: str, reason: str) -> dict[str, Any]:
    return {
        "error_type": error_type,
        "reason": reason,
    }


def _rejection_reason(payload: dict[str, Any]) -> str:
    return f"{payload.get('error_type', 'error')}: {payload.get('reason', '')}"
