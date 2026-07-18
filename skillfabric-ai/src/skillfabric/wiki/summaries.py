"""Deterministic Wiki summaries derived from validated SkillContracts."""

from __future__ import annotations

from skillfabric.wiki.models import NO_WORKFLOW_GUIDANCE, WikiSummaryRecord


def summary_from_payload(
    *,
    page_type: str,
    entity_id: str,
    content_hash: str,
    payload: dict[str, object],
) -> WikiSummaryRecord:
    """Project validated contract fields into renderer-facing summary fields."""

    name = _optional_string(payload.get("name")) or entity_id
    description = _optional_string(payload.get("description"))
    capability = _optional_string(payload.get("capability")) or description or name
    routing_summary = _optional_string(payload.get("when_to_use")) or capability
    requires = _string_list(payload.get("requires"), label="requires")
    produces = _string_list(payload.get("produces"), label="produces")

    if requires and produces:
        workflow_summary = f"Requires {', '.join(requires)}; produces {', '.join(produces)}."
    elif requires:
        workflow_summary = f"Requires {', '.join(requires)} before execution."
    elif produces:
        workflow_summary = f"Produces {', '.join(produces)} for downstream use."
    else:
        workflow_summary = NO_WORKFLOW_GUIDANCE

    return WikiSummaryRecord(
        page_type=page_type,
        entity_id=entity_id,
        content_hash=content_hash,
        routing_summary=routing_summary,
        workflow_summary=workflow_summary,
        summary=capability,
    )


def _optional_string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _string_list(value: object, *, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")

    result = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{label}[{index}] must be a non-empty string")
        result.append(item.strip())
    return result


__all__ = ["summary_from_payload"]
