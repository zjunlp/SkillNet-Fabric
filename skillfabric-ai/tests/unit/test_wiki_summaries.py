from __future__ import annotations

import pytest

from skillfabric.wiki.models import NO_WORKFLOW_GUIDANCE
from skillfabric.wiki.summaries import summary_from_payload


def test_summary_projects_validated_contract_fields() -> None:
    record = summary_from_payload(
        page_type="skill",
        entity_id="skill:tables",
        content_hash="hash-tables",
        payload={
            "name": "tables",
            "description": "Fallback description.",
            "capability": "Parse normalized tables.",
            "when_to_use": "Use for tabular documents.",
            "requires": ["document", "schema"],
            "produces": ["normalized table", "validation report"],
        },
    )

    assert record.summary == "Parse normalized tables."
    assert record.routing_summary == "Use for tabular documents."
    assert record.workflow_summary == (
        "Requires document, schema; produces normalized table, validation report."
    )
    assert record.page_type == "skill"
    assert record.entity_id == "skill:tables"
    assert record.content_hash == "hash-tables"


@pytest.mark.parametrize(
    ("payload", "expected_summary", "expected_routing", "expected_workflow"),
    [
        (
            {"name": "fallback", "description": "Fallback description."},
            "Fallback description.",
            "Fallback description.",
            NO_WORKFLOW_GUIDANCE,
        ),
        (
            {"name": "fallback", "requires": ["input"]},
            "fallback",
            "fallback",
            "Requires input before execution.",
        ),
        (
            {"name": "fallback", "produces": ["output"]},
            "fallback",
            "fallback",
            "Produces output for downstream use.",
        ),
    ],
)
def test_summary_uses_deterministic_fallbacks(
    payload: dict[str, object],
    expected_summary: str,
    expected_routing: str,
    expected_workflow: str,
) -> None:
    record = summary_from_payload(
        page_type="skill",
        entity_id="skill:fallback",
        content_hash="hash-fallback",
        payload=payload,
    )

    assert record.summary == expected_summary
    assert record.routing_summary == expected_routing
    assert record.workflow_summary == expected_workflow


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("requires", "document"),
        ("requires", ["document", 1]),
        ("produces", {"artifact": "report"}),
        ("produces", [""]),
    ],
)
def test_summary_rejects_invalid_workflow_fields(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        summary_from_payload(
            page_type="skill",
            entity_id="skill:invalid",
            content_hash="hash-invalid",
            payload={"name": "invalid", field: value},
        )
