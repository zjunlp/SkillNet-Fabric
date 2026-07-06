"""Health checks for the Interface Semantics Layer."""

from __future__ import annotations

from dataclasses import dataclass, field

from skillfabric.compiled_graph.interface.models import InterfaceField, SkillInterface


@dataclass(slots=True)
class InterfaceHealthReport:
    """Interface health check result."""

    interface_count: int
    missing_summary: list[str] = field(default_factory=list)
    empty_requires: list[str] = field(default_factory=list)
    empty_produces: list[str] = field(default_factory=list)
    fields_missing_evidence: list[dict[str, str]] = field(default_factory=list)
    low_confidence_fields: list[dict[str, str]] = field(default_factory=list)


def analyze_interface_health(interfaces: list[SkillInterface]) -> InterfaceHealthReport:
    """Analyze interface quality."""

    report = InterfaceHealthReport(interface_count=len(interfaces))
    for interface in interfaces:
        if not interface.capability_summary.strip():
            report.missing_summary.append(interface.skill_id)
        if not interface.requires:
            report.empty_requires.append(interface.skill_id)
        if not interface.produces:
            report.empty_produces.append(interface.skill_id)
        for field_name, fields in _iter_field_groups(interface):
            for interface_field in fields:
                if not interface_field.evidence:
                    report.fields_missing_evidence.append(_field_ref(interface, field_name, interface_field))
                if interface_field.confidence and interface_field.confidence < 0.5:
                    report.low_confidence_fields.append(_field_ref(interface, field_name, interface_field))
    return report


def render_interface_health_report(report: InterfaceHealthReport) -> str:
    """Render interface_health_report.md."""

    lines = [
        "# Interface Health Report",
        "",
        f"- interfaces: {report.interface_count}",
        f"- missing summary: {len(report.missing_summary)}",
        f"- empty requires: {len(report.empty_requires)}",
        f"- empty produces: {len(report.empty_produces)}",
        f"- fields missing evidence: {len(report.fields_missing_evidence)}",
        f"- low confidence fields: {len(report.low_confidence_fields)}",
        "",
    ]
    _append_list(lines, "Missing Summary", report.missing_summary)
    _append_list(lines, "Empty Requires", report.empty_requires)
    _append_list(lines, "Empty Produces", report.empty_produces)
    _append_field_refs(lines, "Fields Missing Evidence", report.fields_missing_evidence)
    _append_field_refs(lines, "Low Confidence Fields", report.low_confidence_fields)
    return "\n".join(lines).rstrip() + "\n"


def _iter_field_groups(interface: SkillInterface):
    yield "requires", interface.requires
    yield "produces", interface.produces
    yield "uses_tools", interface.uses_tools


def _field_ref(interface: SkillInterface, field_name: str, field: InterfaceField) -> dict[str, str]:
    return {
        "skill_id": interface.skill_id,
        "field_group": field_name,
        "name": field.name,
    }


def _append_list(lines: list[str], title: str, values: list[str]) -> None:
    lines.extend([f"## {title}", ""])
    if not values:
        lines.extend(["None.", ""])
        return
    for value in values:
        lines.append(f"- {value}")
    lines.append("")


def _append_field_refs(lines: list[str], title: str, values: list[dict[str, str]]) -> None:
    lines.extend([f"## {title}", ""])
    if not values:
        lines.extend(["None.", ""])
        return
    for value in values:
        lines.append(f"- {value['skill_id']} {value['field_group']}:{value['name']}")
    lines.append("")
