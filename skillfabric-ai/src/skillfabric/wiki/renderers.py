"""Markdown renderers for generated wiki pages."""

from __future__ import annotations

import hashlib
from collections import Counter

from skillfabric.compiled_graph.interface.models import InterfaceField, SkillInterface
from skillfabric.compiled_graph.models import Edge
from skillfabric.registry.models import SkillNode
from skillfabric.storage import Workspace
from skillfabric.wiki.indexer import page_path
from skillfabric.wiki.loader import WikiSource
from skillfabric.wiki.models import WikiBuildConfig, WikiPage, WikiSummaryRecord
from skillfabric.wiki.pages import bullet_list, frontmatter, wiki_link


def _skill_page(
    source: WikiSource,
    skill: SkillNode,
    config: WikiBuildConfig,
    summaries: dict[tuple[str, str], WikiSummaryRecord],
    workspace: Workspace,
) -> WikiPage:
    interface = source.interfaces.get(skill.id)
    summary = summaries[("skill", skill.id)]
    core_links = source.skill_core_links(skill.id)
    execution_links = source.skill_execution_links(skill.id)
    source_slug = f"source/{page_path(workspace.wiki_dir, 'skills', skill.id).name}"
    tags = _skill_tags(interface)
    text = "\n\n".join(
        [
            frontmatter(
                {
                    "type": "Skill",
                    "title": skill.name,
                    "description": _one_line(summary.routing_summary or summary.summary or skill.description),
                    "skill_id": skill.id,
                    "selectable": True,
                    "source": source_slug,
                    "tags": tags,
                }
            ),
            "# Skill Card",
            f"Skill: {skill.name}",
            "## Purpose\n\n" + _one_line(summary.routing_summary or summary.summary or skill.description),
            "## Use When\n\n" + _render_use_when(interface, summary),
            "## Do Not Use When\n\n" + _render_do_not_use(interface),
            "## Inputs\n\n" + _render_field_bullets(interface.requires if interface else []),
            "## Outputs\n\n" + _render_field_bullets(interface.produces if interface else []),
            "## Tools And Dependencies\n\n" + _render_field_bullets(interface.uses_tools if interface else []),
            "## Composition Notes\n\n"
            + _render_composition_notes(
                source,
                core_links,
                execution_links.get("workflow_hints", []),
                current_skill_id=skill.id,
                limit=config.max_neighbors_per_section,
            ),
            "## Failure Modes\n\n" + _render_failure_modes(interface),
            "## Read Full Source\n\n"
            + f"Open [full SKILL.md]({source_slug}) when the card is insufficient to decide routing boundaries or execution requirements.",
        ]
    ) + "\n"
    return WikiPage(
        path=page_path(workspace.wiki_dir, "skills", skill.id),
        page_type="skill",
        entity_id=skill.id,
        title=skill.name,
        text=text,
    )


def _community_page(
    source: WikiSource,
    community_id: str,
    config: WikiBuildConfig,
    summaries: dict[tuple[str, str], WikiSummaryRecord],
    workspace: Workspace,
) -> WikiPage:
    community = source.communities[community_id]
    members = source.community_members.get(community_id, [])
    common = _common_interface_terms(source, members)
    summary = summaries[("community", community_id)]
    important_edges = [
        edge
        for edge in source.core_edges
        if edge.type in {"compose_with", "depend_on"}
        and edge.source in members
        and edge.target in members
    ]
    text = "\n\n".join(
        [
            frontmatter(
                {
                    "type": "Community",
                    "title": community.name,
                    "description": _one_line(summary.summary or community.summary),
                    "community_id": community_id,
                    "tags": ["community"],
                }
            ),
            f"# {community.name}",
            "## Capability Cluster Summary\n\n" + (summary.summary or community.summary),
            "## Representative Skills\n\n" + bullet_list([wiki_link("skills", item, source.skills[item].name) for item in community.representative_skill_ids if item in source.skills]),
            "## Member Skills\n\n" + bullet_list([wiki_link("skills", item, source.skills[item].name) for item in members if item in source.skills]),
            "## Common Task Patterns\n\n" + bullet_list(community.task_patterns or [summary.workflow_summary or summary.summary or community.summary]),
            "## Common Contract Terms\n\n" + bullet_list([f"{key}: {', '.join(values)}" for key, values in common.items() if values]),
            "## Important Skill Relations\n\n" + _render_core_links(source, important_edges, current_skill_id="", limit=config.max_neighbors_per_section),
        ]
    ) + "\n"
    return WikiPage(page_path(workspace.wiki_dir, "communities", community_id), "community", community_id, community.name, text)


def _workflow_page(
    source: WikiSource,
    record,
    workspace: Workspace,
) -> WikiPage:
    source_skill = source.skills.get(record.source_skill)
    target_skill = source.skills.get(record.target_skill)
    source_label = source_skill.name if source_skill else record.source_skill
    target_label = target_skill.name if target_skill else record.target_skill
    entity_id = f"{record.source_skill}__{record.target_skill}__{record.relation_type}"
    title = f"{source_label} -> {target_label}"
    text = "\n\n".join(
        [
            frontmatter(
                {
                    "type": "Workflow",
                    "title": title,
                    "description": f"{record.relation_type} via {record.canonical_object}.",
                    "workflow_id": entity_id,
                    "tags": ["workflow"],
                }
            ),
            f"# {title}",
            "## Summary\n\n" + f"{record.relation_type} via `{record.canonical_object}`.",
            "## Skills\n\n" + bullet_list(
                [
                    wiki_link("skills", record.source_skill, source_label),
                    wiki_link("skills", record.target_skill, target_label),
                ]
            ),
            "## Ordering Hint\n\n" + bullet_list([f"{source_label} -> {target_label}"]),
        ]
    ) + "\n"
    return WikiPage(page_path(workspace.wiki_dir, "workflows", entity_id), "workflow", entity_id, title, text)


def _skill_source_page(skill: SkillNode, workspace: Workspace) -> WikiPage:
    """Render the full original SKILL.md as the authoritative skill source."""

    source_filename = page_path(workspace.wiki_dir, "skills", skill.id).name
    text = "\n\n".join(
        [
            frontmatter(
                {
                    "type": "Skill Source",
                    "title": f"{skill.name} Source",
                    "description": f"Full original SKILL.md for {skill.id}.",
                    "skill_id": skill.id,
                    "resource": f"skill://{skill.id.removeprefix('skill:')}/SKILL.md",
                }
            ),
            "# Full SKILL.md",
            skill.raw_text.rstrip() or skill.description,
        ]
    ) + "\n"
    return WikiPage(
        path=workspace.wiki_skill_sources_dir / source_filename,
        page_type="skill",
        entity_id=skill.id,
        title=f"{skill.name} Source",
        text=text,
    )


def _debug_pages(source: WikiSource, workspace: Workspace) -> list[WikiPage]:
    pages: list[WikiPage] = []
    for artifact_id, artifact in sorted(source.raw_artifacts.items(), key=lambda item: item[1].name):
        text = "\n\n".join(
            [
                frontmatter({"type": "debug", "debug_type": "raw_artifact", "artifact_id": artifact_id}),
                f"# {artifact.name}",
                "## Evidence\n\n" + bullet_list([f"{item.skill}:{item.line} - {item.text}" for item in artifact.evidence[:8]]),
            ]
        ) + "\n"
        pages.append(WikiPage(page_path(workspace.wiki_debug_dir, "raw_artifacts", artifact_id), "debug", artifact_id, artifact.name, text))
    for scenario_id, scenario in sorted(source.raw_scenarios.items(), key=lambda item: item[1].name):
        text = "\n\n".join(
            [
                frontmatter({"type": "debug", "debug_type": "raw_scenario", "scenario_id": scenario_id}),
                f"# {scenario.name}",
                "## Evidence\n\n" + bullet_list([f"{item.skill}:{item.line} - {item.text}" for item in scenario.evidence[:8]]),
            ]
        ) + "\n"
        pages.append(WikiPage(page_path(workspace.wiki_debug_dir, "raw_scenarios", scenario_id), "debug", scenario_id, scenario.name, text))
    report = "\n\n".join(
        [
            "# Extraction Report",
            "## Counts",
            bullet_list(
                [
                    f"raw_artifacts: {len(source.raw_artifacts)}",
                    f"raw_scenarios: {len(source.raw_scenarios)}",
                    f"execution_index: {len(source.execution_index)}",
                ]
            ),
        ]
    ) + "\n"
    pages.append(WikiPage(workspace.wiki_debug_dir / "extraction_report.md", "debug", "extraction_report", "Extraction Report", report))
    return pages


def _render_failure_modes(interface: SkillInterface | None) -> str:
    if interface is None:
        return "- None"
    return bullet_list(_field_names(interface.failure_modes))


def _render_use_when(interface: SkillInterface | None, summary: WikiSummaryRecord) -> str:
    values = []
    if interface is not None and interface.when_to_use:
        values.append(interface.when_to_use)
    values.append(summary.workflow_summary or summary.routing_summary or summary.summary)
    return bullet_list([_one_line(value) for value in values if value])


def _render_do_not_use(interface: SkillInterface | None) -> str:
    if interface is None:
        return "- When the task does not match this skill's inputs, outputs, or declared tools."
    values = [
        field.description or field.name
        for field in interface.failure_modes
        if field.name or field.description
    ]
    if values:
        return bullet_list([_one_line(value) for value in values])
    return "- When the task requires unrelated inputs, outputs, tools, or execution responsibilities."


def _render_field_bullets(fields: list[InterfaceField]) -> str:
    values = []
    for field in fields:
        if field.name and field.description:
            values.append(f"{field.name}: {_one_line(field.description)}")
        elif field.name:
            values.append(field.name)
    return bullet_list(values)


def _render_composition_notes(
    source: WikiSource,
    core_links: list[Edge],
    workflow_hints: list,
    *,
    current_skill_id: str,
    limit: int,
) -> str:
    values: list[str] = []
    relation_text = _render_core_links(
        source,
        [edge for edge in core_links if edge.type in {"compose_with", "depend_on", "member_of"}],
        current_skill_id=current_skill_id,
        limit=limit,
    )
    if relation_text != "- None":
        values.extend(relation_text.splitlines())
    workflow_text = _render_workflow_hints(source, workflow_hints)
    if workflow_text != "- None":
        values.extend(workflow_text.splitlines())
    return "\n".join(dict.fromkeys(values)) if values else "- None"


def _render_workflow_hints(source: WikiSource, records: list) -> str:
    items = []
    for record in sorted(records, key=lambda item: (item.source_skill, item.target_skill, item.relation_type)):
        source_label = source.skills[record.source_skill].name if record.source_skill in source.skills else record.source_skill
        target_label = source.skills[record.target_skill].name if record.target_skill in source.skills else record.target_skill
        if record.source_skill in source.skills and record.target_skill in source.skills:
            items.append(
                f"{wiki_link('skills', record.source_skill, source_label)} -> "
                f"{wiki_link('skills', record.target_skill, target_label)} "
                f"({record.relation_type}: `{record.canonical_object}`)"
            )
    return bullet_list(items)


def _render_core_links(source: WikiSource, edges: list[Edge], *, current_skill_id: str, limit: int) -> str:
    items: list[str] = []
    seen: set[tuple[str, str]] = set()
    for edge in sorted(edges, key=lambda item: (item.type, -item.confidence, -item.weight, item.source, item.target)):
        if len(items) >= limit:
            break
        if edge.type == "member_of":
            if edge.target in source.communities:
                key = ("member_of", edge.target)
                if key not in seen:
                    items.append(f"member_of: {wiki_link('communities', edge.target, source.communities[edge.target].name)}")
                    seen.add(key)
            continue
        if current_skill_id:
            label, other = _directed_link_label(edge, current_skill_id)
        else:
            label = edge.type
            other = edge.target
            if edge.source in source.skills and edge.target in source.skills:
                key = (label, f"{edge.source}->{edge.target}")
                if key not in seen:
                    items.append(
                        _format_core_link(
                            label,
                            edge,
                            f"{wiki_link('skills', edge.source, source.skills[edge.source].name)} "
                            f"-> {wiki_link('skills', edge.target, source.skills[edge.target].name)}",
                        )
                    )
                    seen.add(key)
                continue
        if other in source.skills:
            key = (label, other)
            if key not in seen:
                items.append(_format_core_link(label, edge, wiki_link("skills", other, source.skills[other].name)))
                seen.add(key)
    return bullet_list(items)


def _format_core_link(label: str, edge: Edge, target: str) -> str:
    suffix = f" confidence={edge.confidence:.2f}" if edge.type == "depend_on" else ""
    return f"{label}: {target}{suffix}"


def _skill_summary_payload(skill: SkillNode, interface: SkillInterface | None) -> dict[str, object]:
    return {
        "name": skill.name,
        "description": skill.description,
        "when_to_use": interface.when_to_use if interface else "",
        "requires": _field_names(interface.requires) if interface else [],
        "produces": _field_names(interface.produces) if interface else [],
        "uses_tools": _field_names(interface.uses_tools) if interface else [],
    }


def _skill_tags(interface: SkillInterface | None) -> list[str]:
    tags = ["skill"]
    if interface is not None:
        tags.extend(name for name in _field_names(interface.produces)[:4] if name)
        tags.extend(name for name in _field_names(interface.uses_tools)[:4] if name)
    return list(dict.fromkeys(_safe_tag(item) for item in tags if item))


def _safe_tag(value: str) -> str:
    return value.strip().lower().replace(" ", "-")


def _one_line(value: str) -> str:
    return " ".join(str(value).split())


def _field_names(fields: list[InterfaceField]) -> list[str]:
    return [field.name for field in fields if field.name]


def _directed_link_label(edge: Edge, current_skill_id: str) -> tuple[str, str]:
    if edge.type == "depend_on":
        if edge.source == current_skill_id:
            return "depends_on", edge.target
        return "required_by", edge.source
    if edge.type == "compose_with":
        other = edge.target if edge.source == current_skill_id else edge.source
        return "compose_with", other
    other = edge.target if edge.source == current_skill_id else edge.source
    return edge.type, other


def _common_interface_terms(source: WikiSource, members: list[str]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for field_group in ("requires", "produces", "uses_tools"):
        counter: Counter[str] = Counter()
        for skill_id in members:
            interface = source.interfaces.get(skill_id)
            if interface is None:
                continue
            counter.update(_field_names(list(getattr(interface, field_group))))
        output[field_group] = [name for name, _count in counter.most_common(8)]
    return output


def _content_hash(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _first_paragraph(text: str) -> str:
    skipped_keys = (
        "type:",
        "skill_id:",
        "community_id:",
        "artifact_id:",
        "scenario_id:",
        "name:",
        "content_hash:",
        "tags:",
    )
    for line in text.splitlines():
        if line and not line.startswith("---") and not line.startswith("#") and not line.startswith(skipped_keys):
            return line.strip()
    return ""
