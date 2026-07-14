"""Markdown renderers for generated wiki pages."""

from __future__ import annotations

from skillfabric.compiled_graph.contracts.models import ContractField, SkillContract
from skillfabric.compiled_graph.models import Edge
from skillfabric.registry.models import SkillNode
from skillfabric.storage import Workspace
from skillfabric.wiki.indexer import page_path
from skillfabric.wiki.loader import WikiSource
from skillfabric.wiki.models import (
    NO_WORKFLOW_GUIDANCE,
    WikiBuildConfig,
    WikiPage,
    WikiSummaryRecord,
)
from skillfabric.wiki.pages import bullet_list, frontmatter, wiki_link


def _skill_page(
    source: WikiSource,
    skill: SkillNode,
    config: WikiBuildConfig,
    summaries: dict[tuple[str, str], WikiSummaryRecord],
    workspace: Workspace,
) -> WikiPage:
    contract = source.contracts.get(skill.id)
    summary = summaries[("skill", skill.id)]
    core_links = source.skill_core_links(skill.id)
    source_slug = f"../sources/{page_path(workspace.wiki_dir, 'skills', skill.id).name}"
    tags = _skill_tags(contract)
    text = (
        "\n\n".join(
            [
                frontmatter(
                    {
                        "type": "Skill Card",
                        "title": skill.name,
                        "description": _one_line(summary.summary),
                        "skill_id": skill.id,
                        "selectable": True,
                        "source": source_slug,
                        "tags": tags,
                    }
                ),
                "# Skill Card",
                f"Skill: {skill.name}",
                "## Purpose\n\n" + _one_line(summary.summary),
                "## Use When\n\n" + _render_use_when(contract, summary),
                "## Do Not Use When\n\n" + _render_do_not_use(contract),
                "## Inputs\n\n" + _render_field_bullets(contract.requires if contract else ()),
                "## Outputs\n\n" + _render_field_bullets(contract.produces if contract else ()),
                "## Tools And Dependencies\n\n"
                + _render_field_bullets(contract.tools if contract else ()),
                "## Composition Notes\n\n"
                + _render_composition_notes(
                    source,
                    core_links,
                    summary=summary,
                    current_skill_id=skill.id,
                    limit=config.max_neighbors_per_section,
                ),
                "## Read Full Source\n\n"
                + f"Open [full SKILL.md]({source_slug}) when the card is insufficient to decide routing boundaries or execution requirements.",
            ]
        )
        + "\n"
    )
    return WikiPage(
        path=page_path(workspace.wiki_skills_dir, "cards", skill.id),
        page_type="skill",
        entity_id=skill.id,
        title=skill.name,
        text=text,
    )


def _workflow_page(
    source: WikiSource,
    edge: Edge,
    workspace: Workspace,
) -> WikiPage:
    source_skill = source.skills.get(edge.source)
    target_skill = source.skills.get(edge.target)
    source_label = source_skill.name if source_skill else edge.source
    target_label = target_skill.name if target_skill else edge.target
    entity_id = f"{edge.source}__{edge.target}__{edge.type}"
    title = f"{source_label} -> {target_label}"
    ordering = (
        f"{source_label} before {target_label} (required handoff)"
        if edge.type == "depend_on"
        else f"{source_label} before {target_label} (workflow order)"
    )
    text = (
        "\n\n".join(
            [
                frontmatter(
                    {
                        "type": "Workflow",
                        "title": title,
                        "description": edge.reason,
                        "workflow_id": entity_id,
                        "tags": ["workflow"],
                    }
                ),
                f"# {title}",
                "## Summary\n\n" + edge.reason,
                "## Skills\n\n"
                + bullet_list(
                    [
                        wiki_link("skills", edge.source, source_label),
                        wiki_link("skills", edge.target, target_label),
                    ]
                ),
                "## Ordering Hint\n\n" + bullet_list([ordering]),
            ]
        )
        + "\n"
    )
    return WikiPage(
        page_path(workspace.wiki_dir, "workflows", entity_id), "workflow", entity_id, title, text
    )


def _skill_source_page(skill: SkillNode, workspace: Workspace) -> WikiPage:
    """Render the full original SKILL.md as the authoritative skill source."""

    source_filename = page_path(workspace.wiki_dir, "skills", skill.id).name
    text = (
        "\n\n".join(
            [
                frontmatter(
                    {
                        "type": "Skill Source",
                        "title": f"{skill.name} Source",
                        "description": f"Full original SKILL.md for {skill.id}.",
                        "skill_id": skill.id,
                        "card": f"../cards/{source_filename}",
                        "resource": f"skill://{skill.id.removeprefix('skill:')}/SKILL.md",
                    }
                ),
                "# Full SKILL.md",
                skill.raw_text.rstrip() or skill.description,
            ]
        )
        + "\n"
    )
    return WikiPage(
        path=workspace.wiki_skill_sources_dir / source_filename,
        page_type="skill",
        entity_id=skill.id,
        title=f"{skill.name} Source",
        text=text,
    )


def _render_use_when(contract: SkillContract | None, summary: WikiSummaryRecord) -> str:
    values = []
    if contract is not None and contract.when_to_use:
        values.append(contract.when_to_use)
    values.append(summary.routing_summary)
    return bullet_list(list(dict.fromkeys(_one_line(value) for value in values if value)))


def _render_do_not_use(contract: SkillContract | None) -> str:
    if contract is None:
        return "- When the task does not match this skill's inputs, outputs, or declared tools."
    return (
        "- When the task requires unrelated inputs, outputs, tools, or execution responsibilities."
    )


def _render_field_bullets(fields: tuple[ContractField, ...]) -> str:
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
    *,
    summary: WikiSummaryRecord,
    current_skill_id: str,
    limit: int,
) -> str:
    values: list[str] = []
    if summary.workflow_summary != NO_WORKFLOW_GUIDANCE:
        values.append(f"- {_one_line(summary.workflow_summary)}")
    relation_text = _render_core_links(
        source,
        [edge for edge in core_links if edge.type in {"compose_with", "depend_on"}],
        current_skill_id=current_skill_id,
        limit=limit,
    )
    if relation_text != "- None":
        values.extend(relation_text.splitlines())
    return "\n".join(dict.fromkeys(values)) if values else "- None"


def _render_core_links(
    source: WikiSource, edges: list[Edge], *, current_skill_id: str, limit: int
) -> str:
    items: list[str] = []
    seen: set[tuple[str, str]] = set()
    for edge in sorted(
        edges, key=lambda item: (item.type, -item.confidence, item.source, item.target)
    ):
        if len(items) >= limit:
            break
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
                items.append(
                    _format_core_link(
                        label, edge, wiki_link("skills", other, source.skills[other].name)
                    )
                )
                seen.add(key)
    return bullet_list(items)


def _format_core_link(label: str, edge: Edge, target: str) -> str:
    suffix = f" confidence={edge.confidence:.2f}" if edge.type == "depend_on" else ""
    return f"{label}: {target}{suffix}"


def _skill_summary_payload(skill: SkillNode, contract: SkillContract | None) -> dict[str, object]:
    return {
        "name": skill.name,
        "description": skill.description,
        "capability": contract.capability if contract else "",
        "when_to_use": contract.when_to_use if contract else "",
        "requires": _field_names(contract.requires) if contract else [],
        "produces": _field_names(contract.produces) if contract else [],
        "tools": _field_names(contract.tools) if contract else [],
    }


def _skill_tags(contract: SkillContract | None) -> list[str]:
    tags = ["skill"]
    if contract is not None:
        tags.extend(name for name in _field_names(contract.produces)[:4] if name)
        tags.extend(name for name in _field_names(contract.tools)[:4] if name)
    return list(dict.fromkeys(_safe_tag(item) for item in tags if item))


def _safe_tag(value: str) -> str:
    return value.strip().lower().replace(" ", "-")


def _one_line(value: str) -> str:
    return " ".join(str(value).split())


def _short_text(value: str, *, limit: int = 240) -> str:
    text = _one_line(value)
    if len(text) <= limit:
        return text
    clipped = text[: max(0, limit - 3)].rsplit(" ", 1)[0].rstrip(" ,;:")
    return f"{clipped}..." if clipped else text[:limit]


def _field_names(fields: tuple[ContractField, ...]) -> list[str]:
    return [field.name for field in fields if field.name]


def _directed_link_label(edge: Edge, current_skill_id: str) -> tuple[str, str]:
    if edge.type == "depend_on":
        if edge.source == current_skill_id:
            return "provides_to", edge.target
        return "consumes_from", edge.source
    if edge.type == "compose_with":
        if edge.source == current_skill_id:
            return "workflow_next", edge.target
        return "workflow_previous", edge.source
    other = edge.target if edge.source == current_skill_id else edge.source
    return edge.type, other


def _first_paragraph(text: str) -> str:
    skipped_keys = (
        "type:",
        "title:",
        "description:",
        "skill_id:",
        "artifact_id:",
        "scenario_id:",
        "name:",
        "selectable:",
        "source:",
        "card:",
        "resource:",
        "Skill:",
        "content_hash:",
        "tags:",
    )
    for line in text.splitlines():
        if (
            line
            and not line.startswith("---")
            and not line.startswith("#")
            and not line.startswith(skipped_keys)
        ):
            return _short_text(line)
    return ""
