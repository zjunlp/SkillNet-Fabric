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
    text = "\n\n".join(
        [
            frontmatter(
                {
                    "type": "skill",
                    "skill_id": skill.id,
                    "name": skill.name,
                    "content_hash": skill.content_hash,
                    "tags": ["skillfabric", "skill"],
                }
            ),
            f"# {skill.name}",
            "## Routing Summary\n\n" + (summary.routing_summary or summary.summary),
            "## Routing Fit\n\n" + _render_routing_fit(skill, interface, summary),
            "## When To Use\n\n" + (summary.workflow_summary or summary.routing_summary or summary.summary),
            "## Produces\n\n" + _render_produces(interface),
            "## Capability Contract\n\n" + _render_interface(interface),
            "## Works With\n\n" + _render_core_links(source, [edge for edge in core_links if edge.type in {"compose_with", "similar_to", "member_of"}], current_skill_id=skill.id, limit=config.max_neighbors_per_section),
            "## Depends On\n\n" + _render_core_links(source, [edge for edge in core_links if edge.type == "depend_on"], current_skill_id=skill.id, limit=config.max_neighbors_per_section),
            "## Workflow Hints\n\n" + _render_workflow_hints(source, execution_links.get("workflow_hints", [])),
            "## Failure Modes\n\n" + _render_failure_modes(interface),
            "## Evidence\n\n" + _render_skill_evidence(skill, core_links),
            "## Source\n\n" + _render_source(skill, config),
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
            frontmatter({"type": "community", "community_id": community_id, "tags": ["skillfabric", "community"]}),
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
            frontmatter({"type": "workflow", "workflow_id": entity_id, "tags": ["skillfabric", "workflow"]}),
            f"# {title}",
            "## Summary\n\n" + f"{record.relation_type} via `{record.canonical_object}`.",
            "## Skills\n\n" + bullet_list(
                [
                    wiki_link("skills", record.source_skill, source_label),
                    wiki_link("skills", record.target_skill, target_label),
                ]
            ),
            "## Ordering Hint\n\n" + bullet_list([f"{source_label} -> {target_label}"]),
            "## Evidence\n\n" + bullet_list([f"{item.skill}:{item.line} - {item.text}" for item in record.evidence[:8]]),
        ]
    ) + "\n"
    return WikiPage(page_path(workspace.wiki_dir, "workflows", entity_id), "workflow", entity_id, title, text)


def _overview_page(source: WikiSource, workspace: Workspace) -> WikiPage:
    text = "\n\n".join(
        [
            frontmatter({"type": "overview", "tags": ["skillfabric", "overview"]}),
            "# SkillFabric Overview",
            "## Counts\n\n"
            + bullet_list(
                [
                    f"skills: {len(source.skills)}",
                    f"communities: {len(source.communities)}",
                    f"workflow hints: {len(source.execution_index)}",
                ]
            ),
            "## Primary Views\n\n"
            + bullet_list(
                [
                    "Use skill pages for routing fit, produces, required deliverables, and workflow hints.",
                    "Use resolver.md for deterministic deliverable and intent aliases.",
                    "Use deliverables.md to find skills by output artifact family.",
                ]
            ),
        ]
    ) + "\n"
    return WikiPage(workspace.wiki_dir / "overview.md", "overview", "overview", "SkillFabric Overview", text)


def _resolver_page(source: WikiSource, workspace: Workspace) -> WikiPage:
    deliverable_rows = [
        "pptx: powerpoint, presentation, slide deck, slides",
        "docx: word document, report document",
        "xlsx: excel, spreadsheet, workbook, tabular data",
        "pdf: pdf document",
        "md: markdown document",
        "png: image asset, figure, chart, plot",
    ]
    intent_rows = [
        "tabular_or_statistical_analysis: csv, spreadsheet, dataframe, descriptive statistics",
        "data_storytelling: narrative, presentation, report story",
        "financial_statement_analysis: financial statement, kpi, year-over-year trend",
    ]
    text = "\n\n".join(
        [
            frontmatter({"type": "resolver", "tags": ["skillfabric", "resolver"]}),
            "# Resolver",
            "## Deliverable Concept Aliases\n\n" + bullet_list(deliverable_rows),
            "## Intent Concept Aliases\n\n" + bullet_list(intent_rows),
            "## Resolution Rules\n\n"
            + bullet_list(
                [
                    "Resolve concrete skills from skill interface produces/requires fields, execution hints, and skill text.",
                    "Prefer explicit deliverable producers over fuzzy similarity when a task names an output file type.",
                    "Treat docx and pptx as distinct deliverables; do not satisfy presentation.pptx with a generic document skill.",
                    "Use tabular/statistical intent skills when a task asks for CSV analysis, descriptive statistics, tests, or spreadsheet work.",
                ]
            ),
        ]
    ) + "\n"
    return WikiPage(workspace.wiki_dir / "resolver.md", "resolver", "resolver", "Resolver", text)


def _deliverables_page(source: WikiSource, workspace: Workspace) -> WikiPage:
    by_produce: dict[str, list[str]] = {}
    for skill_id, interface in source.interfaces.items():
        for field in interface.produces:
            if not field.name:
                continue
            by_produce.setdefault(field.name, []).append(skill_id)
    rows: list[str] = []
    for produce_name, skill_ids in sorted(by_produce.items()):
        links = [
            wiki_link("skills", skill_id, source.skills[skill_id].name)
            for skill_id in sorted(skill_ids)
            if skill_id in source.skills
        ]
        if links:
            rows.append(f"{produce_name}: {', '.join(links)}")
    text = "\n\n".join(
        [
            frontmatter({"type": "deliverables", "tags": ["skillfabric", "deliverables"]}),
            "# Deliverables",
            "## Producer Index\n\n" + bullet_list(rows),
            "## Canonical Deliverable Requirements\n\n"
            + bullet_list(
                [
                    "deliverable:pptx: presentation document",
                    "deliverable:docx: word/report document",
                    "deliverable:xlsx: spreadsheet workbook",
                    "deliverable:pdf: pdf document",
                    "deliverable:md: markdown document",
                    "deliverable:png: image asset",
                ]
            ),
        ]
    ) + "\n"
    return WikiPage(workspace.wiki_dir / "deliverables.md", "deliverables", "deliverables", "Deliverables", text)


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


def _render_interface(interface: SkillInterface | None) -> str:
    if interface is None:
        return "- No interface extracted."
    rows = [
        ("Capability Summary", interface.capability_summary),
        ("When To Use", interface.when_to_use),
        ("Granularity", interface.granularity),
        ("Execution Role", interface.execution_role),
        ("Requires", _field_text(interface.requires)),
        ("Produces", _field_text(interface.produces)),
        ("Uses tools", _field_text(interface.uses_tools)),
    ]
    return bullet_list([f"{label}: {value}" for label, value in rows if value])


def _render_routing_fit(
    skill: SkillNode,
    interface: SkillInterface | None,
    summary: WikiSummaryRecord,
) -> str:
    items = [
        f"Skill id: `{skill.id}`",
        f"Primary fit: {summary.routing_summary or summary.summary or skill.description}",
    ]
    if interface is not None:
        if interface.when_to_use:
            items.append(f"When to route here: {interface.when_to_use}")
        if interface.requires:
            items.append(f"Requires: {_field_text(interface.requires)}")
        if interface.produces:
            items.append(f"Produces: {_field_text(interface.produces)}")
    return bullet_list(items)


def _render_produces(interface: SkillInterface | None) -> str:
    if interface is None or not interface.produces:
        return "- No explicit produced artifacts extracted."
    items = []
    for field in interface.produces:
        text = field.name
        if field.description:
            text = f"{text}: {field.description}"
        items.append(text)
    return bullet_list(items)


def _render_failure_modes(interface: SkillInterface | None) -> str:
    if interface is None:
        return "- None"
    return bullet_list(_field_names(interface.failure_modes))


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


def _render_skill_evidence(skill: SkillNode, core_links: list[Edge]) -> str:
    values = [f"source_path: `{skill.source_path}`", f"content_hash: `{skill.content_hash}`"]
    for edge in core_links[:5]:
        if edge.reason:
            values.append(f"{edge.type}: {edge.reason}")
        for evidence in edge.evidence[:2]:
            values.append(f"{evidence.skill}:{evidence.line} - {evidence.text}")
    return bullet_list(values)


def _render_source(skill: SkillNode, config: WikiBuildConfig) -> str:
    if not config.include_raw_skill_excerpt:
        return f"- Source path: `{skill.source_path}`"
    excerpt = skill.raw_text[: config.raw_excerpt_chars]
    fence = _markdown_fence(excerpt)
    return f"- Source path: `{skill.source_path}`\n\n{fence}markdown\n{excerpt}\n{fence}"


def _skill_summary_payload(skill: SkillNode, interface: SkillInterface | None) -> dict[str, object]:
    return {
        "name": skill.name,
        "description": skill.description,
        "when_to_use": interface.when_to_use if interface else "",
        "requires": _field_names(interface.requires) if interface else [],
        "produces": _field_names(interface.produces) if interface else [],
        "uses_tools": _field_names(interface.uses_tools) if interface else [],
    }


def _field_names(fields: list[InterfaceField]) -> list[str]:
    return [field.name for field in fields if field.name]


def _field_text(fields: list[InterfaceField]) -> str:
    return ", ".join(_field_names(fields))


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


def _markdown_fence(text: str) -> str:
    longest = 0
    current = 0
    for char in text:
        if char == "`":
            current += 1
            longest = max(longest, current)
            continue
        current = 0
    return "`" * max(3, longest + 1)


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
