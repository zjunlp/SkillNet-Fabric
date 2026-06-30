"""Dynamic coverage resolution for router bundles."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from skillfabric.artifact_ontology import canonical_artifact_name, normalize_artifact_phrase
from skillfabric.compiled_graph.execution.models import ExecutionIndexRecord
from skillfabric.compiled_graph.interface.models import SkillInterface
from skillfabric.compiled_graph.models import GraphDocument
from skillfabric.indexing.canonical import canonical_skill_text
from skillfabric.registry.models import SkillNode
from skillfabric.storage import Workspace
from skillfabric.task_understanding import CoverageRequirement, TaskUnderstanding

GENERIC_MATCH_TERMS = {
    "analysis",
    "analyze",
    "task",
    "intent",
    "deliverable",
    "artifact",
    "output",
    "document",
    "data",
    "result",
    "results",
    "summary",
    "report",
}
GENERIC_CANONICAL_TERMS = {
    "artifact",
    "output",
    "result",
    "results",
    "data",
    "document",
    "json_data",
    "markdown_document",
    "text_document",
}
LOW_INFORMATION_CONTENT_TERMS = {
    "assistant",
    "being",
    "count",
    "display",
    "each",
    "element",
    "elements",
    "have",
    "input",
    "name",
    "please",
    "quick",
    "same",
    "start",
    "their",
    "time",
    "want",
}
INTENT_TERM_PROFILES: dict[str, dict[str, set[str]]] = {
    "intent:tabular_or_statistical_analysis": {
        "requirement": {
            "tabular analysis",
            "statistical analysis",
            "csv",
            "spreadsheet",
            "dataframe",
            "dataset",
            "descriptive statistics",
            "statistical_summary",
        },
    },
    "intent:data_storytelling": {
        "requirement": {"data story", "narrative", "presentation", "slides", "report", "storytelling"},
    },
    "intent:financial_statement_analysis": {
        "requirement": {
            "financial statement",
            "financial kpi",
            "kpi",
            "year over year",
            "financial analysis",
            "executive summary",
        },
    },
    "intent:symbolic_math": {
        "requirement": {
            "symbolic math",
            "symbolic mathematics",
            "symbolic computation",
            "exact algebra",
            "calculus",
            "jacobian",
            "eigenvalue",
            "eigenvalues",
            "matrix",
            "differential equation",
            "stability analysis",
            "sympy",
        },
        "positive": {
            "algebra",
            "calculus",
            "differential",
            "eigenvalue",
            "eigenvalues",
            "equation",
            "equations",
            "exact",
            "integral",
            "integrals",
            "jacobian",
            "matrix",
            "matrices",
            "symbolic",
            "sympy",
        },
        "negative": {"generic", "report", "ui"},
    },
    "intent:browser_interaction": {
        "requirement": {
            "browser automation",
            "web browsing",
            "navigate website",
            "screenshot",
            "screenshots",
            "scrape",
            "crawl",
            "web data extraction",
            "playwright",
            "chromium",
            "browser observation",
        },
        "positive": {
            "browser",
            "browsing",
            "chromium",
            "crawl",
            "navigate",
            "playwright",
            "scrape",
            "screenshot",
            "screenshots",
            "web",
            "website",
        },
        "required": {
            "browser",
            "chromium",
            "navigate",
            "page",
            "playwright",
            "screenshot",
            "screenshots",
            "website",
        },
        "negative": {"report", "reports", "newsletter", "writing"},
    },
    "intent:image_generation": {
        "requirement": {
            "image generation",
            "generate image",
            "generated image",
            "text to image",
            "illustration",
            "artwork",
            "meme",
            "poster",
            "synthesize image",
        },
        "positive": {
            "artwork",
            "edit",
            "edited",
            "generate",
            "generated",
            "generation",
            "image",
            "images",
            "illustration",
            "meme",
            "poster",
            "prompt",
            "synthesize",
        },
        "required": {
            "artwork",
            "banana",
            "edit",
            "edited",
            "fal",
            "gemini",
            "generate",
            "generate-image",
            "generated",
            "generation",
            "illustration",
            "meme",
            "nanobanana",
            "nano",
            "poster",
            "prompt",
            "synthesize",
        },
        "negative": {
            "cdn",
            "cloudflare",
            "delivery",
            "host",
            "hosting",
            "storage",
            "store",
            "upload",
            "variant",
            "variants",
        },
    },
    "intent:image_processing": {
        "requirement": {
            "image processing",
            "resize image",
            "thumbnail",
            "composite image",
            "combine images",
            "banner",
            "collage",
            "format conversion",
            "pillow",
            "imagemagick",
        },
        "positive": {
            "banner",
            "collage",
            "combine",
            "composite",
            "conversion",
            "crop",
            "grid",
            "imagemagick",
            "pillow",
            "processing",
            "resize",
            "thumbnail",
            "thumbnails",
        },
        "required": {
            "banner",
            "collage",
            "combine",
            "composite",
            "conversion",
            "crop",
            "grid",
            "imagemagick",
            "pillow",
            "processing",
            "resize",
            "thumbnail",
            "thumbnails",
        },
        "negative": {"cdn", "cloudflare", "delivery", "host", "hosting", "storage", "store", "upload"},
    },
    "intent:document_creation": {
        "requirement": {
            "docx",
            "word document",
            "document creation",
            "tracked changes",
            "heading",
            "signature lines",
            "ooxml",
            "docx js",
        },
        "positive": {"docx", "document", "documents", "heading", "ooxml", "redlining", "signature", "tracked", "word"},
    },
    "intent:react_video": {
        "requirement": {"remotion", "react video", "video composition", "animation", "1080p", "composition", "timeline"},
        "positive": {"composition", "react", "remotion", "timeline", "video"},
        "required": {"composition", "remotion", "render", "src", "timeline", "video", "1080p"},
        "negative": {"query", "server", "state", "tanstack"},
    },
    "intent:scroll_web_experience": {
        "requirement": {
            "scroll",
            "scroll-driven",
            "parallax",
            "scroll animation",
            "scroll storytelling",
            "progress indicator",
            "sticky section",
        },
        "positive": {"animation", "parallax", "progress", "scroll", "scrolltrigger", "sticky"},
    },
}


@dataclass(slots=True)
class CoverageResolutionEvidence:
    """Evidence that a skill can satisfy one coverage requirement."""

    source: str
    field: str
    value: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "field": self.field,
            "value": self.value,
            "score": round(float(self.score), 6),
        }


@dataclass(slots=True)
class CoverageSkillMatch:
    """One candidate skill match for a coverage requirement."""

    skill_id: str
    score: float = 0.0
    evidence: list[CoverageResolutionEvidence] = field(default_factory=list)

    def add(self, source: str, field: str, value: str, score: float) -> None:
        if score <= 0:
            return
        self.score += score
        self.evidence.append(CoverageResolutionEvidence(source, field, value, score))


@dataclass(slots=True)
class CoverageSkillProfile:
    """Precomputed text facets for one skill during coverage resolution."""

    skill: SkillNode
    interface: SkillInterface | None
    registry_terms: set[str]
    registry_short_terms: set[str]
    operation_terms: set[str]
    content_terms: set[str]


def resolve_coverage_requirements(
    understanding: TaskUnderstanding,
    *,
    skills: dict[str, SkillNode],
    interfaces: dict[str, SkillInterface] | None = None,
    execution_index: list[ExecutionIndexRecord] | None = None,
    graph: GraphDocument | None = None,
) -> TaskUnderstanding:
    """Populate coverage requirement skill ids from built SkillFabric artifacts."""

    interfaces = interfaces or {}
    execution_index = execution_index or []
    profiles = {
        skill_id: _coverage_skill_profile(skill, interfaces.get(skill_id))
        for skill_id, skill in skills.items()
    }
    diagnostics: list[dict[str, Any]] = []
    resolved_requirements: list[CoverageRequirement] = []
    for requirement in understanding.coverage_requirements:
        matches = _matches_for_requirement(
            requirement,
            profiles=profiles,
            execution_index=execution_index,
            graph=graph,
            query=understanding.query,
        )
        ranked = sorted(matches.values(), key=lambda item: (-item.score, item.skill_id))
        acceptable = [item.skill_id for item in ranked]
        preferred = [item.skill_id for item in ranked if item.score >= _preferred_threshold(requirement)]
        if not preferred and acceptable:
            preferred = acceptable[:1]
        resolved_requirements.append(
            CoverageRequirement(
                id=requirement.id,
                kind=requirement.kind,
                label=requirement.label,
                acceptable_skill_ids=acceptable,
                preferred_skill_ids=preferred,
                format=requirement.format,
                minimum_count=requirement.minimum_count,
                source_text=requirement.source_text,
                requires_preferred=requirement.requires_preferred,
            )
        )
        status = "missing"
        if len(acceptable) == 1:
            status = "resolved"
        elif acceptable:
            status = "ambiguous"
        diagnostics.append(
            {
                "requirement_id": requirement.id,
                "status": status,
                "preferred_skill_ids": preferred,
                "acceptable_skill_ids": acceptable,
                "evidence": [
                    {
                        "skill_id": match.skill_id,
                        "score": round(float(match.score), 6),
                        "evidence": [item.to_dict() for item in match.evidence],
                    }
                    for match in ranked
                ],
            }
        )
    return TaskUnderstanding(
        query=understanding.query,
        input_artifacts=list(understanding.input_artifacts),
        required_deliverables=list(understanding.required_deliverables),
        analysis_intents=list(understanding.analysis_intents),
        domain_hints=list(understanding.domain_hints),
        coverage_requirements=resolved_requirements,
        coverage_diagnostics=diagnostics,
    )


def load_interfaces(workspace: Workspace) -> dict[str, SkillInterface]:
    """Load skill_interfaces.jsonl if present."""

    path = workspace.interfaces_dir / "skill_interfaces.jsonl"
    if not path.exists():
        return {}
    return _load_interfaces_cached(*_file_cache_key(path))


@lru_cache(maxsize=16)
def _load_interfaces_cached(path: str, _mtime_ns: int, _size: int) -> dict[str, SkillInterface]:
    interfaces: dict[str, SkillInterface] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            interface = SkillInterface.from_dict(json.loads(line))
            interfaces[interface.skill_id] = interface
    return interfaces


def load_execution_index(workspace: Workspace) -> list[ExecutionIndexRecord]:
    """Load execution_index.jsonl if present."""

    path = workspace.execution_dir / "execution_index.jsonl"
    if not path.exists():
        return []
    return _load_execution_index_cached(*_file_cache_key(path))


@lru_cache(maxsize=16)
def _load_execution_index_cached(path: str, _mtime_ns: int, _size: int) -> list[ExecutionIndexRecord]:
    return [
        ExecutionIndexRecord.from_dict(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _file_cache_key(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return (str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def _matches_for_requirement(
    requirement: CoverageRequirement,
    *,
    profiles: dict[str, CoverageSkillProfile],
    execution_index: list[ExecutionIndexRecord],
    graph: GraphDocument | None,
    query: str = "",
) -> dict[str, CoverageSkillMatch]:
    matches: dict[str, CoverageSkillMatch] = {}
    terms = _requirement_terms(requirement)
    canonical_terms = {canonical_artifact_name(term) for term in terms}
    canonical_terms.discard("")
    for skill_id, profile in profiles.items():
        match = CoverageSkillMatch(skill_id)
        interface = profile.interface
        if requirement.kind == "deliverable":
            _score_interface_fields(match, interface, "produces", terms, canonical_terms, primary=True)
            _score_skill_text(match, profile, terms, canonical_terms, weight=0.8)
        elif requirement.kind == "intent":
            _score_interface_text(match, interface, terms, weight=1.2)
            _score_interface_fields(match, interface, "requires", terms, canonical_terms, primary=False)
            _score_interface_fields(match, interface, "produces", terms, canonical_terms, primary=False)
            _score_interface_fields(match, interface, "uses_tools", terms, canonical_terms, primary=False)
            _score_skill_text(match, profile, terms, canonical_terms, weight=0.7)
            _score_execution_records(match, skill_id, execution_index, terms, canonical_terms)
        else:
            _score_interface_fields(match, interface, "requires", terms, canonical_terms, primary=True)
            _score_skill_text(match, profile, terms, canonical_terms, weight=0.5)
        if graph is not None:
            _score_graph_context(match, skill_id, graph, weight=0.1)
        _adjust_for_deliverable_format(match, requirement, profile, query=query)
        _adjust_for_capability_intent(match, requirement, profile)
        _adjust_for_task_operation(match, requirement, profile, query=query)
        if _is_unrelated_auxiliary_deliverable(match, requirement, profile):
            continue
        if match.score >= _minimum_score(requirement):
            matches[skill_id] = match
    return matches


def _score_interface_fields(
    match: CoverageSkillMatch,
    interface: SkillInterface | None,
    field_name: str,
    terms: set[str],
    canonical_terms: set[str],
    *,
    primary: bool,
) -> None:
    if interface is None:
        return
    weight = 2.0 if primary else 1.0
    for interface_field in getattr(interface, field_name):
        field_terms = _value_terms(interface_field.name, interface_field.description)
        score = _term_score(field_terms, terms, canonical_terms)
        match.add(
            "interface",
            field_name,
            interface_field.name,
            weight * score * max(interface_field.confidence, 0.4),
        )


def _score_interface_text(
    match: CoverageSkillMatch,
    interface: SkillInterface | None,
    terms: set[str],
    *,
    weight: float,
) -> None:
    if interface is None:
        return
    values = [
        ("capability_summary", interface.capability_summary),
        ("when_to_use", interface.when_to_use),
        ("execution_role", interface.execution_role),
        ("granularity", interface.granularity),
        ("uses_tools", " ".join(field.name for field in interface.uses_tools)),
        ("uses_tools", " ".join(field.description for field in interface.uses_tools)),
    ]
    for field_name, value in values:
        score = _text_token_score(value, terms)
        match.add("interface", field_name, value, weight * score)


def _score_skill_text(
    match: CoverageSkillMatch,
    profile: CoverageSkillProfile,
    terms: set[str],
    canonical_terms: set[str],
    *,
    weight: float,
) -> None:
    score = max(
        _term_score(profile.registry_terms, terms, set()),
        _term_score(profile.registry_short_terms, terms, canonical_terms),
    )
    match.add("registry", "canonical_text", profile.skill.name, weight * score)


def _score_execution_records(
    match: CoverageSkillMatch,
    skill_id: str,
    records: list[ExecutionIndexRecord],
    terms: set[str],
    canonical_terms: set[str],
) -> None:
    for record in records:
        if record.source_skill != skill_id and record.target_skill != skill_id:
            continue
        field_terms = _value_terms(record.canonical_object, record.relation_type, record.reason)
        score = _term_score(field_terms, terms, canonical_terms)
        match.add("execution_index", "canonical_object", record.canonical_object, 0.8 * score * record.confidence)


def _adjust_for_capability_intent(
    match: CoverageSkillMatch,
    requirement: CoverageRequirement,
    profile: CoverageSkillProfile,
) -> None:
    if requirement.kind != "intent":
        return
    intent_profile = INTENT_TERM_PROFILES.get(requirement.id, {})
    positive_terms = intent_profile.get("positive", set())
    if not positive_terms:
        return
    skill_terms = profile.operation_terms
    positive_matches = skill_terms & positive_terms
    required_terms = intent_profile.get("required", set())
    required_matches = skill_terms & required_terms
    if required_terms and not required_matches:
        match.score *= 0.35
        match.evidence.append(
            CoverageResolutionEvidence(
                "task_context",
                "capability_intent",
                f"missing_required_operation:{requirement.id}",
                -0.65,
            )
        )
        return
    if positive_matches:
        weight = 2.4 if len(positive_matches) >= 2 else 1.5
        if profile.interface is not None and profile.interface.execution_role in {"transformer", "actor", "navigator", "inspector"}:
            weight += 0.4
        if requirement.id == "intent:browser_interaction" and skill_terms & {"dev", "local", "persistent", "playwright"}:
            weight += 1.4
        if requirement.id == "intent:image_generation" and skill_terms & {"batch", "banana", "fal", "gemini", "nanobanana", "nano"}:
            weight += 1.8
        if requirement.id == "intent:react_video" and skill_terms & {"composition", "remotion", "timeline"}:
            weight += 2.2
        match.add("task_context", "capability_intent", requirement.id, weight)
    negative_matches = skill_terms & intent_profile.get("negative", set())
    if negative_matches and len(positive_matches) < 2:
        match.score *= 0.5
        match.evidence.append(
            CoverageResolutionEvidence(
                "task_context",
                "capability_intent",
                f"operation_mismatch:{requirement.id}",
                -0.5,
            )
        )
    requirement_terms = _value_terms(requirement.source_text, requirement.label)
    _adjust_for_specific_intent_quality(match, requirement.id, skill_terms, requirement_terms)


def _score_graph_context(
    match: CoverageSkillMatch,
    skill_id: str,
    graph: GraphDocument,
    *,
    weight: float,
) -> None:
    count = sum(1 for edge in graph.edges if edge.source == skill_id or edge.target == skill_id)
    if count:
        match.add("graph", "degree", str(count), min(weight, count * 0.01))


def _adjust_for_specific_intent_quality(
    match: CoverageSkillMatch,
    requirement_id: str,
    skill_terms: set[str],
    requirement_terms: set[str],
) -> None:
    if requirement_id == "intent:react_video":
        strong = skill_terms & {"remotion", "timeline", "1080p"}
        if "video" in skill_terms and skill_terms & {"composition", "render"}:
            strong.add("video_composition")
        task_needs_video_project = bool(requirement_terms & {"1080p", "remotion", "src", "video"})
        weak_react_only = bool(skill_terms & {"react", "animation", "video"}) and not strong
        if strong:
            match.add("task_context", "intent_quality", "react_video_specialist", 8.0 if task_needs_video_project else 3.0)
        elif weak_react_only:
            _penalize_match(match, "task_context", "intent_quality", "react_without_video_composition_specialization", factor=0.2 if task_needs_video_project else 0.45)
    elif requirement_id == "intent:browser_interaction":
        local_browser = skill_terms & {"dev", "local", "persistent", "playwright", "chromium"}
        remote_or_testing = skill_terms & {"cloudflare", "firecrawl", "test", "testing", "mcp", "agent"}
        if local_browser:
            match.add("task_context", "intent_quality", "local_browser_execution", 2.2)
        elif remote_or_testing:
            _penalize_match(match, "task_context", "intent_quality", "indirect_browser_or_testing_tool", factor=0.75)
    elif requirement_id == "intent:image_generation":
        image_generation_tool = skill_terms & {"banana", "fal", "gemini", "generate", "generated", "generation", "nanobanana", "nano", "prompt"}
        storage_or_doc_tool = skill_terms & {"cloudflare", "delivery", "pdf", "pptx", "slide", "slides", "storage", "upload", "variant"}
        if image_generation_tool:
            match.add("task_context", "intent_quality", "dedicated_image_generation_tool", 2.4)
        if storage_or_doc_tool and not image_generation_tool:
            _penalize_match(match, "task_context", "intent_quality", "non_generation_image_adjacent_tool", factor=0.45)
    elif requirement_id == "intent:image_processing":
        processing_tool = skill_terms & {"banner", "collage", "combine", "composite", "crop", "imagemagick", "pillow", "resize", "thumbnail"}
        storage_or_doc_tool = skill_terms & {"cloudflare", "pdf", "pptx", "slide", "slides", "storage", "upload", "variant"}
        if processing_tool:
            match.add("task_context", "intent_quality", "dedicated_image_processing_tool", 2.0)
        if storage_or_doc_tool and not processing_tool:
            _penalize_match(match, "task_context", "intent_quality", "non_processing_image_adjacent_tool", factor=0.5)


def _adjust_for_task_operation(
    match: CoverageSkillMatch,
    requirement: CoverageRequirement,
    profile: CoverageSkillProfile,
    *,
    query: str = "",
) -> None:
    if requirement.kind != "deliverable" or requirement.format != "mp4":
        return
    source_terms = _value_terms(requirement.source_text, query)
    task_is_creation = bool(source_terms & {"create", "generate", "render", "animate", "animation", "explainer"})
    task_has_existing_media_input = bool(source_terms & {"url", "youtube", "download", "clip", "transcript", "subtitle"})
    if not task_is_creation or task_has_existing_media_input:
        return
    skill_terms = profile.operation_terms
    creation_terms = {"create", "generate", "render", "animate", "animated"}
    video_terms = {"video", "mp4", "animation", "explainer"}
    distribution_terms = {
        "caption",
        "captions",
        "clip",
        "download",
        "platform",
        "post",
        "publish",
        "reel",
        "reels",
        "social",
        "subtitle",
        "transcript",
        "url",
        "youtube",
    }
    if skill_terms & creation_terms and skill_terms & video_terms and not skill_terms & distribution_terms:
        match.add("task_context", "operation", "create_render_video", 3.0)
    if skill_terms & distribution_terms:
        match.score *= 0.45
        match.evidence.append(
            CoverageResolutionEvidence(
                "task_context",
                "operation",
                "media_distribution_skill_for_creation_task",
                -0.55,
            )
        )


def _adjust_for_deliverable_format(
    match: CoverageSkillMatch,
    requirement: CoverageRequirement,
    profile: CoverageSkillProfile,
    *,
    query: str = "",
) -> None:
    if requirement.kind != "deliverable":
        return
    skill_terms = profile.operation_terms
    source_terms = _value_terms(requirement.source_text, requirement.label, query)
    if requirement.format == "png":
        _adjust_for_png_deliverable(match, skill_terms, source_terms)
    elif requirement.format in {"docx", "xlsx", "pptx", "pdf", "html"}:
        _adjust_for_exact_format_deliverable(match, requirement.format, skill_terms)


def _adjust_for_png_deliverable(
    match: CoverageSkillMatch,
    skill_terms: set[str],
    source_terms: set[str],
) -> None:
    chart_task = bool(
        source_terms
        & {
            "bar",
            "chart",
            "charts",
            "figure",
            "figures",
            "graph",
            "label",
            "labels",
            "latex",
            "plot",
            "plots",
            "publication",
            "visualization",
            "visualizations",
        }
    )
    if source_terms & {"comparison", "error", "errors"} and source_terms & {"integral", "integrals", "numeric", "numerical"}:
        chart_task = True
    screenshot_task = bool(source_terms & {"capture", "screenshot", "screenshots"}) or (
        bool(source_terms & {"browse", "browsing", "navigate", "visit"})
        and bool(source_terms & {"browser", "website", "websites"})
    )
    generated_image_task = bool(
        source_terms
        & {
            "artwork",
            "banner",
            "draw",
            "generate",
            "generated",
            "image",
            "illustration",
            "poster",
            "synthesize",
            "thumbnail",
        }
    )
    if chart_task:
        infrastructure_terms = {"cdn", "cloudflare", "helm", "hosting", "kubernetes", "manifest", "package", "storage", "upload", "variant"}
        strong_chart_skill_terms = {
            "figure",
            "figures",
            "matplotlib",
            "plot",
            "plotting",
            "publication",
            "seaborn",
            "visualization",
            "visualizations",
        }
        chart_skill_terms = {
            "chart",
            "charts",
            "figure",
            "figures",
            "matplotlib",
            "plot",
            "plotting",
            "publication",
            "seaborn",
            "visualization",
            "visualizations",
        }
        if skill_terms & infrastructure_terms and not skill_terms & strong_chart_skill_terms:
            _reject_match(match, "task_context", "deliverable_format", "png_infrastructure_mismatch")
            return
        if skill_terms & chart_skill_terms:
            match.add("task_context", "deliverable_format", "png_chart_or_figure", 2.8)
        elif not (skill_terms & {"image", "png", "screenshot"}):
            _reject_match(match, "task_context", "deliverable_format", "png_without_visual_output_evidence")
    elif screenshot_task:
        if skill_terms & {"browser", "capture", "chromium", "page", "playwright", "screenshot", "screenshots", "website"}:
            match.add("task_context", "deliverable_format", "png_browser_screenshot", 2.5)
        elif skill_terms & {"cdn", "cloudflare", "hosting", "storage", "upload", "variant"}:
            _penalize_match(match, "task_context", "deliverable_format", "png_storage_mismatch", factor=0.4)
    elif generated_image_task:
        if skill_terms & {"artwork", "banner", "generate", "generated", "generation", "image", "illustration", "poster", "prompt"}:
            match.add("task_context", "deliverable_format", "png_generated_image", 2.2)
        elif skill_terms & {"cdn", "cloudflare", "hosting", "storage", "upload", "variant"}:
            _penalize_match(match, "task_context", "deliverable_format", "png_storage_mismatch", factor=0.45)


def _adjust_for_exact_format_deliverable(
    match: CoverageSkillMatch,
    fmt: str,
    skill_terms: set[str],
) -> None:
    required_terms = {
        "docx": {"docx", "word", "ooxml", "office"},
        "xlsx": {"excel", "office", "sheetjs", "spreadsheet", "workbook", "xlsx"},
        "pptx": {"deck", "office", "powerpoint", "presentation", "pptx", "slides"},
        "pdf": {"pdf"},
        "html": {"html", "webpage", "website", "frontend", "react", "browser"},
    }[fmt]
    if skill_terms & required_terms:
        match.add("task_context", "deliverable_format", f"{fmt}_format_evidence", 2.4)
        return
    if fmt == "docx" and skill_terms & {"report", "reports", "research", "analysis"}:
        _reject_match(match, "task_context", "deliverable_format", "docx_report_without_word_format")
    elif fmt == "html" and skill_terms & {"scrape", "scraper", "crawl", "crawler"}:
        _reject_match(match, "task_context", "deliverable_format", "html_input_tool_without_page_creation")
    else:
        _reject_match(match, "task_context", "deliverable_format", f"{fmt}_missing_format_evidence")


def _penalize_match(
    match: CoverageSkillMatch,
    source: str,
    field: str,
    value: str,
    *,
    factor: float,
) -> None:
    match.score *= factor
    match.evidence.append(CoverageResolutionEvidence(source, field, value, factor - 1.0))


def _reject_match(match: CoverageSkillMatch, source: str, field: str, value: str) -> None:
    match.score = 0.0
    match.evidence.append(CoverageResolutionEvidence(source, field, value, -1.0))


def _coverage_skill_profile(skill: SkillNode, interface: SkillInterface | None) -> CoverageSkillProfile:
    registry_text = f"{skill.name} {skill.description} {canonical_skill_text(skill)}"
    operation_text = _skill_operation_text(skill, interface)
    return CoverageSkillProfile(
        skill=skill,
        interface=interface,
        registry_terms=_value_terms(registry_text),
        registry_short_terms=_value_terms(registry_text[:1200]),
        operation_terms=_value_terms(operation_text),
        content_terms=_content_terms(operation_text),
    )


def _skill_operation_text(skill: SkillNode, interface: SkillInterface | None) -> str:
    values = [skill.name, skill.description, canonical_skill_text(skill)[:1200]]
    if interface is not None:
        values.extend(
            [
                interface.capability_summary,
                interface.when_to_use,
                " ".join(field.name for field in interface.requires),
                " ".join(field.description for field in interface.requires),
                " ".join(field.name for field in interface.produces),
                " ".join(field.description for field in interface.produces),
                " ".join(field.name for field in interface.uses_tools),
                " ".join(field.description for field in interface.uses_tools),
            ]
        )
    return " ".join(values)


def _is_unrelated_auxiliary_deliverable(
    match: CoverageSkillMatch,
    requirement: CoverageRequirement,
    profile: CoverageSkillProfile,
) -> bool:
    if requirement.kind != "deliverable" or requirement.format not in {"json", "md", "txt"}:
        return False
    target_terms = _content_terms(" ".join([requirement.source_text, requirement.label]))
    if not target_terms:
        return False
    return not bool(target_terms & profile.content_terms)


def _requirement_terms(requirement: CoverageRequirement) -> set[str]:
    values = {
        requirement.id,
        requirement.kind,
        requirement.label,
        requirement.format,
    }
    if requirement.kind == "deliverable":
        values.add(requirement.source_text)
        values.update(_deliverable_aliases(requirement.format))
    else:
        values.update(INTENT_TERM_PROFILES.get(requirement.id, {}).get("requirement", set()))
    return {term for value in values for term in _value_terms(value)}


def _deliverable_aliases(fmt: str) -> set[str]:
    return {
        "pptx": {"pptx", "powerpoint", "presentation", "slide deck", "slides", "presentation_document"},
        "docx": {"docx", "word document", "report document", "docx_document"},
        "xlsx": {"xlsx", "excel", "spreadsheet", "workbook", "spreadsheet_table"},
        "pdf": {"pdf", "pdf document", "pdf_document"},
        "md": {"markdown", "md", "markdown document", "markdown_document"},
        "png": {"png", "image", "figure", "chart", "plot", "image_asset"},
    }.get(fmt, {fmt})


def _value_terms(*values: str) -> set[str]:
    terms: set[str] = set()
    for value in values:
        terms.update(_value_terms_one(str(value)))
    return terms


@lru_cache(maxsize=65_536)
def _value_terms_one(value: str) -> frozenset[str]:
    terms: set[str] = set()
    for raw in re.findall(r"[a-z0-9][a-z0-9_.+-]*", value.lower()):
        normalized = normalize_artifact_phrase(raw)
        if not normalized:
            continue
        terms.add(normalized)
        terms.update(normalized.split())
        canonical = canonical_artifact_name(normalized)
        if canonical:
            terms.add(canonical)
    normalized = normalize_artifact_phrase(value)
    if normalized:
        terms.add(normalized)
        terms.update(normalized.split())
        canonical = canonical_artifact_name(normalized)
        if canonical:
            terms.add(canonical)
    return frozenset(terms)


def _content_terms(text: str) -> set[str]:
    return {
        term
        for term in _value_terms(text)
        if len(term) >= 4
        and " " not in term
        and term not in GENERIC_MATCH_TERMS
        and term not in GENERIC_CANONICAL_TERMS
        and term not in LOW_INFORMATION_CONTENT_TERMS
        and term not in {"json", "markdown", "text", "file", "create", "generate", "output", "write"}
    }


def _term_score(field_terms: set[str], terms: set[str], canonical_terms: set[str]) -> float:
    if not field_terms:
        return 0.0
    specific_canonical_terms = canonical_terms - GENERIC_CANONICAL_TERMS
    if field_terms & specific_canonical_terms:
        return 1.0
    specific_field_terms = field_terms - GENERIC_MATCH_TERMS
    specific_terms = terms - GENERIC_MATCH_TERMS
    if specific_field_terms & specific_terms:
        return 0.8
    phrases = {term for term in terms if " " in term and not set(term.split()) <= GENERIC_MATCH_TERMS}
    for phrase in phrases:
        if all(part in field_terms for part in phrase.split()):
            return 0.7
    return 0.0


def _text_token_score(text: str, terms: set[str]) -> float:
    field_terms = _value_terms(text)
    return _term_score(field_terms, terms, set())


def _minimum_score(requirement: CoverageRequirement) -> float:
    if requirement.id.startswith("intent:"):
        return 1.15
    return 0.9 if requirement.kind == "deliverable" else 1.0


def _preferred_threshold(requirement: CoverageRequirement) -> float:
    if requirement.id.startswith("intent:"):
        return 1.8
    return 1.3 if requirement.kind == "deliverable" else 1.4


__all__ = [
    "load_execution_index",
    "load_interfaces",
    "resolve_coverage_requirements",
]
