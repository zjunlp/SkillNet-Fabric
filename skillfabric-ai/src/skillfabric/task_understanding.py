"""Deterministic task understanding and coverage checks for routing/planning."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TaskArtifact:
    """Input artifact mentioned by a user task."""

    path: str
    format: str
    source_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "format": self.format, "source_text": self.source_text}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TaskArtifact:
        return cls(
            path=str(payload.get("path", "")),
            format=str(payload.get("format", "")),
            source_text=str(payload.get("source_text", "")),
        )


@dataclass(slots=True)
class RequiredDeliverable:
    """Explicit artifact or output family required by the task."""

    format: str
    label: str
    path: str = ""
    minimum_count: int = 1
    source_text: str = ""

    @property
    def id(self) -> str:
        return f"deliverable:{self.format}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "format": self.format,
            "label": self.label,
            "path": self.path,
            "minimum_count": self.minimum_count,
            "source_text": self.source_text,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RequiredDeliverable:
        return cls(
            format=str(payload.get("format", "")),
            label=str(payload.get("label", "")),
            path=str(payload.get("path", "")),
            minimum_count=_safe_int(payload.get("minimum_count"), 1),
            source_text=str(payload.get("source_text", "")),
        )


@dataclass(slots=True)
class CoverageRequirement:
    """One required routing/planning capability with acceptable skill coverage."""

    id: str
    kind: str
    label: str
    acceptable_skill_ids: list[str] = field(default_factory=list)
    preferred_skill_ids: list[str] = field(default_factory=list)
    format: str = ""
    minimum_count: int = 1
    source_text: str = ""
    requires_preferred: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "acceptable_skill_ids": list(self.acceptable_skill_ids),
            "preferred_skill_ids": list(self.preferred_skill_ids),
            "format": self.format,
            "minimum_count": self.minimum_count,
            "source_text": self.source_text,
            "requires_preferred": self.requires_preferred,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CoverageRequirement:
        return cls(
            id=str(payload.get("id", "")),
            kind=str(payload.get("kind", "")),
            label=str(payload.get("label", "")),
            acceptable_skill_ids=_string_list(payload.get("acceptable_skill_ids", [])),
            preferred_skill_ids=_string_list(payload.get("preferred_skill_ids", [])),
            format=str(payload.get("format", "")),
            minimum_count=_safe_int(payload.get("minimum_count"), 1),
            source_text=str(payload.get("source_text", "")),
            requires_preferred=bool(payload.get("requires_preferred", False)),
        )


@dataclass(slots=True)
class TaskUnderstanding:
    """Compact deterministic understanding of a user task."""

    query: str
    input_artifacts: list[TaskArtifact] = field(default_factory=list)
    required_deliverables: list[RequiredDeliverable] = field(default_factory=list)
    analysis_intents: list[str] = field(default_factory=list)
    domain_hints: list[str] = field(default_factory=list)
    coverage_requirements: list[CoverageRequirement] = field(default_factory=list)
    coverage_diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "input_artifacts": [item.to_dict() for item in self.input_artifacts],
            "required_deliverables": [item.to_dict() for item in self.required_deliverables],
            "analysis_intents": list(self.analysis_intents),
            "domain_hints": list(self.domain_hints),
            "coverage_requirements": [item.to_dict() for item in self.coverage_requirements],
            "coverage_diagnostics": [dict(item) for item in self.coverage_diagnostics],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> TaskUnderstanding:
        if not isinstance(payload, dict):
            return cls(query="")
        return cls(
            query=str(payload.get("query", "")),
            input_artifacts=[
                TaskArtifact.from_dict(item)
                for item in payload.get("input_artifacts", [])
                if isinstance(item, dict)
            ],
            required_deliverables=[
                RequiredDeliverable.from_dict(item)
                for item in payload.get("required_deliverables", [])
                if isinstance(item, dict)
            ],
            analysis_intents=_string_list(payload.get("analysis_intents", [])),
            domain_hints=_string_list(payload.get("domain_hints", [])),
            coverage_requirements=[
                CoverageRequirement.from_dict(item)
                for item in payload.get("coverage_requirements", [])
                if isinstance(item, dict)
            ],
            coverage_diagnostics=[
                dict(item)
                for item in payload.get("coverage_diagnostics", [])
                if isinstance(item, dict)
            ],
        )


INPUT_ARTIFACT_FORMATS = {
    "csv",
    "tsv",
    "xlsx",
    "xls",
    "json",
    "pdf",
    "docx",
    "pptx",
    "md",
    "png",
    "jpg",
    "jpeg",
    "html",
    "txt",
    "pkl",
    "mp4",
}
DELIVERABLE_PATH_FORMATS = {"docx", "pptx", "xlsx", "pdf", "md", "png", "json", "html", "txt", "pkl", "mp4"}


def analyze_task(query: str) -> TaskUnderstanding:
    """Extract deterministic routing constraints from a task string."""

    input_artifacts = _input_artifacts(query)
    required_deliverables = _required_deliverables(query)
    analysis_intents = _analysis_intents(query)
    domain_hints = _domain_hints(query)
    coverage_requirements = _coverage_requirements(required_deliverables, analysis_intents, query)
    return TaskUnderstanding(
        query=query,
        input_artifacts=input_artifacts,
        required_deliverables=required_deliverables,
        analysis_intents=analysis_intents,
        domain_hints=domain_hints,
        coverage_requirements=coverage_requirements,
    )


def filter_task_understanding_skills(
    understanding: TaskUnderstanding,
    available_skill_ids: set[str],
) -> TaskUnderstanding:
    """Return task understanding whose coverage skills exist in the active registry."""

    filtered_requirements: list[CoverageRequirement] = []
    for requirement in understanding.coverage_requirements:
        filtered_requirements.append(
            CoverageRequirement(
                id=requirement.id,
                kind=requirement.kind,
                label=requirement.label,
                acceptable_skill_ids=[
                    skill_id
                    for skill_id in requirement.acceptable_skill_ids
                    if skill_id in available_skill_ids
                ],
                preferred_skill_ids=[
                    skill_id
                    for skill_id in requirement.preferred_skill_ids
                    if skill_id in available_skill_ids
                ],
                format=requirement.format,
                minimum_count=requirement.minimum_count,
                source_text=requirement.source_text,
                requires_preferred=requirement.requires_preferred,
            )
        )
    return TaskUnderstanding(
        query=understanding.query,
        input_artifacts=list(understanding.input_artifacts),
        required_deliverables=list(understanding.required_deliverables),
        analysis_intents=list(understanding.analysis_intents),
        domain_hints=list(understanding.domain_hints),
        coverage_requirements=filtered_requirements,
        coverage_diagnostics=list(understanding.coverage_diagnostics),
    )


def coverage_diagnostics(
    understanding: TaskUnderstanding,
    selected_skill_ids: set[str],
) -> dict[str, Any]:
    """Return which task coverage requirements are covered by selected skills."""

    covered: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    auxiliary: list[dict[str, Any]] = []
    for requirement in understanding.coverage_requirements:
        matching: list[str] = []
        seen_matching: set[str] = set()
        for skill_id in [*requirement.preferred_skill_ids, *requirement.acceptable_skill_ids]:
            if skill_id in seen_matching:
                continue
            if skill_id in selected_skill_ids and skill_satisfies_requirement(skill_id, requirement):
                matching.append(skill_id)
                seen_matching.add(skill_id)
        row = requirement.to_dict()
        if matching:
            covered.append({**row, "covered_by": matching})
        elif _is_auxiliary_text_deliverable(requirement):
            auxiliary.append({**row, "coverage_optional": True})
        else:
            missing.append(row)
    return {
        "requirements": [item.to_dict() for item in understanding.coverage_requirements],
        "covered": covered,
        "missing": missing,
        "auxiliary": auxiliary,
    }


def hard_include_skill_ids(
    understanding: TaskUnderstanding,
    available_skill_ids: set[str],
    *,
    max_per_requirement: int = 3,
) -> dict[str, list[str]]:
    """Return top coverage skills that should bypass fuzzy retrieval."""

    output: dict[str, list[str]] = {}
    limit = max(max_per_requirement, 0)
    if limit == 0:
        return output
    for requirement in understanding.coverage_requirements:
        if _is_auxiliary_text_deliverable(requirement):
            continue
        preferred_available = [
            skill_id for skill_id in requirement.preferred_skill_ids if skill_id in available_skill_ids
        ]
        acceptable_available = [
            skill_id for skill_id in requirement.acceptable_skill_ids if skill_id in available_skill_ids
        ]
        chosen = preferred_available[:limit] or acceptable_available[:limit]
        for skill_id in chosen:
            output.setdefault(skill_id, []).append(requirement.id)
    return output


def _is_auxiliary_text_deliverable(requirement: CoverageRequirement) -> bool:
    return requirement.kind == "deliverable" and requirement.format in {"json", "md", "txt"}


def skill_satisfies_requirement(skill_id: str, requirement: CoverageRequirement) -> bool:
    """Return true if a skill can satisfy the coverage requirement."""

    if requirement.requires_preferred and requirement.preferred_skill_ids:
        return skill_id in requirement.preferred_skill_ids
    return skill_id in requirement.acceptable_skill_ids


def _input_artifacts(query: str) -> list[TaskArtifact]:
    artifacts: list[TaskArtifact] = []
    seen: set[str] = set()
    for match in _path_matches(query):
        path = match.group(0).rstrip(".,);:")
        fmt = _format_from_path(path)
        if fmt in INPUT_ARTIFACT_FORMATS and _looks_like_input(query, match.start()):
            key = path.lower()
            if key not in seen:
                seen.add(key)
                artifacts.append(TaskArtifact(path=path, format=fmt, source_text=path))
    return artifacts


def _required_deliverables(query: str) -> list[RequiredDeliverable]:
    deliverables: dict[str, RequiredDeliverable] = {}
    lower = query.lower()
    for match in _path_matches(query):
        path = match.group(0).rstrip(".,);:")
        fmt = _format_from_path(path)
        if fmt in DELIVERABLE_PATH_FORMATS and _looks_like_deliverable_path(query, match.start()):
            _add_deliverable(deliverables, fmt, path=path, source_text=path)
    if re.search(r"\b(slides?|slide deck|presentation|powerpoint)\b", lower) and "pptx" not in deliverables:
        _add_deliverable(deliverables, "pptx", label="presentation deck", source_text="presentation/slides")
    if re.search(r"\b(word document|docx report|report\.docx)\b", lower) and "docx" not in deliverables:
        _add_deliverable(deliverables, "docx", label="Word report", source_text="report/docx")
    if re.search(r"\b(markdown report|\.md report|readme\.md)\b", lower) and "md" not in deliverables:
        _add_deliverable(deliverables, "md", label="Markdown document", source_text="markdown/md")
    if re.search(r"\b(png|\.png|figures?|charts?|plots?)\b", lower) and _is_output_context(lower):
        count = _minimum_count_for_format(lower, "png")
        explicit_figure_context = bool(re.search(r"\b(figures?|charts?|plots?|visuali[sz]ations?)\b", lower))
        if "png" not in deliverables or explicit_figure_context:
            _add_deliverable(
                deliverables,
                "png",
                label="PNG figures" if explicit_figure_context else "PNG images",
                minimum_count=max(count, 1),
                source_text="png figures/charts" if explicit_figure_context else "png image outputs",
            )
    return sorted(deliverables.values(), key=lambda item: item.format)


def _analysis_intents(query: str) -> list[str]:
    lower = query.lower()
    intents: list[str] = []
    if re.search(r"\b(statistical|statistics|descriptive statistics|comparison tests?|anova|t-test|regression)\b", lower):
        intents.append("statistical_analysis")
    if re.search(r"\b(csv|spreadsheet|tabular data|data table|dataframe|dataset)\b", lower):
        intents.append("tabular_data_analysis")
    if re.search(r"\b(data story|conference|slides?|presentation|narrative)\b", lower):
        intents.append("data_storytelling")
    if re.search(
        r"\b(financial statements?|financial kpis?|kpis?|year-over-year|analyst summary|executive summary report)\b",
        lower,
    ):
        intents.append("financial_statement_analysis")
    if re.search(
        r"\b(symbolic|exact|jacobian|eigenvalues?|eigenvectors?|equilibria|equilibrium|stability analysis|"
        r"differential equations?|fourier coefficients?|partial sums?|integrals?|derivatives?|matrix|matrices)\b",
        lower,
    ):
        intents.append("symbolic_math")
    if re.search(
        r"\b(browse|visit|navigate|website|websites|web page|webpage|homepage|screenshot|screenshots|"
        r"scrape|crawl|extract from|official website|hacker news|books\.toscrape)\b",
        lower,
    ):
        intents.append("browser_interaction")
    if re.search(
        r"\b(generate|create|synthesize|illustrate|draw|render)\b[^.。\n]*(?:images?|illustrations?|artwork|poster|meme|banner|thumbnail|thumbs?)",
        lower,
    ) or re.search(r"\b(images?|illustrations?|artwork|meme|poster)\b[^.。\n]*\b(generate|create|synthesize|draw)\b", lower):
        intents.append("image_generation")
    if re.search(
        r"\b(thumbnail|thumbs?|resize|resized|preview|grid|composite|combine|combined|banner|collage|"
        r"format conversion|compress|crop)\b",
        lower,
    ):
        intents.append("image_processing")
    if re.search(r"\b(docx|word document|tracked changes|signature lines?|heading 1|heading 2)\b", lower):
        intents.append("document_creation")
    if re.search(r"\b(remotion|react video|src/ directory|1080p)\b", lower):
        intents.append("react_video")
    if re.search(r"\b(scroll|parallax|scrolling|scroll-driven|scroll triggered|progress indicator)\b", lower):
        intents.append("scroll_web_experience")
    return sorted(set(intents))


def _domain_hints(query: str) -> list[str]:
    lower = query.lower()
    hints: list[str] = []
    for name in ("financial", "clinical", "zoologist", "penguin", "research"):
        if name in lower:
            hints.append(name)
    return hints


def _coverage_requirements(
    deliverables: list[RequiredDeliverable],
    intents: list[str],
    query: str,
) -> list[CoverageRequirement]:
    requirements: list[CoverageRequirement] = []
    for deliverable in deliverables:
        requirements.append(
            CoverageRequirement(
                id=deliverable.id,
                kind="deliverable",
                label=deliverable.label,
                format=deliverable.format,
                minimum_count=deliverable.minimum_count,
                source_text=deliverable.source_text,
            )
        )
    if {"statistical_analysis", "tabular_data_analysis"} & set(intents):
        requirements.append(
            CoverageRequirement(
                id="intent:tabular_or_statistical_analysis",
                kind="intent",
                label="tabular or statistical analysis",
                source_text=_intent_source_text(query),
                requires_preferred=True,
            )
        )
    if "data_storytelling" in intents:
        requirements.append(
            CoverageRequirement(
                id="intent:data_storytelling",
                kind="intent",
                label="data storytelling",
                source_text="presentation/report narrative",
            )
        )
    if "financial_statement_analysis" in intents:
        requirements.append(
            CoverageRequirement(
                id="intent:financial_statement_analysis",
                kind="intent",
                label="financial statement or KPI analysis",
                source_text=_financial_source_text(query),
            )
        )
    if "symbolic_math" in intents:
        requirements.append(
            CoverageRequirement(
                id="intent:symbolic_math",
                kind="intent",
                label="symbolic mathematics",
                source_text=_symbolic_math_source_text(query),
                requires_preferred=True,
            )
        )
    if "browser_interaction" in intents:
        requirements.append(
            CoverageRequirement(
                id="intent:browser_interaction",
                kind="intent",
                label="browser interaction or web data acquisition",
                source_text=_browser_source_text(query),
                requires_preferred=True,
            )
        )
    if "image_generation" in intents:
        requirements.append(
            CoverageRequirement(
                id="intent:image_generation",
                kind="intent",
                label="image generation",
                source_text=_image_generation_source_text(query),
                requires_preferred=True,
            )
        )
    if "image_processing" in intents:
        requirements.append(
            CoverageRequirement(
                id="intent:image_processing",
                kind="intent",
                label="image processing or compositing",
                source_text=_image_processing_source_text(query),
                requires_preferred=True,
            )
        )
    if "document_creation" in intents:
        requirements.append(
            CoverageRequirement(
                id="intent:document_creation",
                kind="intent",
                label="document creation or editing",
                source_text=_document_source_text(query),
                requires_preferred=True,
            )
        )
    if "react_video" in intents:
        requirements.append(
            CoverageRequirement(
                id="intent:react_video",
                kind="intent",
                label="React or Remotion video creation",
                source_text=_react_video_source_text(query),
                requires_preferred=True,
            )
        )
    if "scroll_web_experience" in intents:
        requirements.append(
            CoverageRequirement(
                id="intent:scroll_web_experience",
                kind="intent",
                label="scroll-driven web experience",
                source_text=_scroll_source_text(query),
                requires_preferred=True,
            )
        )
    return _dedupe_requirements(requirements)


def _add_deliverable(
    deliverables: dict[str, RequiredDeliverable],
    fmt: str,
    *,
    label: str = "",
    path: str = "",
    minimum_count: int = 1,
    source_text: str = "",
) -> None:
    existing = deliverables.get(fmt)
    label = label or _label_for_format(fmt)
    if existing is None:
        deliverables[fmt] = RequiredDeliverable(
            format=fmt,
            label=label,
            path=path,
            minimum_count=minimum_count,
            source_text=source_text,
        )
        return
    if path and not existing.path:
        existing.path = path
    existing.minimum_count = max(existing.minimum_count, minimum_count)
    if source_text and source_text not in existing.source_text:
        existing.source_text = " | ".join(item for item in [existing.source_text, source_text] if item)


def _path_matches(query: str) -> list[re.Match[str]]:
    return list(
        re.finditer(
            r"(?<![A-Za-z0-9])[\w./-]+\.(?:csv|tsv|xlsx|xls|docx|pptx|pdf|md|png|jpe?g|json|html|txt|pkl|mp4)\b",
            query,
            re.I,
        )
    )


def _format_from_path(path: str) -> str:
    return path.rsplit(".", 1)[-1].lower()


def _looks_like_deliverable_path(query: str, start: int) -> bool:
    return _looks_like_output(query, start) or not _looks_like_input(query, start)


def _looks_like_input(query: str, start: int) -> bool:
    if _looks_like_output(query, start):
        return False
    window = _local_context_before_path(query, start)
    return bool(
        re.search(
            r"\b(path|stored at|source|input|from|using|use|file at|located at|read|load|data at|dataset at|csv at)\b",
            window,
        )
        or re.search(
            r"\b(analy[sz]e|inspect|process|parse|convert|summari[sz]e)\s+$",
            window,
        )
    )


def _looks_like_output(query: str, start: int) -> bool:
    window = _local_context_before_path(query, start)
    return bool(
        re.search(
            r"\b(create|produce|generate|write|save|export|render|build|make|deliver|output|return)\b",
            window,
        )
    )


def _local_context_before_path(query: str, start: int) -> str:
    prefix = query[max(0, start - 120): start]
    boundary = 0
    for match in re.finditer(r"[,.;:()\n。！？]|(?:\b(?:and|then)\b)", prefix, re.I):
        boundary = match.end()
    return prefix[boundary:].lower()


def _is_output_context(lower_query: str) -> bool:
    return bool(re.search(r"\b(generate|create|produce|save|write|deliver|output)\b", lower_query))


def _minimum_count_for_format(lower_query: str, fmt: str) -> int:
    patterns = [
        rf"at least\s+(\d+)[^.。\n]*\b{fmt}\b",
        rf"(\d+)[^.。\n]*\b{fmt}\b",
        r"at least\s+(\d+)[^.。\n]*\b(figures?|charts?|plots?)\b",
        r"(\d+)[^.。\n]*\b(figures?|charts?|plots?)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, lower_query)
        if match:
            return _safe_int(match.group(1), 1)
    return 1


def _label_for_format(fmt: str) -> str:
    return {
        "pptx": "PowerPoint presentation",
        "docx": "Word document",
        "xlsx": "Excel workbook",
        "pdf": "PDF document",
        "md": "Markdown document",
        "png": "PNG figures",
        "json": "JSON artifact",
        "html": "HTML page",
        "txt": "text file",
        "pkl": "pickle artifact",
        "mp4": "video artifact",
    }.get(fmt, fmt)


def _intent_source_text(query: str) -> str:
    lower = query.lower()
    match = re.search(r"(statistical[^.。\n]*|descriptive statistics[^.。\n]*|csv[^.。\n]*)", lower)
    return match.group(0) if match else "statistical/tabular analysis"


def _financial_source_text(query: str) -> str:
    lower = query.lower()
    match = re.search(r"(financial[^.。\n]*|kpi[^.。\n]*|year-over-year[^.。\n]*)", lower)
    return match.group(0) if match else "financial/KPI analysis"


def _symbolic_math_source_text(query: str) -> str:
    lower = query.lower()
    match = re.search(
        r"(symbolic[^.。\n]*|jacobian[^.。\n]*|eigenvalues?[^.。\n]*|stability analysis[^.。\n]*|"
        r"fourier[^.。\n]*|integrals?[^.。\n]*|derivatives?[^.。\n]*)",
        lower,
    )
    return match.group(0) if match else "symbolic mathematics"


def _browser_source_text(query: str) -> str:
    lower = query.lower()
    match = re.search(
        r"(browse[^.。\n]*|visit[^.。\n]*|screenshots?[^.。\n]*|scrape[^.。\n]*|crawl[^.。\n]*|website[^.。\n]*)",
        lower,
    )
    return match.group(0) if match else "browser interaction"


def _image_generation_source_text(query: str) -> str:
    lower = query.lower()
    match = re.search(
        r"(generate[^.。\n]*(?:images?|illustrations?|artwork|meme|poster)|"
        r"create[^.。\n]*(?:images?|illustrations?|artwork|meme|poster))",
        lower,
    )
    return match.group(0) if match else "image generation"


def _image_processing_source_text(query: str) -> str:
    lower = query.lower()
    match = re.search(r"(thumbnail[^.。\n]*|combine[^.。\n]*|composite[^.。\n]*|banner[^.。\n]*|grid[^.。\n]*)", lower)
    return match.group(0) if match else "image processing"


def _document_source_text(query: str) -> str:
    lower = query.lower()
    match = re.search(r"(docx[^.。\n]*|word document[^.。\n]*|tracked changes?[^.。\n]*|heading [12][^.。\n]*)", lower)
    return match.group(0) if match else "document creation"


def _react_video_source_text(query: str) -> str:
    lower = query.lower()
    match = re.search(r"(remotion[^.。\n]*|react video[^.。\n]*|src/[^.。\n]*|1080p[^.。\n]*)", lower)
    return match.group(0) if match else "React video creation"


def _scroll_source_text(query: str) -> str:
    lower = query.lower()
    match = re.search(r"(scroll[^.。\n]*|parallax[^.。\n]*|progress indicator[^.。\n]*)", lower)
    return match.group(0) if match else "scroll web experience"


def _dedupe_requirements(requirements: list[CoverageRequirement]) -> list[CoverageRequirement]:
    output: dict[str, CoverageRequirement] = {}
    for requirement in requirements:
        existing = output.get(requirement.id)
        if existing is None:
            output[requirement.id] = requirement
            continue
        existing.minimum_count = max(existing.minimum_count, requirement.minimum_count)
        existing.acceptable_skill_ids = _dedupe_strings([*existing.acceptable_skill_ids, *requirement.acceptable_skill_ids])
        existing.preferred_skill_ids = _dedupe_strings([*existing.preferred_skill_ids, *requirement.preferred_skill_ids])
    return [output[key] for key in sorted(output)]


def _dedupe_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in values if item))


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
