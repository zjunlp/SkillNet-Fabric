"""Shared artifact and deliverable format ontology."""

from __future__ import annotations


def canonical_artifact_name(value: str) -> str:
    """Return a stable artifact family name for common file/deliverable aliases."""

    normalized = normalize_artifact_phrase(value)
    if not normalized:
        return ""
    tokens = normalized.split()
    token_set = set(tokens)
    if "docx" in token_set or "word" in token_set:
        return "docx_document"
    if "pptx" in token_set or "powerpoint" in token_set or "slides" in token_set:
        return "presentation_document"
    if "pdf" in token_set:
        return "pdf_document"
    if "csv" in token_set or "tsv" in token_set:
        return "csv_table"
    if "json" in token_set:
        return "json_data"
    if "markdown" in token_set or "md" in token_set:
        return "markdown_document"
    if "prompt" in token_set:
        return "prompt_text"
    if "image" in token_set or "png" in token_set or "jpeg" in token_set or "jpg" in token_set:
        return "image_asset"
    if "video" in token_set or "mp4" in token_set or "webm" in token_set:
        return "video_asset"
    if "html" in token_set:
        return "html_document"
    if token_set & {"excel", "xlsx", "xls", "workbook", "spreadsheet", "worksheet", "tabular"}:
        return "spreadsheet_table"
    if "table" in token_set:
        return "table_data"
    return "_".join(tokens)


def normalize_artifact_phrase(value: str) -> str:
    """Normalize separators and case for artifact ontology lookups."""

    return " ".join(value.lower().replace("_", " ").replace("-", " ").split())
