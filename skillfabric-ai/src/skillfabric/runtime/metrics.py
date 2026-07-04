"""Shared non-secret workflow metrics helpers."""

from __future__ import annotations

from skillfabric.runtime.usage import load_usage_records, summarize_usage
from skillfabric.storage import Workspace
from skillfabric.wiki.models import WikiBuildResult


def merge_wiki_metrics(workspace: Workspace, wiki_result: WikiBuildResult) -> None:
    """Merge wiki summary counters into reports/build_summary.json."""

    metrics_path = workspace.reports_dir / "build_summary.json"
    payload = workspace.read_json(metrics_path, default={}) or {}
    if not isinstance(payload, dict):
        payload = {}
    payload["wiki_summary"] = {
        "llm_calls": wiki_result.llm_calls,
        "fallback_count": wiki_result.fallback_count,
        "cache_hits": wiki_result.cache_hits,
        "pages_written": wiki_result.pages_written,
    }
    usage_path = workspace.reports_dir / "llm_usage.jsonl"
    payload["llm_usage"] = summarize_usage(load_usage_records(usage_path)).to_dict() if usage_path.exists() else {}
    workspace.write_json(metrics_path, payload)
