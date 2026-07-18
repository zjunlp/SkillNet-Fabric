"""Shared non-secret workflow metrics helpers."""

from __future__ import annotations

from skillfabric.storage import Workspace
from skillfabric.wiki.models import WikiBuildResult


def merge_wiki_metrics(workspace: Workspace, wiki_result: WikiBuildResult) -> None:
    """Merge deterministic Wiki metrics into reports/build_summary.json."""

    metrics_path = workspace.reports_dir / "build_summary.json"
    payload = workspace.read_json(metrics_path, default={}) or {}
    if not isinstance(payload, dict):
        payload = {}
    payload["wiki"] = {
        "pages_written": wiki_result.pages_written,
    }
    workspace.write_json(metrics_path, payload)
