"""Build page-level wiki indexes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from skillfabric.storage import Workspace, atomic_write_text
from skillfabric.wiki.explorer.models import WikiPageEntry


def build_wiki_page_index(workspace: Workspace) -> dict[str, int]:
    """Build wiki_page_index.jsonl from generated markdown pages."""

    pages = _collect_pages(workspace)
    _write_jsonl(workspace.wiki_dir / "wiki_page_index.jsonl", [page.to_dict() for page in pages])
    return {"page_count": len(pages)}


def load_page_index(workspace: Workspace | str | Path) -> list[WikiPageEntry]:
    """Load page-level wiki index."""

    workspace = workspace if isinstance(workspace, Workspace) else Workspace(workspace)
    path = workspace.wiki_dir / "wiki_page_index.jsonl"
    if not path.exists():
        return []
    return [
        WikiPageEntry.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _collect_pages(workspace: Workspace) -> list[WikiPageEntry]:
    pages: list[WikiPageEntry] = []
    for path in sorted(workspace.wiki_dir.rglob("*.md")):
        if _skip_page(workspace, path):
            continue
        text = path.read_text(encoding="utf-8")
        metadata, body = _split_frontmatter(text)
        rel_path = path.relative_to(workspace.wiki_dir).as_posix()
        page_type, entity_id = _page_identity(rel_path, metadata)
        title = _title(body) or str(metadata.get("name", "")) or path.stem
        summary = _summary(body)
        pages.append(
            WikiPageEntry(
                page_id=_stable_id(rel_path),
                path=rel_path,
                page_type=page_type,
                entity_id=entity_id,
                title=title,
                summary=summary,
                text=text,
            )
        )
    return pages


def _skip_page(workspace: Workspace, path: Path) -> bool:
    rel = path.relative_to(workspace.wiki_dir).as_posix()
    if rel.startswith("debug/"):
        return True
    if rel in {"hot.md", "log.md", "wiki_health_report.md"}:
        return True
    return False


def _page_identity(rel_path: str, metadata: dict[str, str]) -> tuple[str, str]:
    if rel_path.startswith("skills/"):
        return "skill", metadata.get("skill_id", f"skill:{Path(rel_path).stem}")
    if rel_path.startswith("communities/"):
        return "community", metadata.get("community_id", Path(rel_path).stem)
    if rel_path.startswith("workflows/"):
        return "workflow", metadata.get("workflow_id", Path(rel_path).stem)
    if rel_path == "index.md":
        return "index", "index"
    return metadata.get("type", "page"), Path(rel_path).stem


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    end_index = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_index = index
            break
    if end_index is None:
        return {}, text
    metadata: dict[str, str] = {}
    for line in lines[1:end_index]:
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        metadata[key.strip()] = value.strip().strip('"').strip("'")
    return metadata, "\n".join(lines[end_index + 1 :]).strip()


def _title(body: str) -> str:
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _summary(body: str) -> str:
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("- Source path:"):
            continue
        return stripped[:300]
    return ""


def _stable_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
