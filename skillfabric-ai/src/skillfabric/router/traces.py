"""Route trace helpers."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4

_TRACE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def _new_trace_id(query: str) -> str:
    """Return a deterministic-ish trace id prefix plus query digest."""

    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:10]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"route-{stamp}-{digest}-{uuid4().hex[:8]}"


def validate_trace_id(trace_id: str) -> str:
    """Validate a trace id as one bounded ASCII path component."""

    if not isinstance(trace_id, str) or _TRACE_ID_PATTERN.fullmatch(trace_id) is None:
        raise ValueError(
            "invalid trace_id: use 1-128 ASCII letters, digits, dots, underscores, or hyphens; "
            "the first character must be alphanumeric"
        )
    return trace_id


def _create_trace_dir(runs_dir: Path, trace_id: str) -> Path:
    path = runs_dir / validate_trace_id(trace_id)
    try:
        path.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(f"route trace already exists: {path}") from exc
    return path
