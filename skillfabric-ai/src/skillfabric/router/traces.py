"""Route trace helpers."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime

_TRACE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def _new_trace_id(query: str) -> str:
    """Return a deterministic-ish trace id prefix plus query digest."""

    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:10]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"route-{stamp}-{digest}"


def validate_trace_id(trace_id: str) -> str:
    """Return a safe single-path-component trace id."""

    value = str(trace_id).strip()
    if not _TRACE_ID_PATTERN.fullmatch(value):
        raise ValueError(
            "invalid trace_id: use 1-128 ASCII letters, digits, dots, underscores, or hyphens; "
            "the first character must be alphanumeric"
        )
    return value
