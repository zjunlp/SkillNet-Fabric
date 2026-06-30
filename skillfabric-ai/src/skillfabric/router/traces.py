"""Route trace helpers."""

from __future__ import annotations

import hashlib
from datetime import datetime


def _new_trace_id(query: str) -> str:
    """Return a deterministic-ish trace id prefix plus query digest."""

    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:10]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"route-{stamp}-{digest}"
