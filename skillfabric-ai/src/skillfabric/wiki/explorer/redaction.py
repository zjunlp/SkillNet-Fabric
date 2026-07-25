"""Shared redaction for explorer error and lifecycle artifacts."""

from __future__ import annotations

import re
from pathlib import Path

_NAMED_SECRET = re.compile(
    r"\b([A-Za-z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)[A-Za-z0-9_]*)"
    r"\s*(?:=|:|\s)\s*([^\s,;]+)",
    re.IGNORECASE,
)


def sanitize_error_text(
    value: str,
    *,
    paths: tuple[Path, ...] = (),
    path_replacement: str,
    limit: int = 2_000,
    collapse_whitespace: bool = False,
) -> str:
    """Redact common credential forms and private runtime paths."""

    text = re.sub(r"(?i)\bsk-[a-z0-9._-]+", "[redacted]", value)
    text = _NAMED_SECRET.sub(lambda match: f"{match.group(1)}=[redacted]", text)
    text = re.sub(r"(?i)\bBearer\s+\S+", "Bearer [redacted]", text)
    for path in sorted((str(path) for path in paths), key=len, reverse=True):
        if path:
            text = text.replace(path, path_replacement)
    if collapse_whitespace:
        text = " ".join(text.split())
    return text[:limit]


__all__ = ["sanitize_error_text"]
