"""Stable identity for trusted prompt policies and output schemas."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def prompt_fingerprint(prompt_name: str, *policy: Any) -> str:
    """Hash a stable prompt name and its trusted policy content."""

    if not prompt_name.strip():
        raise ValueError("prompt_name must not be empty")
    encoded = json.dumps(
        [prompt_name, *policy],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
