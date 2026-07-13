"""Stable identity for trusted prompt policies and output schemas."""

from __future__ import annotations

import hashlib
import json
from html import escape
from typing import Any

UNTRUSTED_JSON_SERIALIZATION = "indented-json-in-xml-text"


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


def render_untrusted_json(payload: Any) -> str:
    """Serialize untrusted data without allowing it to close an XML text element."""

    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    return escape(serialized, quote=False)
