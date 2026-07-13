"""Utilities for JSON-only LLM responses."""

from __future__ import annotations

import json
from typing import Any


def extract_response_text(response: Any) -> str:
    """Return assistant text from a LiteLLM-style response or a test double."""

    if isinstance(response, str):
        return response
    if hasattr(response, "model_dump"):
        return extract_response_text(response.model_dump())
    if hasattr(response, "dict"):
        return extract_response_text(response.dict())
    if isinstance(response, dict):
        if "content" in response:
            return str(response["content"])
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict) and message.get("content") is not None:
                    return str(message["content"])
                if first.get("text") is not None:
                    return str(first["text"])
    return str(response)


def parse_json_response(response: Any) -> dict[str, Any]:
    """Parse one exact JSON object from an LLM response."""

    text = extract_response_text(response).strip()
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object")
    return payload
