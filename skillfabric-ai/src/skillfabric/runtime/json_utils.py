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
        if response.get("content") is not None:
            content_text = _content_to_text(response["content"])
            if content_text:
                return content_text
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            parts: list[str] = []
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                message = choice.get("message")
                if isinstance(message, dict) and message.get("content") is not None:
                    parts.append(_content_to_text(message["content"]))
                elif choice.get("text") is not None:
                    parts.append(str(choice["text"]))
            if any(parts):
                return "\n".join(part for part in parts if part)
        if response.get("output_text") is not None:
            return str(response["output_text"])
        output = response.get("output")
        if isinstance(output, list):
            parts = [
                _content_to_text(item.get("content"))
                for item in output
                if isinstance(item, dict) and item.get("content") is not None
            ]
            if any(parts):
                return "\n".join(part for part in parts if part)
    return str(response)


def parse_json_response(response: Any) -> dict[str, Any]:
    """Parse a JSON object from a strict or fenced LLM response."""

    text = extract_response_text(response).strip()
    text = _strip_fence(text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        payload = _first_json_object(text, original_error=exc)
    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object")
    return payload


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [_content_to_text(item) for item in content]
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        block_type = str(content.get("type", "")).casefold()
        if block_type and block_type not in {"text", "output_text"}:
            return ""
        if content.get("text") is not None:
            return str(content["text"])
        return ""
    block_type = str(getattr(content, "type", "")).casefold()
    if block_type and block_type not in {"text", "output_text"}:
        return ""
    text = getattr(content, "text", None)
    if text is not None:
        return str(text)
    return str(content)


def _first_json_object(text: str, *, original_error: json.JSONDecodeError) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise original_error


def _strip_fence(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text
