from __future__ import annotations

import json

import pytest

from skillfabric.runtime.json_utils import parse_json_response


def test_parse_json_response_accepts_exact_json_object() -> None:
    assert parse_json_response('{"status": "ok"}') == {"status": "ok"}


@pytest.mark.parametrize(
    "response",
    [
        '```json\n{"status": "ok"}\n```',
        'Result: {"status": "ok"}',
        '{"status": "ok"}\nDone.',
    ],
)
def test_parse_json_response_rejects_wrapped_or_fenced_json(response) -> None:
    with pytest.raises(json.JSONDecodeError):
        parse_json_response(response)


def test_parse_json_response_rejects_non_object_json() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        parse_json_response("[]")
