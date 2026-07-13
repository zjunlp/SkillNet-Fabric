from __future__ import annotations

import pytest

from skillfabric.runtime.prompting import prompt_fingerprint


def test_prompt_fingerprint_is_stable_for_mapping_order() -> None:
    first = prompt_fingerprint("skill_contract", {"b": 2, "a": 1})
    second = prompt_fingerprint("skill_contract", {"a": 1, "b": 2})

    assert first == second
    assert len(first) == 64


def test_prompt_fingerprint_changes_with_policy() -> None:
    assert prompt_fingerprint("judge", "strict") != prompt_fingerprint("judge", "complete")


def test_prompt_fingerprint_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="prompt_name must not be empty"):
        prompt_fingerprint("  ", "policy")
