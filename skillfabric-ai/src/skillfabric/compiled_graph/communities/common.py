"""Shared helpers for community assignment modules."""

from __future__ import annotations

from typing import Any


def _string_list(payload: Any) -> list[str]:
    if not isinstance(payload, list):
        return []
    return [str(item).strip() for item in payload if str(item).strip()]


def _members_by_community(membership: dict[str, str]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for skill_id, community_id in membership.items():
        output.setdefault(community_id, []).append(skill_id)
    return {key: sorted(values) for key, values in output.items()}
