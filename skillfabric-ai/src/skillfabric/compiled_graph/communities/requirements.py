"""Assignment health requirements for community normalization."""

from __future__ import annotations

from typing import Any


def _assignment_health_requirements(skill_count: int) -> dict[str, Any]:
    if skill_count < 20:
        return {
            "minimum_communities": 1,
            "relaxed_minimum_communities": 1,
            "max_largest_fraction": 1.0,
            "balanced_near_minimum_max_largest_fraction": 1.0,
            "preferred_member_count_range": [1, max(1, skill_count)],
        }
    minimum_communities = max(4, int(skill_count ** 0.5) - 1)
    return {
        "minimum_communities": minimum_communities,
        "relaxed_minimum_communities": max(4, minimum_communities - 1),
        "max_largest_fraction": 0.30,
        "balanced_near_minimum_max_largest_fraction": 0.25,
        "preferred_member_count_range": [3, 12],
    }
