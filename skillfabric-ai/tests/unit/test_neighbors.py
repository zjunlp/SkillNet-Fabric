from __future__ import annotations

import math
import unittest

from skillfabric.indexing import neighbors
from skillfabric.indexing.neighbors import lexical_overlap
from skillfabric.registry.models import SkillNode


class NeighborScoringTests(unittest.TestCase):
    def test_token_overlap_helper_is_named_for_cosine_formula(self) -> None:
        self.assertFalse(hasattr(neighbors, "_jaccard"))
        self.assertTrue(hasattr(neighbors, "_token_cosine_overlap"))

    def test_lexical_overlap_uses_cosine_overlap_not_jaccard(self) -> None:
        left = _skill("left", "Alpha Beta")
        right = _skill("right", "Alpha Beta Gamma Delta")

        overlap = lexical_overlap(left, right)

        self.assertAlmostEqual(overlap, 2 / math.sqrt(2 * 4))
        self.assertNotEqual(overlap, 2 / 4)


def _skill(skill_id: str, description: str) -> SkillNode:
    return SkillNode(
        id=skill_id,
        type="skill",
        name="",
        description=description,
        content_hash=f"hash-{skill_id}",
    )


if __name__ == "__main__":
    unittest.main()
