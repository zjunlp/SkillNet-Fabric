from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from skillfabric.compiled_graph.communities.clustering import (
    _prepare_native_library_cache_dirs,
    cluster_communities,
)
from skillfabric.compiled_graph.models import Edge
from tests.unit.relation_helpers import make_skill


class CommunityClusteringTests(unittest.TestCase):
    def test_dense_similarity_groups_become_separate_communities(self) -> None:
        skills = [make_skill(f"skill:a-{index}", f"a-{index}", "Alpha.") for index in range(4)]
        skills += [make_skill(f"skill:b-{index}", f"b-{index}", "Beta.") for index in range(4)]
        edges = _clique_edges([skill.id for skill in skills[:4]]) + _clique_edges(
            [skill.id for skill in skills[4:]]
        )

        _communities, _member_edges, membership, stats = cluster_communities(skills, edges, [])

        self.assertEqual(_partition(membership, [skill.id for skill in skills[:4]]), 1)
        self.assertEqual(_partition(membership, [skill.id for skill in skills[4:]]), 1)
        self.assertNotEqual(membership["skill:a-0"], membership["skill:b-0"])
        self.assertEqual(stats["community_clustering_algorithm"], "leiden")

    def test_single_compose_edge_does_not_merge_dense_similarity_groups(self) -> None:
        left = [f"skill:left-{index}" for index in range(5)]
        right = [f"skill:right-{index}" for index in range(5)]
        skills = [make_skill(skill_id, skill_id.removeprefix("skill:"), "Skill.") for skill_id in left + right]
        similar_edges = _clique_edges(left) + _clique_edges(right)
        relation_edges = [
            Edge(left[0], right[0], "compose_with", confidence=0.98, weight=0.98)
        ]

        _communities, _member_edges, membership, _stats = cluster_communities(
            skills,
            similar_edges,
            relation_edges,
        )

        self.assertNotEqual(membership[left[0]], membership[right[0]])

    def test_depend_on_edges_do_not_drive_membership(self) -> None:
        skill_ids = [f"skill:step-{index}" for index in range(4)]
        skills = [make_skill(skill_id, skill_id.removeprefix("skill:"), "Step.") for skill_id in skill_ids]
        relation_edges = [
            Edge(skill_ids[index + 1], skill_ids[index], "depend_on", confidence=0.99, weight=0.99)
            for index in range(len(skill_ids) - 1)
        ]

        communities, _member_edges, membership, stats = cluster_communities(skills, [], relation_edges)

        self.assertEqual(len(communities), len(skills))
        self.assertEqual(len(set(membership.values())), len(skills))
        self.assertEqual(stats["community_projection_depend_on_ignored_count"], len(relation_edges))

    def test_isolates_become_singletons(self) -> None:
        skills = [
            make_skill("skill:alpha", "alpha", "Alpha."),
            make_skill("skill:beta", "beta", "Beta."),
        ]

        communities, _member_edges, membership, _stats = cluster_communities(skills, [], [])

        self.assertEqual(len(communities), 2)
        self.assertNotEqual(membership["skill:alpha"], membership["skill:beta"])

    def test_oversized_community_is_split_to_health_shape(self) -> None:
        skills = [
            make_skill(f"skill:item-{index:02d}", f"item-{index:02d}", "Skill.")
            for index in range(40)
        ]
        edges = _clique_edges([skill.id for skill in skills])

        communities, _member_edges, membership, stats = cluster_communities(skills, edges, [])

        self.assertGreaterEqual(len(communities), 5)
        self.assertLessEqual(max(community.member_count for community in communities), 12)
        self.assertEqual(set(membership), {skill.id for skill in skills})
        self.assertGreater(stats["community_oversize_split_count"], 0)

    def test_large_community_uses_preferred_member_count_cap(self) -> None:
        skills = [
            make_skill(f"skill:item-{index:03d}", f"item-{index:03d}", "Skill.")
            for index in range(100)
        ]
        edges = _clique_edges([skill.id for skill in skills])

        communities, _member_edges, membership, stats = cluster_communities(skills, edges, [])

        self.assertLessEqual(max(community.member_count for community in communities), 12)
        self.assertEqual(set(membership), {skill.id for skill in skills})
        self.assertGreater(stats["community_oversize_split_count"], 0)

    def test_leiden_partition_uses_fixed_random_seed(self) -> None:
        skills = [
            make_skill("skill:alpha", "alpha", "Alpha."),
            make_skill("skill:beta", "beta", "Beta."),
        ]
        edges = [Edge("skill:alpha", "skill:beta", "similar_to", confidence=1.0, weight=1.0)]
        seeds: list[int | None] = []

        def fake_leiden(graph, *, weight_attribute="weight", random_seed=None):
            seeds.append(random_seed)
            return {node: 0 for node in graph.nodes}

        with patch("graspologic.partition.leiden", new=fake_leiden):
            cluster_communities(skills, edges, [])

        self.assertEqual(seeds, [42])

    def test_native_library_cache_dirs_default_to_workspace_tmp_when_unset(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SKILLFABRIC_NATIVE_CACHE_DIR": "",
                "NUMBA_CACHE_DIR": "",
                "MPLCONFIGDIR": "",
            },
            clear=False,
        ):
            expected_root = Path(__file__).resolve().parents[4] / "tmp" / "skillfabric-native-cache"
            os.environ.pop("SKILLFABRIC_NATIVE_CACHE_DIR", None)
            os.environ.pop("NUMBA_CACHE_DIR", None)
            os.environ.pop("MPLCONFIGDIR", None)

            _prepare_native_library_cache_dirs()

            self.assertEqual(
                os.environ["NUMBA_CACHE_DIR"],
                str(expected_root / "numba"),
            )
            self.assertEqual(
                os.environ["MPLCONFIGDIR"],
                str(expected_root / "matplotlib"),
            )

    def test_leiden_unavailable_raises_without_louvain_fallback(self) -> None:
        skills = [
            make_skill("skill:alpha", "alpha", "Alpha."),
            make_skill("skill:beta", "beta", "Beta."),
        ]
        edges = [Edge("skill:alpha", "skill:beta", "similar_to", confidence=1.0, weight=1.0)]

        def broken_leiden(*_args, **_kwargs):
            raise RuntimeError("blocked for test")

        with patch("graspologic.partition.leiden", new=broken_leiden):
            with self.assertRaisesRegex(RuntimeError, "Leiden community detection failed"):
                cluster_communities(skills, edges, [])


def _clique_edges(skill_ids: list[str]) -> list[Edge]:
    edges: list[Edge] = []
    for left_index, source in enumerate(skill_ids):
        for target in skill_ids[left_index + 1 :]:
            edges.append(Edge(source, target, "similar_to", confidence=1.0, weight=1.0))
    return edges


def _partition(membership: dict[str, str], skill_ids: list[str]) -> int:
    return len({membership[skill_id] for skill_id in skill_ids})


if __name__ == "__main__":
    unittest.main()
