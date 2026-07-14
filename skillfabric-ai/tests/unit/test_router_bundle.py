from __future__ import annotations

from pathlib import Path

import pytest

import skillfabric.router.bundle as bundle_module
from skillfabric.compiled_graph.models import Edge
from skillfabric.indexing.ranking import reciprocal_rank_fusion
from skillfabric.router.bundle import RouterBundleConfig, build_router_bundle
from skillfabric.router.expansion import ExpansionResult, expand_semantic_candidates
from skillfabric.router.models import RouterAlternative, RouterBundle, RouterSkillCandidate
from skillfabric.wiki.loader import WikiSource
from tests.unit.fake_embeddings import FakeEmbeddingProvider
from tests.unit.relation_helpers import make_skill
from tests.unit.wiki_helpers import build_fixture_workspace


def _edge(source: str, target: str, edge_type: str, confidence: float = 0.9) -> Edge:
    return Edge(
        source=source,
        target=target,
        type=edge_type,
        confidence=confidence,
        reason=f"Validated {edge_type} relation.",
    )


def _seed(skill_id: str, rank: int = 1) -> RouterSkillCandidate:
    return RouterSkillCandidate(
        skill_id=skill_id,
        name=skill_id.removeprefix("skill:"),
        score=1.0 / (60 + rank),
        is_seed=True,
        retrieval_ranks={"bm25": rank},
    )


def test_router_bundle_config_has_only_bounded_expansion_controls() -> None:
    config = RouterBundleConfig()

    assert config.seed_limit == 24
    assert config.expanded_limit == 100
    assert config.max_depth == 2
    assert not hasattr(config, "graph_expansion_mode")
    assert not hasattr(config, "ppr_alpha")
    assert not hasattr(config, "workflow_confidence_threshold")


@pytest.mark.parametrize(
    "overrides",
    [
        {"seed_limit": True},
        {"seed_limit": -1},
        {"seed_limit": 2, "expanded_limit": 1},
        {"max_depth": -1},
    ],
)
def test_router_bundle_config_rejects_invalid_limits(overrides) -> None:
    with pytest.raises(ValueError):
        RouterBundleConfig(**overrides)


def test_reciprocal_rank_fusion_uses_channel_order_only() -> None:
    fused = reciprocal_rank_fusion(
        {
            "bm25": ["skill:a", "skill:b", "skill:c"],
            "embedding": ["skill:b", "skill:c", "skill:a"],
        },
        k=60,
    )

    assert [row.skill_id for row in fused] == ["skill:b", "skill:a", "skill:c"]
    assert fused[0].ranks == {"bm25": 2, "embedding": 1}
    assert fused[0].score == pytest.approx((1 / 62) + (1 / 61))


def test_bounded_expansion_keeps_every_seed_even_with_score_gap() -> None:
    skills = {
        skill.id: skill
        for skill in (
            make_skill("skill:strong", "strong", "Strong match."),
            make_skill("skill:weak", "weak", "Weak match."),
        )
    }
    seeds = [_seed("skill:strong", 1), _seed("skill:weak", 2)]

    result = expand_semantic_candidates(
        seeds,
        [],
        skills,
        max_depth=2,
        limit=2,
    )

    assert [candidate.skill_id for candidate in result.candidates] == [
        "skill:strong",
        "skill:weak",
    ]


def test_operational_expansion_records_exact_two_hop_path() -> None:
    skills = {
        skill.id: skill
        for skill in (
            make_skill("skill:goal", "goal", "Goal skill."),
            make_skill("skill:prerequisite", "prerequisite", "Prepare input."),
            make_skill("skill:helper", "helper", "Complement preparation."),
        )
    }
    edges = [
        _edge("skill:goal", "skill:prerequisite", "depend_on"),
        _edge("skill:helper", "skill:prerequisite", "compose_with"),
    ]

    result = expand_semantic_candidates(
        [_seed("skill:goal")],
        edges,
        skills,
        max_depth=2,
        limit=3,
    )
    by_id = {candidate.skill_id: candidate for candidate in result.candidates}

    assert by_id["skill:prerequisite"].graph_depth == 1
    assert by_id["skill:helper"].graph_depth == 2
    path = by_id["skill:helper"].introduced_by[0]
    assert path.seed_skill == "skill:goal"
    assert [(step.source, step.target, step.edge_type) for step in path.steps] == [
        ("skill:goal", "skill:prerequisite", "depend_on"),
        ("skill:prerequisite", "skill:helper", "compose_with"),
    ]


def test_expansion_uses_the_shortest_supported_path_across_seeds() -> None:
    skills = {
        skill.id: skill
        for skill in (
            make_skill("skill:z-high", "z-high", "High-ranked seed."),
            make_skill("skill:a-low", "a-low", "Lower-ranked seed."),
            make_skill("skill:middle", "middle", "Intermediate skill."),
            make_skill("skill:target", "target", "Target skill."),
        )
    }
    edges = [
        _edge("skill:z-high", "skill:middle", "compose_with"),
        _edge("skill:middle", "skill:target", "compose_with"),
        _edge("skill:a-low", "skill:target", "compose_with"),
    ]

    result = expand_semantic_candidates(
        [_seed("skill:z-high", 1), _seed("skill:a-low", 2)],
        edges,
        skills,
        max_depth=2,
        limit=4,
    )
    target = next(item for item in result.candidates if item.skill_id == "skill:target")

    assert target.graph_depth == 1
    assert target.introduced_by[0].seed_skill == "skill:a-low"


def test_expansion_limit_orders_same_depth_by_path_score() -> None:
    skills = {
        skill.id: skill
        for skill in (
            make_skill("skill:z-high", "z-high", "High-ranked seed."),
            make_skill("skill:a-low", "a-low", "Lower-ranked seed."),
            make_skill("skill:high-neighbor", "high-neighbor", "High-score neighbor."),
            make_skill("skill:low-neighbor", "low-neighbor", "Low-score neighbor."),
        )
    }
    edges = [
        _edge("skill:z-high", "skill:high-neighbor", "compose_with"),
        _edge("skill:a-low", "skill:low-neighbor", "compose_with"),
    ]

    result = expand_semantic_candidates(
        [_seed("skill:z-high", 1), _seed("skill:a-low", 20)],
        edges,
        skills,
        max_depth=1,
        limit=3,
    )

    assert [item.skill_id for item in result.candidates[2:]] == ["skill:high-neighbor"]


def test_workflow_expansion_prefers_forward_execution_direction() -> None:
    skills = {
        skill.id: skill
        for skill in (
            make_skill("skill:seed", "seed", "Workflow seed."),
            make_skill("skill:a-previous", "a-previous", "Previous stage."),
            make_skill("skill:z-next", "z-next", "Next stage."),
        )
    }
    edges = [
        _edge("skill:a-previous", "skill:seed", "compose_with"),
        _edge("skill:seed", "skill:z-next", "compose_with"),
    ]

    result = expand_semantic_candidates(
        [_seed("skill:seed")],
        edges,
        skills,
        max_depth=1,
        limit=2,
    )

    assert [item.skill_id for item in result.candidates] == ["skill:seed", "skill:z-next"]


def test_similarity_is_exposed_as_alternative_but_not_traversed() -> None:
    skills = {
        skill.id: skill
        for skill in (
            make_skill("skill:seed", "seed", "Seed skill."),
            make_skill("skill:alternative", "alternative", "Near substitute."),
        )
    }

    result = expand_semantic_candidates(
        [_seed("skill:seed")],
        [_edge("skill:alternative", "skill:seed", "similar_to")],
        skills,
        max_depth=2,
        limit=10,
    )

    assert [candidate.skill_id for candidate in result.candidates] == ["skill:seed"]
    assert len(result.alternatives) == 1
    assert result.alternatives[0].skill_id == "skill:alternative"
    assert result.alternatives[0].alternative_to == "skill:seed"


def test_similarity_between_selected_candidates_keeps_alternative_metadata() -> None:
    skills = {
        skill.id: skill
        for skill in (
            make_skill("skill:preferred", "preferred", "Preferred implementation."),
            make_skill("skill:substitute", "substitute", "Near substitute."),
        )
    }

    result = expand_semantic_candidates(
        [_seed("skill:preferred", 1), _seed("skill:substitute", 2)],
        [_edge("skill:preferred", "skill:substitute", "similar_to")],
        skills,
        max_depth=2,
        limit=2,
    )

    assert len(result.alternatives) == 1
    assert result.alternatives[0].skill_id == "skill:substitute"
    assert result.alternatives[0].alternative_to == "skill:preferred"


def test_similarity_keeps_one_strongest_relation_per_alternative() -> None:
    skills = {
        skill.id: skill
        for skill in (
            make_skill("skill:first", "first", "First selected skill."),
            make_skill("skill:second", "second", "Second selected skill."),
            make_skill("skill:alternative", "alternative", "Shared near substitute."),
        )
    }

    result = expand_semantic_candidates(
        [_seed("skill:first", 1), _seed("skill:second", 2)],
        [
            _edge("skill:alternative", "skill:first", "similar_to", confidence=0.95),
            _edge("skill:alternative", "skill:second", "similar_to", confidence=0.80),
        ],
        skills,
        max_depth=1,
        limit=2,
    )

    assert len(result.alternatives) == 1
    assert result.alternatives[0].skill_id == "skill:alternative"
    assert result.alternatives[0].alternative_to == "skill:first"
    assert result.alternatives[0].confidence == pytest.approx(0.95)


def test_fixture_bundle_uses_rrf_and_validated_graph_only(tmp_path) -> None:
    workspace = tmp_path / ".skillfabric"
    build_fixture_workspace(workspace)

    bundle = build_router_bundle(
        RouterBundleConfig(
            workspace=workspace,
            query="extract financial KPI values from PDF tables",
            seed_limit=2,
            expanded_limit=8,
            max_depth=2,
        ),
        embedding_provider=FakeEmbeddingProvider(),
    )
    payload = bundle.to_dict()

    assert set(payload) == {"query", "selected_skills", "graph_edges", "alternatives"}
    assert all("seed_rank" not in item for item in payload["selected_skills"])
    seeds = [candidate for candidate in bundle.selected_skills if candidate.is_seed]
    assert len(seeds) == 2
    assert all(set(candidate.retrieval_ranks) <= {"bm25", "embedding"} for candidate in seeds)
    assert {edge.type for edge in bundle.graph_edges} <= {"depend_on", "compose_with"}
    assert "workflow_hints" not in payload
    assert "ppr_score" not in str(payload)
    assert "score_breakdown" not in str(payload)
    assert "execution" not in str(payload).lower()


def test_bundle_keeps_operational_edges_for_selectable_alternatives(monkeypatch) -> None:
    seed_skill = make_skill("skill:seed", "seed", "Seed skill.")
    prerequisite = make_skill("skill:prerequisite", "prerequisite", "Prerequisite skill.")
    alternative = make_skill("skill:alternative", "alternative", "Alternative skill.")
    skills = {skill.id: skill for skill in (seed_skill, prerequisite, alternative)}
    dependency = _edge(alternative.id, prerequisite.id, "depend_on")
    seed = _seed(seed_skill.id)
    prerequisite_candidate = RouterSkillCandidate(
        skill_id=prerequisite.id,
        name=prerequisite.name,
        score=0.5,
        graph_depth=1,
    )
    alternative_candidate = RouterAlternative(
        skill_id=alternative.id,
        name=alternative.name,
        alternative_to=seed_skill.id,
        confidence=0.95,
        reason="Validated near substitute.",
    )
    monkeypatch.setattr(
        bundle_module,
        "load_wiki_source",
        lambda _workspace: WikiSource(
            build_id="test-build",
            skills=skills,
            contracts={},
            core_edges=[dependency],
        ),
    )
    monkeypatch.setattr(
        bundle_module,
        "retrieve_seed_candidates",
        lambda *_args, **_kwargs: [seed],
    )
    monkeypatch.setattr(
        bundle_module,
        "expand_semantic_candidates",
        lambda *_args, **_kwargs: ExpansionResult(
            candidates=(seed, prerequisite_candidate),
            alternatives=(alternative_candidate,),
        ),
    )

    bundle = build_router_bundle(
        RouterBundleConfig(query="use the best implementation"),
    )

    assert bundle.graph_edges == (dependency,)


def test_missing_canonical_artifacts_fail_instead_of_returning_empty_bundle(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="required semantic artifact"):
        build_router_bundle(
            RouterBundleConfig(
                workspace=Path(tmp_path) / ".skillfabric",
                query="parse a PDF table",
            ),
            embedding_provider=FakeEmbeddingProvider(),
        )


def test_router_bundle_rejects_unexpected_nested_candidate_fields() -> None:
    payload = {
        "query": "test query",
        "selected_skills": [{**_seed("skill:seed").to_dict(), "unused": True}],
        "graph_edges": [],
        "alternatives": [],
    }

    with pytest.raises(ValueError, match="router skill candidate"):
        RouterBundle.from_dict(payload)


def test_router_bundle_rejects_non_object_expansion_paths() -> None:
    candidate = _seed("skill:seed").to_dict()
    candidate["introduced_by"] = ["invalid-path"]
    payload = {
        "query": "test query",
        "selected_skills": [candidate],
        "graph_edges": [],
        "alternatives": [],
    }

    with pytest.raises(ValueError, match=r"introduced_by\[0\]"):
        RouterBundle.from_dict(payload)
