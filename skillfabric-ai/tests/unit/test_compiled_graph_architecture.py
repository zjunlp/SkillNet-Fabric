from __future__ import annotations

import importlib
from pathlib import Path


def test_semantic_compiler_is_the_only_graph_build_pipeline() -> None:
    builder = importlib.import_module("skillfabric.compiled_graph.builder")
    models = importlib.import_module("skillfabric.compiled_graph.models")
    contracts = importlib.import_module("skillfabric.compiled_graph.contracts")
    semantic = importlib.import_module("skillfabric.compiled_graph.semantic")

    assert hasattr(builder, "build_graph")
    assert hasattr(builder, "BuildConfig")
    assert hasattr(models, "GraphDocument")
    assert hasattr(contracts, "SkillContract")
    assert hasattr(semantic, "retrieve_candidate_pairs")
    assert hasattr(semantic, "project_relation_decisions")


def test_removed_graph_compilers_have_no_source_files() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "skillfabric" / "compiled_graph"

    for name in ("canonicalization", "execution", "interface", "relations"):
        assert not list((root / name).glob("*.py"))
    assert not (root / "community_sidecar.py").exists()


def test_removed_wiki_page_index_has_no_source_modules() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "skillfabric" / "wiki" / "explorer"

    assert not (root / "models.py").exists()
    assert not (root / "search_index.py").exists()
