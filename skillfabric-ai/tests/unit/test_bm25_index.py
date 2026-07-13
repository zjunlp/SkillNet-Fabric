from __future__ import annotations

from skillfabric.indexing.bm25 import BM25Hit, _fts_query, build_bm25_index, search_bm25
from tests.unit.relation_helpers import make_skill


def test_fts_query_keeps_content_words_without_a_handwritten_stoplist() -> None:
    query = _fts_query("heat the potato then put it in a receptacle")

    assert query == (
        '"heat" OR "the" OR "potato" OR "then" OR "put" OR "it" OR "in" OR "a" OR "receptacle"'
    )


def test_search_returns_raw_bm25_order_and_explicit_ranks(tmp_path) -> None:
    skills = [
        make_skill("skill:pdf", "pdf", "Extract PDF tables into CSV."),
        make_skill("skill:report", "report", "Write a report from JSON."),
        make_skill("skill:testing", "testing", "Run Python tests."),
    ]
    path = tmp_path / "bm25.sqlite"
    build_bm25_index(skills, path)

    hits = search_bm25(path, "extract PDF table", limit=3)

    assert hits
    assert all(isinstance(hit, BM25Hit) for hit in hits)
    assert [hit.rank for hit in hits] == list(range(1, len(hits) + 1))
    assert hits[0].skill_id == "skill:pdf"
    assert all(hit.raw_score <= 0 for hit in hits)
