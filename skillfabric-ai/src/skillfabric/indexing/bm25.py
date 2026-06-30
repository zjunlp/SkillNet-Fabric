"""SQLite FTS5 and BM25 index wrappers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from skillfabric.indexing.canonical import canonical_skill_text
from skillfabric.registry.models import SkillNode

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "then",
    "this",
    "to",
    "with",
}


def build_bm25_index(skills: list[SkillNode], db_path: str | Path) -> None:
    """Write a SQLite FTS5 index."""

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE IF EXISTS skills_fts")
        conn.execute(
            "CREATE VIRTUAL TABLE skills_fts USING fts5(skill_id UNINDEXED, name, description, body)"
        )
        conn.executemany(
            "INSERT INTO skills_fts(skill_id, name, description, body) VALUES (?, ?, ?, ?)",
            [
                (
                    skill.id,
                    skill.name,
                    skill.description,
                    canonical_skill_text(skill),
                )
                for skill in skills
            ],
        )
        conn.commit()


def search_bm25(db_path: str | Path, query: str, *, limit: int = 10) -> list[tuple[str, float]]:
    """Query BM25 and return skill ids with normalized scores."""

    if not query.strip():
        return []
    path = Path(db_path)
    if not path.exists():
        return []
    safe_query = _fts_query(query)
    if not safe_query:
        return []
    with sqlite3.connect(path) as conn:
        try:
            rows = conn.execute(
                """
                SELECT skill_id, bm25(skills_fts) AS rank
                FROM skills_fts
                WHERE skills_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (safe_query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    if not rows:
        return []
    raw_scores = [1.0 / (1.0 + abs(float(rank))) for _, rank in rows]
    max_score = max(raw_scores) or 1.0
    return [
        (str(skill_id), score / max_score)
        for (skill_id, _), score in zip(rows, raw_scores, strict=False)
    ]


def _fts_query(query: str) -> str:
    tokens = []
    for token in query.replace("-", " ").replace("_", " ").split():
        cleaned = "".join(ch for ch in token.lower() if ch.isalnum())
        if cleaned and cleaned not in _STOPWORDS:
            tokens.append(cleaned)
    return " OR ".join(tokens[:24])
