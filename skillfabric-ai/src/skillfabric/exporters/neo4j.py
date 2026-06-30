"""Export the canonical SkillFabric KG to Neo4j Cypher."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skillfabric.storage import Workspace, atomic_write_text

CORE_RELATION_TYPES = {
    "similar_to": "SIMILAR_TO",
    "member_of": "MEMBER_OF",
    "compose_with": "COMPOSE_WITH",
    "depend_on": "DEPEND_ON",
}

@dataclass(frozen=True)
class Neo4jExportConfig:
    """Configuration for Neo4j Cypher export."""

    workspace: str | Path
    output_path: str | Path | None = None
    batch_size: int = 500
    include_cleanup: bool = False


@dataclass(frozen=True)
class Neo4jExportResult:
    """Summary of a Neo4j Cypher export."""

    output_path: Path
    node_count: int
    relationship_count: int
    statement_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_path": str(self.output_path),
            "node_count": self.node_count,
            "relationship_count": self.relationship_count,
            "statement_count": self.statement_count,
        }


def export_neo4j(config: Neo4jExportConfig) -> Neo4jExportResult:
    """Write a Cypher import script for the clean skill-level KG."""

    workspace = Workspace(config.workspace)
    compiled_path = workspace.graph_dir / "compiled_skill_graph.json"
    if not compiled_path.exists():
        raise FileNotFoundError(
            f"compiled graph not found: {compiled_path}. Run `skillfabric build` first."
        )

    payload = json.loads(compiled_path.read_text(encoding="utf-8"))
    statements: list[str] = []
    if config.include_cleanup:
        statements.append("MATCH (n:SkillFabricEntity) DETACH DELETE n;")
    statements.extend(_constraint_statements())

    node_count = 0
    relationship_count = 0
    core_graph = dict(payload.get("core_graph", {}))

    skill_rows: list[dict[str, Any]] = []
    community_rows: list[dict[str, Any]] = []
    for node in core_graph.get("nodes", []):
        if not isinstance(node, dict):
            continue
        if node.get("type") == "skill":
            skill_rows.append(_skill_row(node))
        elif node.get("type") == "community":
            community_rows.append(_community_row(node))

    statements.extend(
        _batched_unwind(
            skill_rows,
            config.batch_size,
            """
UNWIND $rows AS row
MERGE (n:SkillFabricEntity:Skill {id: row.id})
SET n += row
""",
        )
    )
    statements.extend(
        _batched_unwind(
            community_rows,
            config.batch_size,
            """
UNWIND $rows AS row
MERGE (n:SkillFabricEntity:Community {id: row.id})
SET n += row
""",
        )
    )
    node_count += len(skill_rows) + len(community_rows)

    core_relationship_rows = [_edge_row(edge) for edge in core_graph.get("edges", []) if isinstance(edge, dict)]
    for rel_type, rows in _group_by_type(core_relationship_rows).items():
        statements.extend(
            _batched_unwind(
                rows,
                config.batch_size,
                f"""
UNWIND $rows AS row
MATCH (source:SkillFabricEntity {{id: row.source}})
MATCH (target:SkillFabricEntity {{id: row.target}})
MERGE (source)-[r:{rel_type}]->(target)
SET r += row.props
""",
            )
        )
        relationship_count += len(rows)

    output_path = Path(config.output_path) if config.output_path else workspace.root / "neo4j" / "compiled_skill_graph.cypher"
    atomic_write_text(output_path, "\n\n".join(statement.strip() for statement in statements if statement.strip()) + "\n")
    return Neo4jExportResult(
        output_path=output_path,
        node_count=node_count,
        relationship_count=relationship_count,
        statement_count=len(statements),
    )


def _constraint_statements() -> list[str]:
    return [
        "CREATE CONSTRAINT skillfabric_entity_id IF NOT EXISTS FOR (n:SkillFabricEntity) REQUIRE n.id IS UNIQUE;",
        "CREATE INDEX skillfabric_skill_name IF NOT EXISTS FOR (n:Skill) ON (n.name);",
        "CREATE INDEX skillfabric_community_name IF NOT EXISTS FOR (n:Community) ON (n.name);",
    ]


def _skill_row(node: dict[str, Any]) -> dict[str, Any]:
    return _compact(
        {
            "id": node.get("id", ""),
            "name": node.get("name", ""),
            "description": node.get("description", ""),
            "source_path": node.get("source_path", ""),
            "wiki_path": node.get("wiki_path", ""),
            "content_hash": node.get("content_hash", ""),
            "token_count": int(node.get("token_count", 0) or 0),
            "canonical_skill_text_hash": node.get("canonical_skill_text_hash", ""),
        }
    )


def _community_row(node: dict[str, Any]) -> dict[str, Any]:
    return _compact(
        {
            "id": node.get("id", ""),
            "name": node.get("name", ""),
            "summary": node.get("summary", ""),
            "member_count": int(node.get("member_count", 0) or 0),
            "representative_skill_ids": _string_list(node.get("representative_skill_ids", [])),
            "cohesion_score": float(node.get("cohesion_score", 0.0) or 0.0),
        }
    )


def _edge_row(edge: dict[str, Any]) -> dict[str, Any]:
    rel_type = CORE_RELATION_TYPES.get(str(edge.get("type", "")), str(edge.get("type", "")).upper())
    return {
        "type": rel_type,
        "source": str(edge.get("source", "")),
        "target": str(edge.get("target", "")),
        "props": _edge_props(edge),
    }


def _edge_props(edge: dict[str, Any]) -> dict[str, Any]:
    evidence = edge.get("evidence", [])
    metadata = edge.get("metadata", {})
    props: dict[str, Any] = {
        "confidence": float(edge.get("confidence", 0.0) or 0.0),
        "weight": float(edge.get("weight", 0.0) or 0.0),
        "provenance": str(edge.get("provenance", "")),
        "reason": str(edge.get("reason", "")),
        "evidence_count": len(evidence) if isinstance(evidence, list) else 0,
    }
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            props[f"metadata_{key}"] = str(value)
    return _compact(props)


def _batched_unwind(rows: list[dict[str, Any]], batch_size: int, template: str) -> list[str]:
    if not rows:
        return []
    size = max(int(batch_size or 500), 1)
    statements: list[str] = []
    for batch in _chunks(rows, size):
        statements.append(template.replace("$rows", _cypher_list(batch)).strip() + ";")
    return statements


def _group_by_type(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rel_type = str(row.get("type", "")).upper()
        if not rel_type:
            continue
        grouped.setdefault(rel_type, []).append(
            {
                "source": str(row.get("source", "")),
                "target": str(row.get("target", "")),
                "props": dict(row.get("props", {})),
            }
        )
    return grouped


def _chunks(rows: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for index in range(0, len(rows), size):
        yield rows[index : index + size]


def _cypher_list(rows: list[dict[str, Any]]) -> str:
    return "[" + ", ".join(_cypher_map(row) for row in rows) + "]"


def _cypher_map(payload: dict[str, Any]) -> str:
    parts = [f"{_cypher_key(key)}: {_cypher_value(value)}" for key, value in payload.items()]
    return "{" + ", ".join(parts) + "}"


def _cypher_key(key: str) -> str:
    if key.replace("_", "").isalnum() and not key[:1].isdigit():
        return key
    return f"`{key.replace('`', '``')}`"


def _cypher_value(value: Any) -> str:
    if isinstance(value, dict):
        return _cypher_map(value)
    if isinstance(value, list):
        return "[" + ", ".join(_cypher_value(item) for item in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return repr(value)
    return json.dumps(str(value), ensure_ascii=False)


def _compact(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if value not in ("", None, []) and value != {}
    }


def _string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if str(value)]
