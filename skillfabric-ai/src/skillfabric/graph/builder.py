"""Semantic graph build orchestration."""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from skillfabric.graph.contracts.extraction import (
    ContractExtractor,
    LiteLLMContractExtractor,
    extract_skill_contracts,
)
from skillfabric.graph.contracts.models import SkillContract
from skillfabric.graph.models import GraphDocument
from skillfabric.graph.semantic.candidates import (
    DEFAULT_CANDIDATE_TOP_K,
    retrieve_candidate_pairs,
)
from skillfabric.graph.semantic.models import RelationDecision
from skillfabric.graph.semantic.projection import (
    CycleAdjudicator,
    LiteLLMCycleAdjudicator,
    project_relation_decisions,
)
from skillfabric.graph.semantic.validation import (
    LiteLLMRelationJudge,
    RelationJudge,
    validate_candidate_pairs,
)
from skillfabric.indexing.bm25 import build_bm25_index
from skillfabric.indexing.embeddings import EmbeddingProvider, default_embedding_provider
from skillfabric.registry.models import SkillNode
from skillfabric.registry.scanner import scan_and_parse
from skillfabric.runtime.jobs import LLMJobOptions
from skillfabric.runtime.llm import LLMConfig
from skillfabric.storage import Workspace
from skillfabric.wiki.loader import WikiSource
from skillfabric.wiki.materializer import materialize_full_wiki
from skillfabric.wiki.models import WikiBuildConfig, WikiBuildResult


@dataclass(slots=True)
class BuildConfig:
    """Public configuration for one SkillFabric workspace build."""

    skill_root: str | Path
    workspace: str | Path = ".skillfabric"
    llm_env_path: str | Path = ".env"
    llm_options: LLMJobOptions | None = None
    llm_model: str | None = None
    llm_reasoning_effort: str | None = None


@dataclass(slots=True)
class _BuildDependencies:
    """Private provider overrides for tests and internal adapters."""

    contract_extractor: ContractExtractor | None = None
    relation_judge: RelationJudge | None = None
    cycle_adjudicator: CycleAdjudicator | None = None
    embedding_provider: EmbeddingProvider | None = None
    build_id: str | None = None


@dataclass(slots=True)
class BuildResult:
    """Result of a complete graph and Full Wiki build."""

    graph: GraphDocument
    wiki: WikiBuildResult
    workspace: Workspace
    stats: dict[str, Any] = field(default_factory=dict)


def build_workspace(
    config: BuildConfig,
    *,
    dependencies: _BuildDependencies | None = None,
) -> BuildResult:
    """Compile skills into one ready, evidence-grounded workspace."""

    deps = dependencies or _BuildDependencies()
    workspace = Workspace(config.workspace)
    workspace.ensure()
    build_id = deps.build_id or str(time.time_ns())
    stage = "initialization"
    lock_acquired = False
    resolved_llm_config: LLMConfig | None = None

    def build_llm_config() -> LLMConfig:
        nonlocal resolved_llm_config
        if resolved_llm_config is None:
            resolved_llm_config = LLMConfig.from_env(
                env_path=config.llm_env_path,
                model=config.llm_model,
                reasoning_effort=config.llm_reasoning_effort,
            )
        return resolved_llm_config

    try:
        with workspace.lock():
            lock_acquired = True
            previous_skills = _load_previous_skill_snapshot(workspace)
            contract_extractor = deps.contract_extractor or LiteLLMContractExtractor(
                config=build_llm_config()
            )
            relation_judge = deps.relation_judge or LiteLLMRelationJudge(config=build_llm_config())
            embedding_provider = deps.embedding_provider or default_embedding_provider(
                env_path=config.llm_env_path
            )
            cycle_adjudicator = _resolve_cycle_adjudicator(
                deps,
                relation_judge,
            )
            job_options = (config.llm_options or LLMJobOptions()).normalized()
            _write_running_status(workspace, build_id, stage="scan")

            stage = "scan"
            skills = scan_and_parse(config.skill_root)
            if not skills:
                raise ValueError(f"no SKILL.md files found under {config.skill_root}")
            duplicate_ids = sorted(
                skill_id
                for skill_id, count in Counter(skill.id for skill in skills).items()
                if count > 1
            )
            if duplicate_ids:
                raise ValueError(f"duplicate skill id(s): {', '.join(duplicate_ids)}")
            change_stats = _skill_change_stats(previous_skills, skills)

            stage = "contracts"
            _write_running_status(workspace, build_id, stage=stage)
            contract_records = extract_skill_contracts(
                skills,
                extractor=contract_extractor,
                cache_path=workspace.cache_dir / "contracts.json",
                job_options=job_options,
            )
            contracts = {record.contract.skill_id: record.contract for record in contract_records}

            stage = "indexes"
            _write_running_status(workspace, build_id, stage=stage)
            build_bm25_index(
                skills,
                workspace.graph_dir / "bm25.sqlite",
                contracts=contracts,
            )
            retrieval = retrieve_candidate_pairs(
                contracts,
                skills,
                provider=embedding_provider,
                bm25_path=workspace.graph_dir / "bm25.sqlite",
                store_path=workspace.cache_dir / "embeddings.json",
                binary_store_path=workspace.graph_dir / "embeddings.json",
                candidate_top_k=DEFAULT_CANDIDATE_TOP_K,
            )

            stage = "decisions"
            _write_running_status(workspace, build_id, stage=stage)
            decisions = validate_candidate_pairs(
                retrieval.pairs,
                skills,
                contracts,
                judge=relation_judge,
                cache_path=workspace.cache_dir / "relation_decisions.json",
                job_options=job_options,
            )

            stage = "projection"
            _write_running_status(workspace, build_id, stage=stage)
            projection = project_relation_decisions(
                decisions,
                skills,
                cycle_adjudicator=cycle_adjudicator,
            )

            stage = "artifacts"
            _write_running_status(workspace, build_id, stage=stage)
            edge_counts = Counter(edge.type for edge in projection.edges)
            stats = {
                **change_stats,
                "skill_count": len(skills),
                "edge_count": len(projection.edges),
                "edge_counts": {
                    "depend_on": edge_counts.get("depend_on", 0),
                    "compose_with": edge_counts.get("compose_with", 0),
                    "similar_to": edge_counts.get("similar_to", 0),
                },
                "candidate_pair_count": len(retrieval.pairs),
                "accepted_relation_count": sum(
                    1 for decision in projection.decisions if decision.relation != "none"
                ),
                "rejected_relation_count": sum(
                    1 for decision in projection.decisions if decision.relation == "none"
                ),
                "contract_cache_hits": sum(record.cache_hit for record in contract_records),
                "relation_cache_hits": sum(decision.cache_hit for decision in decisions),
                "cycle_review_count": projection.cycle_review_count,
                "embedding_cache_hits": int(retrieval.metrics.get("cache_hit_count", 0)),
                "new_embedding_count": int(retrieval.metrics.get("new_embedding_count", 0)),
            }
            _write_canonical_artifacts(
                workspace,
                skills=skills,
                contracts=contracts,
                decisions=projection.decisions,
            )
            graph = GraphDocument(
                build_id=build_id,
                nodes=skills,
                edges=list(projection.edges),
            )
            workspace.write_json(workspace.graph_dir / "graph.json", graph.to_dict())
            stage = "wiki"
            _write_running_status(workspace, build_id, stage=stage)
            wiki = materialize_full_wiki(
                WikiBuildConfig(workspace=workspace.root),
                source=WikiSource(
                    build_id=build_id,
                    skills={skill.id: skill for skill in skills},
                    contracts=contracts,
                    core_edges=graph.edges,
                ),
            )
            _write_ready_status(workspace, graph=graph)
            return BuildResult(
                graph=graph,
                wiki=wiki,
                workspace=workspace,
                stats=stats,
            )
    except Exception as exc:
        if lock_acquired:
            _write_failed_status(
                workspace,
                build_id=build_id,
                stage=stage,
                error=exc,
            )
        raise


def _write_canonical_artifacts(
    workspace: Workspace,
    *,
    skills: list[SkillNode],
    contracts: dict[str, SkillContract],
    decisions: tuple[RelationDecision, ...],
) -> None:
    workspace.write_jsonl(
        workspace.graph_dir / "registry.jsonl",
        [skill.to_dict(include_raw_text=True) for skill in skills],
    )
    workspace.write_jsonl(
        workspace.graph_dir / "contracts.jsonl",
        [contracts[skill_id].to_dict() for skill_id in sorted(contracts)],
    )
    workspace.write_jsonl(
        workspace.graph_dir / "relation_decisions.jsonl",
        [decision.to_dict() for decision in decisions],
    )


def _load_previous_skill_snapshot(workspace: Workspace) -> dict[str, str] | None:
    """Load the last ready registry for user-facing incremental build reporting."""

    status_path = workspace.status_path
    registry_path = workspace.graph_dir / "registry.jsonl"
    if not status_path.is_file() or not registry_path.is_file():
        return None
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if (
            not isinstance(status, dict)
            or status.get("state") != "ready"
            or status.get("stage") != "complete"
        ):
            return None
        snapshot: dict[str, str] = {}
        for line in registry_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                return None
            skill_id = row.get("id")
            content_hash = row.get("content_hash")
            if (
                not isinstance(skill_id, str)
                or not skill_id.strip()
                or not isinstance(content_hash, str)
                or not content_hash.strip()
                or skill_id in snapshot
            ):
                return None
            snapshot[skill_id] = content_hash
        return snapshot or None
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _skill_change_stats(
    previous: dict[str, str] | None,
    skills: list[SkillNode],
) -> dict[str, int | bool]:
    """Describe source changes without exposing internal cache controls."""

    current = {skill.id: skill.content_hash for skill in skills}
    if previous is None:
        return {
            "incremental": False,
            "added_skill_count": 0,
            "modified_skill_count": 0,
            "removed_skill_count": 0,
            "reused_skill_count": 0,
        }
    added = set(current) - set(previous)
    removed = set(previous) - set(current)
    modified = {
        skill_id
        for skill_id in set(current) & set(previous)
        if current[skill_id] != previous[skill_id]
    }
    return {
        "incremental": True,
        "added_skill_count": len(added),
        "modified_skill_count": len(modified),
        "removed_skill_count": len(removed),
        "reused_skill_count": len(set(current) - added - modified),
    }


def _write_running_status(
    workspace: Workspace,
    build_id: str,
    *,
    stage: str,
) -> None:
    workspace.write_json(
        workspace.status_path,
        {
            "state": "building",
            "stage": stage,
            "build_id": build_id,
        },
    )


def _write_ready_status(
    workspace: Workspace,
    *,
    graph: GraphDocument,
) -> None:
    workspace.write_json(
        workspace.status_path,
        {
            "state": "ready",
            "stage": "complete",
            "build_id": graph.build_id,
        },
    )


def _write_failed_status(
    workspace: Workspace,
    *,
    build_id: str,
    stage: str,
    error: Exception,
) -> None:
    workspace.ensure()
    workspace.write_json(
        workspace.status_path,
        {
            "state": "failed",
            "failed_stage": stage,
            "build_id": build_id,
            "error_type": type(error).__name__,
            "error": _safe_error(error),
        },
    )


def _safe_error(error: Exception, *, limit: int = 500) -> str:
    text = " ".join(str(error).split())
    text = re.sub(r"(?i)\bsk-[a-z0-9._-]+", "[redacted]", text)
    text = re.sub(
        r"(?i)\b(api[_-]?key|token|authorization)\s*[:=]\s*\S+",
        r"\1=[redacted]",
        text,
    )
    return text[:limit]


def _resolve_cycle_adjudicator(
    deps: _BuildDependencies,
    relation_judge: RelationJudge,
) -> CycleAdjudicator | None:
    if deps.cycle_adjudicator is not None:
        return deps.cycle_adjudicator
    if isinstance(relation_judge, LiteLLMRelationJudge):
        return LiteLLMCycleAdjudicator(config=relation_judge.config)
    return None
