"""Schema-v2 semantic graph build orchestration."""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from skillfabric.compiled_graph.contracts.extraction import (
    ContractExtractor,
    LiteLLMContractExtractor,
    extract_skill_contracts,
)
from skillfabric.compiled_graph.contracts.models import SkillContract
from skillfabric.compiled_graph.models import GraphDocument
from skillfabric.compiled_graph.semantic.candidates import (
    DEFAULT_HANDOFF_TOP_K,
    DEFAULT_SIMILARITY_TOP_K,
    retrieve_candidate_pairs,
)
from skillfabric.compiled_graph.semantic.models import RelationDecision
from skillfabric.compiled_graph.semantic.projection import (
    CycleAdjudicator,
    LiteLLMCycleAdjudicator,
    project_relation_decisions,
)
from skillfabric.compiled_graph.semantic.validation import (
    LiteLLMRelationJudge,
    RelationJudge,
    validate_candidate_pairs,
)
from skillfabric.indexing.bm25 import build_bm25_index
from skillfabric.indexing.embeddings import EmbeddingProvider, default_embedding_provider
from skillfabric.registry.models import SkillNode
from skillfabric.registry.scanner import scan_and_parse
from skillfabric.runtime.jobs import LLMJobOptions
from skillfabric.runtime.llm import llm_usage_context
from skillfabric.runtime.usage import load_usage_records, summarize_usage
from skillfabric.storage import Workspace, atomic_write_text

SCHEMA_VERSION = "2.0"


@dataclass(slots=True)
class BuildConfig:
    """Public configuration for one semantic graph build."""

    skill_root: str | Path
    workspace: str | Path = ".skillfabric"
    llm_env_path: str | Path = ".env"
    llm_options: LLMJobOptions | None = None


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
    """Result of a complete schema-v2 graph build."""

    graph: GraphDocument
    workspace: Workspace
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _StageTimer:
    started_at: float = field(default_factory=time.perf_counter)
    last_mark: float = field(default_factory=time.perf_counter)
    timings: dict[str, float] = field(default_factory=dict)

    def mark(self, stage: str) -> None:
        now = time.perf_counter()
        self.timings[stage] = round(now - self.last_mark, 6)
        self.last_mark = now

    def total(self) -> float:
        return round(time.perf_counter() - self.started_at, 6)


def build_graph(
    config: BuildConfig,
    *,
    dependencies: _BuildDependencies | None = None,
) -> BuildResult:
    """Compile skills into one evidence-grounded semantic graph."""

    deps = dependencies or _BuildDependencies()
    workspace = Workspace(config.workspace)
    _require_compatible_workspace(workspace)
    workspace.ensure()
    _prepare_usage_log(workspace)
    build_id = deps.build_id or str(time.time_ns())
    stage = "initialization"
    timer = _StageTimer()
    try:
        with (
            workspace.lock(),
            llm_usage_context(
                log_path=workspace.reports_dir / "llm_usage.jsonl",
                metadata={"build_id": build_id},
            ),
        ):
            contract_extractor = deps.contract_extractor or LiteLLMContractExtractor.from_env(
                env_path=config.llm_env_path
            )
            relation_judge = deps.relation_judge or LiteLLMRelationJudge.from_env(
                env_path=config.llm_env_path
            )
            embedding_provider = deps.embedding_provider or default_embedding_provider(
                env_path=config.llm_env_path
            )
            cycle_adjudicator = _resolve_cycle_adjudicator(
                deps,
                relation_judge,
                env_path=config.llm_env_path,
            )
            job_options = config.llm_options or LLMJobOptions.from_env(env_path=config.llm_env_path)
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
            timer.mark(stage)

            stage = "contracts"
            _write_running_status(workspace, build_id, stage=stage)
            contract_records = extract_skill_contracts(
                skills,
                extractor=contract_extractor,
                cache_path=workspace.cache_dir / "contracts.json",
                job_options=job_options,
            )
            contracts = {record.contract.skill_id: record.contract for record in contract_records}
            timer.mark(stage)

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
                store_path=workspace.graph_dir / "embeddings.json",
                handoff_top_k=DEFAULT_HANDOFF_TOP_K,
                similarity_top_k=DEFAULT_SIMILARITY_TOP_K,
            )
            timer.mark(stage)

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
            timer.mark(stage)

            stage = "projection"
            _write_running_status(workspace, build_id, stage=stage)
            projection = project_relation_decisions(
                decisions,
                skills,
                cycle_adjudicator=cycle_adjudicator,
            )
            timer.mark(stage)

            stage = "artifacts"
            _write_running_status(workspace, build_id, stage=stage)
            edge_counts = Counter(edge.type for edge in projection.edges)
            stats = {
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
                "contract_model_id": contract_extractor.model_id,
                "relation_model_id": relation_judge.model_id,
                "embedding": retrieval.metrics,
                "stage_wall_time_seconds": timer.timings,
                "build_wall_time_seconds": timer.total(),
            }
            _write_canonical_artifacts(
                workspace,
                skills=skills,
                contracts=contracts,
                decisions=projection.decisions,
            )
            timer.mark(stage)
            stats["stage_wall_time_seconds"] = timer.timings
            stats["build_wall_time_seconds"] = timer.total()
            graph = GraphDocument(
                schema_version=SCHEMA_VERSION,
                build_id=build_id,
                nodes=skills,
                edges=list(projection.edges),
            )
            workspace.write_json(workspace.graph_dir / "graph.json", graph.to_dict())
            _write_build_summary(workspace, build_id, stats)
            _write_ready_status(
                workspace,
                graph=graph,
            )
            return BuildResult(
                graph=graph,
                workspace=workspace,
                stats=stats,
            )
    except Exception as exc:
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


def _write_build_summary(
    workspace: Workspace,
    build_id: str,
    stats: dict[str, Any],
) -> None:
    usage_path = workspace.reports_dir / "llm_usage.jsonl"
    usage = summarize_usage(
        load_usage_records(usage_path),
        metadata={"build_id": build_id},
    ).to_dict()
    workspace.write_json(
        workspace.reports_dir / "build_summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            **stats,
            "llm_usage": usage,
        },
    )


def _write_running_status(
    workspace: Workspace,
    build_id: str,
    *,
    stage: str,
) -> None:
    workspace.write_json(
        workspace.status_path,
        {
            "schema_version": SCHEMA_VERSION,
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
            "schema_version": SCHEMA_VERSION,
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
            "schema_version": SCHEMA_VERSION,
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


def _prepare_usage_log(workspace: Workspace) -> None:
    path = workspace.reports_dir / "llm_usage.jsonl"
    if not path.exists():
        atomic_write_text(path, "")


def _require_compatible_workspace(workspace: Workspace) -> None:
    if not workspace.root.exists():
        return
    if workspace.status_path.exists():
        try:
            status = json.loads(workspace.status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("workspace is incompatible; use a new workspace") from exc
        if not isinstance(status, dict) or status.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("workspace is incompatible; use a new workspace")
        return
    for generated_dir in (workspace.graph_dir, workspace.cache_dir, workspace.wiki_dir):
        if generated_dir.is_dir() and any(generated_dir.iterdir()):
            raise ValueError("workspace is incompatible; use a new workspace")


def _resolve_cycle_adjudicator(
    deps: _BuildDependencies,
    relation_judge: RelationJudge,
    *,
    env_path: str | Path,
) -> CycleAdjudicator | None:
    if deps.cycle_adjudicator is not None:
        return deps.cycle_adjudicator
    if isinstance(relation_judge, LiteLLMRelationJudge):
        return LiteLLMCycleAdjudicator.from_env(env_path=env_path)
    return None
