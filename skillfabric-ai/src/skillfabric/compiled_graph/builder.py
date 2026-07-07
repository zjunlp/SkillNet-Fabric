"""KG build pipeline orchestration."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from skillfabric.compiled_graph.canonicalization.candidates import (
    EmbeddingProviderCanonicalEmbedder,
)
from skillfabric.compiled_graph.canonicalization.compiler import (
    DeterministicCanonicalizationProvider,
    LiteLLMCanonicalizationProvider,
    canonicalize_contract_objects,
)
from skillfabric.compiled_graph.canonicalization.health import (
    analyze_canonicalization_health,
    render_canonicalization_health_report,
)
from skillfabric.compiled_graph.canonicalization.models import (
    CanonicalizationBuild,
    CanonicalizationProvider,
)
from skillfabric.compiled_graph.execution.compiler import (
    ExecutionGraphBuild,
    compile_execution_graph,
    execution_index_from_validation_records,
)
from skillfabric.compiled_graph.execution.health import (
    analyze_execution_health,
    render_execution_health_report,
)
from skillfabric.compiled_graph.execution.models import ExecutionValidationRecord
from skillfabric.compiled_graph.execution.projection import project_execution_records
from skillfabric.compiled_graph.execution.validation import (
    DeterministicExecutionFlowValidator,
    ExecutionFlowValidator,
    LiteLLMExecutionFlowValidator,
    summarize_execution_validation_records,
    validate_execution_flow_candidates,
)
from skillfabric.compiled_graph.health import render_health_report
from skillfabric.compiled_graph.interface.extraction import (
    DeterministicInterfaceExtractor,
    LiteLLMInterfaceExtractor,
    SkillInterfaceExtractor,
    extract_skill_interfaces,
)
from skillfabric.compiled_graph.interface.health import (
    analyze_interface_health,
    render_interface_health_report,
)
from skillfabric.compiled_graph.interface.models import InterfaceExtractionRecord, SkillInterface
from skillfabric.compiled_graph.models import Edge, GraphDocument
from skillfabric.compiled_graph.relations.candidates import generate_relation_candidates
from skillfabric.compiled_graph.relations.edge_safety import enforce_depend_on_acyclicity
from skillfabric.compiled_graph.relations.similarity import build_similar_edges
from skillfabric.compiled_graph.relations.validation import (
    LiteLLMPairValidator,
    NoopPairValidator,
    PairValidator,
    summarize_relation_validation_records,
    validate_relation_candidates,
)
from skillfabric.indexing.bm25 import build_bm25_index, search_bm25
from skillfabric.indexing.canonical import canonical_skill_text
from skillfabric.indexing.embeddings import (
    EmbeddingProvider,
    build_embedding_store,
    default_embedding_provider,
)
from skillfabric.indexing.neighbors import build_neighbor_scores
from skillfabric.registry.models import SkillNode
from skillfabric.registry.scanner import scan_and_parse
from skillfabric.runtime.jobs import LLMJobOptions
from skillfabric.runtime.llm import llm_usage_context
from skillfabric.runtime.usage import load_usage_records, summarize_usage
from skillfabric.storage import Workspace, atomic_write_text


@dataclass(slots=True)
class BuildConfig:
    """Configuration for a KG build."""

    skill_root: str | Path
    workspace: str | Path = ".skillfabric"
    similar_top_k: int = 5
    candidate_top_k: int = 20
    validator: PairValidator | None = None
    execution_validator: ExecutionFlowValidator | None = None
    interface_extractor: SkillInterfaceExtractor | None = None
    canonicalization_provider: CanonicalizationProvider | None = None
    embedding_provider: EmbeddingProvider | None = None
    llm_env_path: str | Path = ".env"
    build_id: str | None = None
    skip_llm_validation: bool = False
    skip_interface_extraction: bool = False
    skip_execution_layer: bool = False
    execution_bucket_limit: int = 100
    llm_concurrency: int | None = None
    llm_rate_limit_per_minute: float | None = None
    llm_max_retries: int | None = None
    llm_retry_backoff_seconds: float | None = None
    llm_progress_every: int | None = None
    llm_batch_size: int | None = None


@dataclass(slots=True)
class BuildResult:
    """Result returned by a KG build."""

    graph: GraphDocument
    skills: list[SkillNode]
    workspace: Workspace
    interfaces: dict[str, SkillInterface] = field(default_factory=dict)
    execution_graph: ExecutionGraphBuild = field(default_factory=ExecutionGraphBuild)
    execution_records: list[ExecutionValidationRecord] = field(default_factory=list)
    canonicalization: CanonicalizationBuild = field(default_factory=CanonicalizationBuild)
    stats: dict[str, Any] = field(default_factory=dict)

    def neighbors(self, skill_id: str) -> list[dict[str, Any]]:
        """Return graph neighbors for a skill."""

        output: dict[tuple[str, str], dict[str, Any]] = {}
        for edge in self.graph.edges:
            if edge.source == skill_id:
                record = _neighbor_record(edge, edge.target)
            elif edge.target == skill_id:
                record = _neighbor_record(edge, edge.source)
            else:
                continue
            if not str(record["skill_id"]).startswith("skill:"):
                continue
            key = (str(record["skill_id"]), str(record["edge_type"]))
            existing = output.get(key)
            if existing is None or float(record["weight"]) > float(existing["weight"]):
                output[key] = record
        values = list(output.values())
        values.sort(key=lambda item: (-float(item["weight"]), item["skill_id"]))
        return values


@dataclass(slots=True)
class _StageTimer:
    """Collect stage-level wall-clock timings for build metrics."""

    started_at: float = field(default_factory=time.perf_counter)
    last_mark: float = field(default_factory=time.perf_counter)
    timings: dict[str, float] = field(default_factory=dict)

    def mark(self, stage: str) -> None:
        now = time.perf_counter()
        self.timings[stage] = round(now - self.last_mark, 6)
        self.last_mark = now

    def total(self) -> float:
        return round(time.perf_counter() - self.started_at, 6)


def build_graph(config: BuildConfig) -> BuildResult:
    """Run the full offline KG build pipeline."""

    workspace = Workspace(config.workspace)
    build_id = config.build_id or str(int(time.time()))
    embedding_provider = config.embedding_provider or default_embedding_provider(env_path=config.llm_env_path)
    llm_job_options = _llm_job_options(config)
    config_digest = _config_digest(config)
    stage_timer = _StageTimer()
    usage_log_path = workspace.reports_dir / "llm_usage.jsonl"
    workspace.ensure()

    with workspace.lock(), llm_usage_context(log_path=usage_log_path, metadata={"build_id": build_id}):
        validator = _resolve_pair_validator(config)
        interface_extractor = _resolve_interface_extractor(config)
        execution_validator = _resolve_execution_validator(config)
        canonicalization_provider = _resolve_canonicalization_provider(config)
        workspace.write_json(
            workspace.checkpoint_path,
            {"stage": "scan", "build_id": build_id, "config_digest": config_digest},
        )
        old_status = workspace.read_json(workspace.status_path, default={}) or {}
        old_hashes = old_status.get("skill_hashes", {}) if isinstance(old_status, dict) else {}
        skills = scan_and_parse(config.skill_root, workspace=workspace.root)
        skipped_unchanged = sum(1 for skill in skills if old_hashes.get(skill.id) == skill.content_hash)
        _write_registry(workspace, skills)
        stage_timer.mark("scan")

        workspace.write_json(
            workspace.checkpoint_path,
            {"stage": "index", "build_id": build_id, "config_digest": config_digest},
        )
        bm25_path = workspace.index_dir / "bm25.sqlite"
        build_bm25_index(skills, bm25_path)
        embeddings = build_embedding_store(
            skills,
            workspace.index_dir / "embeddings.json",
            provider=embedding_provider,
        )
        embedding_metrics = _embedding_metrics(skills, embeddings, embedding_provider)
        workspace.write_jsonl(
            workspace.index_dir / "embedding_meta.jsonl",
            [
                {
                    "skill_id": skill.id,
                    "content_hash": skill.content_hash,
                    "canonical_skill_text_hash": skill.canonical_skill_text_hash,
                    "model_id": embedding_provider.model_id,
                    "vector_index": index,
                }
                for index, skill in enumerate(skills)
            ],
        )
        stage_timer.mark("index")

        workspace.write_json(
            workspace.checkpoint_path,
            {"stage": "similarity", "build_id": build_id, "config_digest": config_digest},
        )
        neighbor_scores = build_neighbor_scores(
            skills,
            embeddings,
            top_k=config.similar_top_k,
            bm25_neighbors=_bm25_neighbor_lookup(
                skills,
                bm25_path,
                limit=max(config.similar_top_k * 4, 20),
            ),
        )
        similar_edges = build_similar_edges(neighbor_scores)
        stage_timer.mark("similarity")

        workspace.write_json(
            workspace.checkpoint_path,
            {"stage": "interface", "build_id": build_id, "config_digest": config_digest},
        )
        interface_records = extract_skill_interfaces(
            skills,
            extractor=interface_extractor,
            cache_path=workspace.cache_dir / "interface_cache.json",
            job_options=llm_job_options,
        )
        interfaces = {record.interface.skill_id: record.interface for record in interface_records}
        _write_interface_artifacts(workspace, interface_records)
        stage_timer.mark("interface")

        workspace.write_json(
            workspace.checkpoint_path,
            {"stage": "canonicalization", "build_id": build_id, "config_digest": config_digest},
        )
        canonicalization = canonicalize_contract_objects(
            interfaces,
            provider=canonicalization_provider,
            cache_path=workspace.cache_dir / "canonicalization_cache.json",
            job_options=llm_job_options,
            semantic_embedder=EmbeddingProviderCanonicalEmbedder(embedding_provider),
        )
        _write_canonicalization_artifacts(workspace, canonicalization)
        stage_timer.mark("canonicalization")

        workspace.write_json(
            workspace.checkpoint_path,
            {"stage": "execution", "build_id": build_id, "config_digest": config_digest},
        )
        if config.skip_execution_layer:
            execution_graph = ExecutionGraphBuild()
            execution_records: list[ExecutionValidationRecord] = []
        else:
            execution_graph = compile_execution_graph(
                interfaces,
                bucket_limit=config.execution_bucket_limit,
                canonicalization=canonicalization,
            )
            execution_records = validate_execution_flow_candidates(
                execution_graph.candidates,
                skills,
                interfaces=interfaces,
                validator=execution_validator,
                cache_path=workspace.cache_dir / "execution_validation_cache.json",
                job_options=llm_job_options,
            )
            execution_graph.execution_index = execution_index_from_validation_records(execution_records)
        _write_execution_artifacts(workspace, execution_graph, execution_records)
        execution_validation_summary = summarize_execution_validation_records(execution_records)
        stage_timer.mark("execution")

        workspace.write_json(
            workspace.checkpoint_path,
            {"stage": "compose_depend", "build_id": build_id, "config_digest": config_digest},
        )
        relation_candidates = generate_relation_candidates(
            skills,
            similar_edges,
            per_skill_limit=config.candidate_top_k,
            interfaces=interfaces,
            execution_records=execution_records,
        )
        validation_records = validate_relation_candidates(
            relation_candidates,
            skills,
            validator=validator,
            cache_path=workspace.cache_dir / "relation_validation_cache.json",
            interfaces=interfaces,
            execution_records=execution_records,
            job_options=llm_job_options,
        )
        relation_validation_summary = summarize_relation_validation_records(validation_records)
        compose_depend_edges = [record.edge for record in validation_records if record.edge is not None]
        edge_evidence_rows = [record.to_record() for record in validation_records]
        stage_timer.mark("compose_depend")

        all_edges: list[Edge] = []
        all_edges.extend(similar_edges)
        all_edges.extend(compose_depend_edges)
        all_edges = project_execution_records(execution_records, all_edges)
        edge_safety = enforce_depend_on_acyclicity(all_edges)
        all_edges = edge_safety.edges
        stats = {
            "skill_count": len(skills),
            "edge_count": len(all_edges),
            "similar_to_count": len(similar_edges),
            "compose_depend_candidate_count": len(relation_candidates),
            "compose_depend_edge_count": len(compose_depend_edges),
            "skipped_unchanged": skipped_unchanged,
            "embedding_model_id": embedding_provider.model_id,
            "embedding": embedding_metrics,
            "interface_count": len(interfaces),
            "interface_accepted_count": sum(1 for record in interface_records if record.accepted),
            "interface_rejected_count": sum(1 for record in interface_records if not record.accepted),
            "interface_model_id": interface_extractor.model_id,
            "canonicalization_model_id": canonicalization_provider.model_id,
            "raw_contract_object_count": len(canonicalization.raw_terms),
            "canonical_object_count": len(canonicalization.objects),
            "promoted_object_count": sum(1 for item in canonicalization.objects if item.promoted),
            "canonical_candidate_edge_count": len(canonicalization.candidate_edges),
            "canonical_candidate_component_count": len(canonicalization.candidate_components),
            "canonical_ambiguous_component_count": sum(
                1 for item in canonicalization.candidate_components if getattr(item, "ambiguous", False)
            ),
            "canonical_merge_audit_count": len(canonicalization.merge_audit),
            "raw_artifact_count": len(execution_graph.raw_artifact_nodes),
            "raw_scenario_count": len(execution_graph.raw_scenario_nodes),
            "canonical_artifact_count": len(
                {
                    record.canonical_object
                    for record in execution_graph.execution_index
                    if record.relation_type == "artifact_compatibility"
                }
            ),
            "reusable_state_count": len(
                {
                    record.canonical_object
                    for record in execution_graph.execution_index
                    if record.relation_type == "state_compatibility"
                }
            ),
            "execution_compatibility_count": len(execution_graph.execution_index),
            "canonical_alias_count": len(canonicalization.assignments),
            "alias_merge_ratio": _canonicalization_alias_merge_ratio(canonicalization),
            "execution_candidate_count": len(execution_graph.candidates),
            "execution_accepted_flow_count": sum(1 for record in execution_records if record.accepted),
            "execution_rejected_flow_count": sum(1 for record in execution_records if not record.accepted),
            "execution_projected_edge_count": _execution_projected_edge_count(execution_records),
            "execution_model_id": execution_validator.model_id,
            "depend_on_cycle_pruned_count": len(edge_safety.removed_depend_on_cycle_edges),
            "execution_validation": execution_validation_summary,
            "relation_validation": relation_validation_summary,
            "stage_wall_time_seconds": stage_timer.timings,
            "build_wall_time_seconds": stage_timer.total(),
        }
        graph = GraphDocument(
            schema_version="1.0",
            build_id=build_id,
            nodes=skills,
            edges=all_edges,
            stats=stats,
            config_digest=config_digest,
        )

        _write_graph_artifacts(workspace, graph, edge_evidence_rows, validation_records)
        _write_compiled_graph_artifact(
            workspace,
            graph,
            interfaces,
            canonicalization,
            execution_graph,
            execution_records,
            stats,
            config_digest,
        )
        health = render_health_report(
            graph,
            cache_hits=0,
            llm_validations=int(relation_validation_summary.get("validator_calls", 0)),
            skipped_unchanged=skipped_unchanged,
        )
        atomic_write_text(workspace.graph_dir / "graph_health_report.md", health)
        workspace.write_json(
            workspace.status_path,
            {
                "build_id": build_id,
                "schema_version": graph.schema_version,
                "config_digest": config_digest,
                "skill_count": len(skills),
                "edge_count": len(all_edges),
                "interface_count": stats["interface_count"],
                "interface_accepted_count": stats["interface_accepted_count"],
                "interface_rejected_count": stats["interface_rejected_count"],
                "interface_model_id": stats["interface_model_id"],
                "canonicalization_model_id": stats["canonicalization_model_id"],
                "raw_contract_object_count": stats["raw_contract_object_count"],
                "canonical_object_count": stats["canonical_object_count"],
                "promoted_object_count": stats["promoted_object_count"],
                "raw_artifact_count": stats["raw_artifact_count"],
                "raw_scenario_count": stats["raw_scenario_count"],
                "canonical_artifact_count": stats["canonical_artifact_count"],
                "reusable_state_count": stats["reusable_state_count"],
                "execution_compatibility_count": stats["execution_compatibility_count"],
                "canonical_alias_count": stats["canonical_alias_count"],
                "alias_merge_ratio": stats["alias_merge_ratio"],
                "execution_candidate_count": stats["execution_candidate_count"],
                "execution_accepted_flow_count": stats["execution_accepted_flow_count"],
                "execution_rejected_flow_count": stats["execution_rejected_flow_count"],
                "execution_projected_edge_count": stats["execution_projected_edge_count"],
                "execution_model_id": stats["execution_model_id"],
                "depend_on_cycle_pruned_count": stats["depend_on_cycle_pruned_count"],
                "skill_hashes": {skill.id: skill.content_hash for skill in skills},
                "artifacts": {
                    "graph": str(workspace.graph_dir / "graph.json"),
                    "compiled_skill_graph": str(workspace.graph_dir / "compiled.json"),
                    "edge_evidence": str(workspace.graph_dir / "edge_evidence.jsonl"),
                    "health_report": str(workspace.graph_dir / "graph_health_report.md"),
                    "skill_interfaces": str(workspace.graph_dir / "contracts.jsonl"),
                    "interface_evidence": str(workspace.graph_dir / "interface_evidence.jsonl"),
                    "interface_health_report": str(workspace.graph_dir / "interface_health_report.md"),
                    "canonical_objects": str(workspace.execution_dir / "canonical_objects.jsonl"),
                    "canonical_aliases": str(workspace.execution_dir / "canonical_aliases.jsonl"),
                    "canonicalization_cache": str(workspace.cache_dir / "canonicalization_cache.json"),
                    "canonicalization_health_report": str(workspace.execution_dir / "canonicalization_health_report.md"),
                    "execution_index": str(workspace.execution_dir / "execution_index.jsonl"),
                    "canonicalization_aliases": str(workspace.execution_dir / "canonicalization_aliases.json"),
                    "execution_evidence": str(workspace.execution_dir / "execution_evidence.jsonl"),
                    "execution_health_report": str(workspace.execution_dir / "execution_health_report.md"),
                    "build_metrics": str(workspace.reports_dir / "build_summary.json"),
                },
            },
        )
        _write_build_metrics(workspace, stats, config_digest=config_digest, build_id=build_id)
        workspace.write_json(
            workspace.checkpoint_path,
            {"stage": "complete", "build_id": build_id, "config_digest": config_digest},
        )
    return BuildResult(
        graph=graph,
        skills=skills,
        interfaces=interfaces,
        execution_graph=execution_graph,
        execution_records=execution_records,
        canonicalization=canonicalization,
        workspace=workspace,
        stats=stats,
    )


def load_graph(workspace_root: str | Path = ".skillfabric") -> GraphDocument:
    """Load graph.json from a workspace."""

    path = Path(workspace_root) / "graph" / "graph.json"
    return GraphDocument.from_dict(json.loads(path.read_text(encoding="utf-8")))


def get_skill_neighbors(
    skill_id: str,
    *,
    workspace_root: str | Path = ".skillfabric",
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Query skill neighbors from graph.json."""

    graph = load_graph(workspace_root)
    result = BuildResult(graph=graph, skills=[], workspace=Workspace(workspace_root))
    return result.neighbors(skill_id)[:limit]


def _write_registry(workspace: Workspace, skills: list[SkillNode]) -> None:
    workspace.write_jsonl(
        workspace.graph_dir / "registry.jsonl",
        [skill.to_dict(include_raw_text=True) for skill in skills],
    )
    workspace.write_jsonl(
        workspace.registry_dir / "skill_sources.jsonl",
        [
            {
                "skill_id": skill.id,
                "source_path": skill.source_path,
                "content_hash": skill.content_hash,
                "token_count": skill.token_count,
            }
            for skill in skills
        ],
    )


def _write_graph_artifacts(
    workspace: Workspace,
    graph: GraphDocument,
    edge_evidence_rows: list[dict[str, Any]],
    validation_records: list[Any],
) -> None:
    workspace.write_json(workspace.graph_dir / "graph.json", graph.to_dict())
    workspace.write_jsonl(workspace.graph_dir / "edge_evidence.jsonl", edge_evidence_rows)
    workspace.write_json(
        workspace.graph_dir / "relation_validation_summary.json",
        summarize_relation_validation_records(validation_records),
    )


def _write_interface_artifacts(
    workspace: Workspace,
    records: list[InterfaceExtractionRecord],
) -> None:
    workspace.write_jsonl(
        workspace.graph_dir / "contracts.jsonl",
        [record.interface.to_dict() for record in records],
    )
    workspace.write_jsonl(
        workspace.interfaces_dir / "interface_evidence.jsonl",
        [record.to_record() for record in records],
    )
    health = analyze_interface_health([record.interface for record in records])
    atomic_write_text(
        workspace.interfaces_dir / "interface_health_report.md",
        render_interface_health_report(health),
    )


def _write_canonicalization_artifacts(
    workspace: Workspace,
    build: CanonicalizationBuild,
) -> None:
    workspace.write_jsonl(
        workspace.execution_dir / "canonical_objects.jsonl",
        [item.to_dict() for item in build.objects],
    )
    workspace.write_jsonl(
        workspace.execution_dir / "canonical_aliases.jsonl",
        [item.to_dict() for item in build.assignments],
    )
    workspace.write_json(
        workspace.graph_dir / "canonicalization_aliases.json",
        {
            "schema_version": "1.0",
            "aliases": {item.raw_key: item.canonical_id for item in build.assignments},
            "alias_count": len(build.assignments),
            "canonical_count": len({item.canonical_id for item in build.assignments}),
            "alias_merge_ratio": _canonicalization_alias_merge_ratio(build),
        },
    )
    health = analyze_canonicalization_health(build)
    atomic_write_text(
        workspace.execution_dir / "canonicalization_health_report.md",
        render_canonicalization_health_report(health),
    )


def _write_execution_artifacts(
    workspace: Workspace,
    build: ExecutionGraphBuild,
    records: list[ExecutionValidationRecord],
) -> None:
    _remove_obsolete_execution_artifacts(workspace)
    workspace.write_jsonl(
        workspace.execution_dir / "execution_index.jsonl",
        [record.to_dict() for record in build.execution_index],
    )
    workspace.write_json(
        workspace.execution_dir / "canonicalization_aliases.json",
        {
            "schema_version": "1.0",
            "aliases": build.canonical_aliases,
            "alias_count": len(build.canonical_aliases),
            "canonical_count": len(set(build.canonical_aliases.values())),
            "alias_merge_ratio": _alias_merge_ratio(build.canonical_aliases),
        },
    )
    workspace.write_jsonl(
        workspace.execution_dir / "execution_evidence.jsonl",
        [record.to_record() for record in records],
    )
    workspace.write_json(
        workspace.execution_dir / "execution_validation_summary.json",
        summarize_execution_validation_records(records),
    )
    health = analyze_execution_health(build, records)
    atomic_write_text(
        workspace.execution_dir / "execution_health_report.md",
        render_execution_health_report(health),
    )


def _remove_obsolete_execution_artifacts(workspace: Workspace) -> None:
    for filename in (
        "artifact_nodes.jsonl",
        "scenario_nodes.jsonl",
        "skill_artifact_edges.jsonl",
        "skill_scenario_edges.jsonl",
        "artifact_flows.jsonl",
        "scenario_transitions.jsonl",
        "predicate_interfaces.jsonl",
        "predicate_inventory.json",
        "predicate_aliases.jsonl",
        "predicate_rejections.jsonl",
        "workflow_compatibility.jsonl",
        "semantic_quality_report.md",
        "predicate_cache.json",
    ):
        try:
            (workspace.execution_dir / filename).unlink()
        except FileNotFoundError:
            pass


def _write_compiled_graph_artifact(
    workspace: Workspace,
    graph: GraphDocument,
    interfaces: dict[str, SkillInterface],
    canonicalization: CanonicalizationBuild,
    execution_graph: ExecutionGraphBuild,
    execution_records: list[ExecutionValidationRecord],
    stats: dict[str, Any],
    config_digest: str,
) -> None:
    workspace.write_json(
        workspace.graph_dir / "compiled.json",
        {
            "schema_version": "1.0",
            "core_graph": graph.to_dict(),
            "interfaces": [interfaces[key].to_dict() for key in sorted(interfaces)],
            "canonicalization": {
                "objects": [item.to_dict() for item in canonicalization.objects],
                "aliases": [item.to_dict() for item in canonicalization.assignments],
                "model_id": canonicalization.model_id,
            },
            "execution_graph": {
                "execution_index": [record.to_dict() for record in execution_graph.execution_index],
                "canonicalization": {
                    "aliases": execution_graph.canonical_aliases,
                    "alias_count": len(execution_graph.canonical_aliases),
                    "canonical_count": len(set(execution_graph.canonical_aliases.values())),
                    "alias_merge_ratio": _alias_merge_ratio(execution_graph.canonical_aliases),
                },
                "validated_flows": [record.to_record() for record in execution_records],
                "debug_extraction": {
                    "raw_artifact_nodes": [node.to_dict() for node in execution_graph.raw_artifact_nodes],
                    "raw_scenario_nodes": [node.to_dict() for node in execution_graph.raw_scenario_nodes],
                    "raw_skill_artifact_edges": [
                        edge.to_dict() for edge in execution_graph.raw_skill_artifact_edges
                    ],
                    "raw_skill_scenario_edges": [
                        edge.to_dict() for edge in execution_graph.raw_skill_scenario_edges
                    ],
                },
            },
            "stats": stats,
            "config_digest": config_digest,
        },
    )


def _write_build_metrics(
    workspace: Workspace,
    stats: dict[str, Any],
    *,
    config_digest: str,
    build_id: str,
) -> None:
    usage_log_path = workspace.reports_dir / "llm_usage.jsonl"
    llm_usage = summarize_usage(load_usage_records(usage_log_path)).to_dict() if usage_log_path.exists() else {}
    workspace.write_json(
        workspace.reports_dir / "build_summary.json",
        {
            "schema_version": "1.0",
            "build_id": build_id,
            "config_digest": config_digest,
            "skill_count": stats.get("skill_count", 0),
            "edge_count": stats.get("edge_count", 0),
            "stage_wall_time_seconds": stats.get("stage_wall_time_seconds", {}),
            "build_wall_time_seconds": stats.get("build_wall_time_seconds", 0.0),
            "llm_usage": llm_usage,
            "embedding": stats.get("embedding", {}),
            "execution_validation": stats.get("execution_validation", {}),
            "relation_validation": stats.get("relation_validation", {}),
            "cache": {
                "skipped_unchanged": stats.get("skipped_unchanged", 0),
                "execution_validation_cache_hits": stats.get("execution_validation", {}).get(
                    "cache_hits",
                    0,
                )
                if isinstance(stats.get("execution_validation", {}), dict)
                else 0,
                "relation_validation_cache_hits": stats.get("relation_validation", {}).get(
                    "cache_hits",
                    0,
                )
                if isinstance(stats.get("relation_validation", {}), dict)
                else 0,
            },
            "models": {
                "embedding": stats.get("embedding_model_id", ""),
                "interface": stats.get("interface_model_id", ""),
                "canonicalization": stats.get("canonicalization_model_id", ""),
                "execution": stats.get("execution_model_id", ""),
            },
        },
    )


def _embedding_metrics(
    skills: list[SkillNode],
    embeddings: dict[str, list[float]],
    provider: EmbeddingProvider,
) -> dict[str, Any]:
    vector_count = sum(1 for vector in embeddings.values() if vector)
    estimated_input_tokens = sum(_estimate_tokens(canonical_skill_text(skill)) for skill in skills)
    provider_name = str(getattr(provider, "provider_name", provider.__class__.__name__))
    batch_size = int(getattr(provider, "batch_size", 1) or 1)
    return {
        "provider": provider_name,
        "model_id": provider.model_id,
        "dimension": int(getattr(provider, "dimension", 0) or 0),
        "skill_count": len(skills),
        "vector_count": vector_count,
        "disabled": provider_name == "disabled",
        "estimated_input_tokens": estimated_input_tokens,
        "estimated_calls": 0 if provider_name == "disabled" else (len(skills) + max(batch_size, 1) - 1) // max(batch_size, 1),
    }


def _execution_projected_edge_count(records: list[ExecutionValidationRecord]) -> int:
    return sum(
        1
        for record in records
        if record.accepted
        and str(record.normalized.get("projected_edge_type", "none")) in {"depend_on", "compose_with"}
    )


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 3.5)) if text else 0


def _bm25_neighbor_lookup(
    skills: list[SkillNode],
    bm25_path: Path,
    *,
    limit: int,
) -> dict[str, dict[str, float]]:
    lookup: dict[str, dict[str, float]] = {}
    for skill in skills:
        hits = search_bm25(bm25_path, canonical_skill_text(skill), limit=limit)
        lookup[skill.id] = {
            skill_id: score
            for skill_id, score in hits
            if skill_id != skill.id
        }
    return lookup


def _config_digest(config: BuildConfig) -> str:
    payload = {
        "skill_root": str(config.skill_root),
        "similar_top_k": config.similar_top_k,
        "candidate_top_k": config.candidate_top_k,
        "llm_env_path": str(config.llm_env_path),
        "skip_llm_validation": config.skip_llm_validation,
        "skip_interface_extraction": config.skip_interface_extraction,
        "skip_execution_layer": config.skip_execution_layer,
        "canonicalization_provider": type(config.canonicalization_provider).__name__ if config.canonicalization_provider else "",
        "execution_bucket_limit": config.execution_bucket_limit,
        "llm_concurrency": config.llm_concurrency,
        "llm_rate_limit_per_minute": config.llm_rate_limit_per_minute,
        "llm_max_retries": config.llm_max_retries,
        "llm_retry_backoff_seconds": config.llm_retry_backoff_seconds,
        "llm_progress_every": config.llm_progress_every,
        "llm_batch_size": config.llm_batch_size,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _alias_merge_ratio(aliases: dict[str, str]) -> float:
    if not aliases:
        return 0.0
    canonical_count = len(set(aliases.values()))
    return round(1.0 - (canonical_count / len(aliases)), 6)


def _canonicalization_alias_merge_ratio(build: CanonicalizationBuild) -> float:
    if not build.assignments:
        return 0.0
    canonical_count = len({item.canonical_id for item in build.assignments})
    return round(1.0 - (canonical_count / len(build.assignments)), 6)


def _canonicalization_evidence_rows(build: CanonicalizationBuild) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend({"record_type": "canonical_object", **item.to_dict()} for item in build.objects)
    rows.extend({"record_type": "assignment", **item.to_dict()} for item in build.assignments)
    return rows


def _resolve_pair_validator(config: BuildConfig) -> PairValidator:
    if config.skip_llm_validation:
        return NoopPairValidator()
    if config.validator is not None:
        return config.validator
    return LiteLLMPairValidator.from_env(env_path=config.llm_env_path)


def _resolve_interface_extractor(config: BuildConfig) -> SkillInterfaceExtractor:
    if config.interface_extractor is not None:
        return config.interface_extractor
    if config.skip_llm_validation or config.skip_interface_extraction:
        return DeterministicInterfaceExtractor()
    return LiteLLMInterfaceExtractor.from_env(env_path=config.llm_env_path)


def _resolve_canonicalization_provider(config: BuildConfig) -> CanonicalizationProvider:
    if config.canonicalization_provider is not None:
        return config.canonicalization_provider
    if config.skip_llm_validation or config.skip_execution_layer:
        return DeterministicCanonicalizationProvider()
    return LiteLLMCanonicalizationProvider.from_env(env_path=config.llm_env_path)


def _resolve_execution_validator(config: BuildConfig) -> ExecutionFlowValidator:
    if config.execution_validator is not None:
        return config.execution_validator
    if config.skip_llm_validation or config.skip_execution_layer:
        return DeterministicExecutionFlowValidator()
    return LiteLLMExecutionFlowValidator.from_env(env_path=config.llm_env_path)


def _llm_job_options(config: BuildConfig) -> LLMJobOptions:
    return LLMJobOptions.from_env(
        env_path=config.llm_env_path,
        concurrency=config.llm_concurrency,
        rate_limit_per_minute=config.llm_rate_limit_per_minute,
        max_retries=config.llm_max_retries,
        retry_backoff_seconds=config.llm_retry_backoff_seconds,
        progress_every=config.llm_progress_every,
        batch_size=config.llm_batch_size,
    )


def _neighbor_record(edge: Edge, neighbor_id: str) -> dict[str, Any]:
    return {
        "skill_id": neighbor_id,
        "name": neighbor_id.removeprefix("skill:"),
        "edge_type": edge.type,
        "source": edge.source,
        "target": edge.target,
        "confidence": edge.confidence,
        "weight": edge.weight,
        "provenance": edge.provenance,
        "reason": edge.reason,
    }
