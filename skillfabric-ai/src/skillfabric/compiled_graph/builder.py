"""KG build pipeline orchestration."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from skillfabric.compiled_graph.canonicalization.candidates import (
    DEFAULT_MAX_GROUP_SIZE,
    DEFAULT_SEMANTIC_THRESHOLD,
    DEFAULT_SEMANTIC_TOP_K,
    EmbeddingProviderCanonicalEmbedder,
)
from skillfabric.compiled_graph.canonicalization.compiler import (
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
from skillfabric.compiled_graph.canonicalization.prompts import CANONICALIZATION_PROMPT_ID
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
from skillfabric.compiled_graph.execution.policy import (
    EXECUTION_POLICY_DIGEST,
    EXECUTION_POLICY_VERSION,
)
from skillfabric.compiled_graph.execution.projection import project_execution_records
from skillfabric.compiled_graph.execution.prompts import (
    COMPACT_EXECUTION_PROMPT_ID,
    EXECUTION_PROMPT_ID,
)
from skillfabric.compiled_graph.execution.validation import (
    ExecutionFlowValidator,
    LiteLLMExecutionFlowValidator,
    summarize_execution_validation_records,
    validate_execution_flow_candidates,
)
from skillfabric.compiled_graph.health import render_health_report
from skillfabric.compiled_graph.interface.extraction import (
    LiteLLMInterfaceExtractor,
    SkillInterfaceExtractor,
    extract_skill_interfaces,
)
from skillfabric.compiled_graph.interface.health import (
    analyze_interface_health,
    render_interface_health_report,
)
from skillfabric.compiled_graph.interface.models import InterfaceExtractionRecord, SkillInterface
from skillfabric.compiled_graph.interface.prompts import INTERFACE_PROMPT_ID
from skillfabric.compiled_graph.models import Edge, GraphDocument
from skillfabric.compiled_graph.relations.edge_safety import enforce_depend_on_acyclicity
from skillfabric.compiled_graph.relations.similarity import build_similar_edges
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

DEFAULT_SIMILAR_TOP_K = 5
DEFAULT_EXECUTION_BUCKET_LIMIT = 100


@dataclass(slots=True)
class BuildConfig:
    """Configuration for a KG build."""

    skill_root: str | Path
    workspace: str | Path = ".skillfabric"
    llm_env_path: str | Path = ".env"
    llm_options: LLMJobOptions | None = None


@dataclass(slots=True)
class _BuildDependencies:
    """Private dependency overrides for tests and internal adapters."""

    execution_validator: ExecutionFlowValidator | None = None
    interface_extractor: SkillInterfaceExtractor | None = None
    canonicalization_provider: CanonicalizationProvider | None = None
    embedding_provider: EmbeddingProvider | None = None
    build_id: str | None = None


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


def build_graph(config: BuildConfig, *, dependencies: _BuildDependencies | None = None) -> BuildResult:
    """Run the full offline KG build pipeline."""

    deps = dependencies or _BuildDependencies()
    workspace = Workspace(config.workspace)
    build_id = deps.build_id or str(int(time.time()))
    embedding_provider = deps.embedding_provider or default_embedding_provider(env_path=config.llm_env_path)
    llm_job_options = _llm_job_options(config)
    stage_timer = _StageTimer()
    usage_log_path = workspace.reports_dir / "llm_usage.jsonl"
    workspace.ensure()

    with workspace.lock(), llm_usage_context(log_path=usage_log_path, metadata={"build_id": build_id}):
        interface_extractor = _resolve_interface_extractor(config, deps)
        execution_validator = _resolve_execution_validator(config, deps)
        canonicalization_provider = _resolve_canonicalization_provider(config, deps)
        config_digest = _config_digest(
            config,
            embedding_provider=embedding_provider,
            interface_extractor=interface_extractor,
            canonicalization_provider=canonicalization_provider,
            execution_validator=execution_validator,
            llm_job_options=llm_job_options,
        )
        workspace.write_json(
            workspace.checkpoint_path,
            {"stage": "scan", "build_id": build_id, "config_digest": config_digest},
        )
        old_status = workspace.read_json(workspace.status_path, default={}) or {}
        old_hashes = old_status.get("skill_hashes", {}) if isinstance(old_status, dict) else {}
        skills = scan_and_parse(config.skill_root)
        skipped_unchanged = sum(1 for skill in skills if old_hashes.get(skill.id) == skill.content_hash)
        _write_registry(workspace, skills)
        stage_timer.mark("scan")

        workspace.write_json(
            workspace.checkpoint_path,
            {"stage": "index", "build_id": build_id, "config_digest": config_digest},
        )
        bm25_path = workspace.graph_dir / "bm25.sqlite"
        build_bm25_index(skills, bm25_path)
        embeddings = build_embedding_store(
            skills,
            workspace.graph_dir / "embeddings.json",
            provider=embedding_provider,
        )
        _remove_obsolete_embedding_artifacts(workspace)
        embedding_metrics = _embedding_metrics(skills, embeddings, embedding_provider)
        stage_timer.mark("index")

        workspace.write_json(
            workspace.checkpoint_path,
            {"stage": "similarity", "build_id": build_id, "config_digest": config_digest},
        )
        neighbor_scores = build_neighbor_scores(
            skills,
            embeddings,
            top_k=DEFAULT_SIMILAR_TOP_K,
            bm25_neighbors=_bm25_neighbor_lookup(
                skills,
                bm25_path,
                limit=max(DEFAULT_SIMILAR_TOP_K * 4, 20),
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
        execution_graph = compile_execution_graph(
            interfaces,
            bucket_limit=DEFAULT_EXECUTION_BUCKET_LIMIT,
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

        all_edges: list[Edge] = []
        all_edges.extend(similar_edges)
        all_edges = project_execution_records(execution_records, all_edges)
        edge_safety = enforce_depend_on_acyclicity(all_edges)
        all_edges = edge_safety.edges
        stats = {
            "skill_count": len(skills),
            "edge_count": len(all_edges),
            "similar_to_count": len(similar_edges),
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
            "canonical_assignment_count": len(canonicalization.assignments),
            "canonicalization_cluster_count": canonicalization.cluster_count,
            "canonicalization_llm_call_count": canonicalization.llm_call_count,
            "canonicalization_omitted_term_count": canonicalization.omitted_term_count,
            "canonicalization_cache_hit_count": canonicalization.cache_hit_count,
            "semantic_threshold": DEFAULT_SEMANTIC_THRESHOLD,
            "semantic_top_k": DEFAULT_SEMANTIC_TOP_K,
            "max_group_size": DEFAULT_MAX_GROUP_SIZE,
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

        _write_graph_artifacts(workspace, graph)
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
            llm_validations=int(execution_validation_summary.get("validator_calls", 0)),
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
                "embedding_model_id": stats["embedding_model_id"],
                "interface_model_id": stats["interface_model_id"],
                "canonicalization_model_id": stats["canonicalization_model_id"],
                "raw_contract_object_count": stats["raw_contract_object_count"],
                "canonical_object_count": stats["canonical_object_count"],
                "canonical_assignment_count": stats["canonical_assignment_count"],
                "canonicalization_cluster_count": stats["canonicalization_cluster_count"],
                "canonicalization_llm_call_count": stats["canonicalization_llm_call_count"],
                "canonicalization_omitted_term_count": stats["canonicalization_omitted_term_count"],
                "canonicalization_cache_hit_count": stats["canonicalization_cache_hit_count"],
                "semantic_threshold": stats["semantic_threshold"],
                "semantic_top_k": stats["semantic_top_k"],
                "max_group_size": stats["max_group_size"],
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
                    "health_report": str(workspace.graph_dir / "graph_health_report.md"),
                    "skill_interfaces": str(workspace.graph_dir / "contracts.jsonl"),
                    "interface_evidence": str(workspace.graph_dir / "interface_evidence.jsonl"),
                    "interface_health_report": str(workspace.graph_dir / "interface_health_report.md"),
                    "canonical_objects": str(workspace.graph_dir / "canonical_objects.jsonl"),
                    "canonical_aliases": str(workspace.graph_dir / "canonical_aliases.jsonl"),
                    "canonicalization_aliases": str(workspace.graph_dir / "canonicalization_aliases.json"),
                    "canonicalization_cache": str(workspace.cache_dir / "canonicalization_cache.json"),
                    "canonicalization_health_report": str(workspace.graph_dir / "canonicalization_health_report.md"),
                    "execution_index": str(workspace.graph_dir / "execution_index.jsonl"),
                    "execution_aliases": str(workspace.graph_dir / "execution_aliases.json"),
                    "execution_evidence": str(workspace.graph_dir / "execution_evidence.jsonl"),
                    "execution_health_report": str(workspace.graph_dir / "execution_health_report.md"),
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


def _write_graph_artifacts(
    workspace: Workspace,
    graph: GraphDocument,
) -> None:
    _remove_obsolete_relation_artifacts(workspace)
    workspace.write_json(workspace.graph_dir / "graph.json", graph.to_dict())


def _write_interface_artifacts(
    workspace: Workspace,
    records: list[InterfaceExtractionRecord],
) -> None:
    workspace.write_jsonl(
        workspace.graph_dir / "contracts.jsonl",
        [record.interface.to_dict() for record in records],
    )
    workspace.write_jsonl(
        workspace.graph_dir / "interface_evidence.jsonl",
        [record.to_record() for record in records],
    )
    health = analyze_interface_health([record.interface for record in records])
    atomic_write_text(
        workspace.graph_dir / "interface_health_report.md",
        render_interface_health_report(health),
    )


def _write_canonicalization_artifacts(
    workspace: Workspace,
    build: CanonicalizationBuild,
) -> None:
    workspace.write_jsonl(
        workspace.graph_dir / "canonical_objects.jsonl",
        [item.to_dict() for item in build.objects],
    )
    workspace.write_jsonl(
        workspace.graph_dir / "canonical_aliases.jsonl",
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
        workspace.graph_dir / "canonicalization_health_report.md",
        render_canonicalization_health_report(health),
    )


def _write_execution_artifacts(
    workspace: Workspace,
    build: ExecutionGraphBuild,
    records: list[ExecutionValidationRecord],
) -> None:
    _remove_obsolete_execution_artifacts(workspace)
    workspace.write_jsonl(
        workspace.graph_dir / "execution_index.jsonl",
        [record.to_dict() for record in build.execution_index],
    )
    workspace.write_json(
        workspace.graph_dir / "execution_aliases.json",
        {
            "schema_version": "1.0",
            "aliases": build.canonical_aliases,
            "alias_count": len(build.canonical_aliases),
            "canonical_count": len(set(build.canonical_aliases.values())),
            "alias_merge_ratio": _alias_merge_ratio(build.canonical_aliases),
        },
    )
    workspace.write_jsonl(
        workspace.graph_dir / "execution_evidence.jsonl",
        [record.to_record() for record in records],
    )
    workspace.write_json(
        workspace.graph_dir / "execution_validation_summary.json",
        summarize_execution_validation_records(records),
    )
    health = analyze_execution_health(build, records)
    atomic_write_text(
        workspace.graph_dir / "execution_health_report.md",
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
            (workspace.graph_dir / filename).unlink()
        except FileNotFoundError:
            pass


def _remove_obsolete_relation_artifacts(workspace: Workspace) -> None:
    for filename in (
        "edge_evidence.jsonl",
        "relation_validation_audit.jsonl",
        "relation_validation_summary.json",
    ):
        try:
            (workspace.graph_dir / filename).unlink()
        except FileNotFoundError:
            pass
    try:
        (workspace.cache_dir / "relation_validation_cache.json").unlink()
    except FileNotFoundError:
        pass


def _remove_obsolete_embedding_artifacts(workspace: Workspace) -> None:
    try:
        (workspace.graph_dir / "embedding_meta.jsonl").unlink()
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
            "canonicalization": {
                "cluster_count": stats.get("canonicalization_cluster_count", 0),
                "llm_call_count": stats.get("canonicalization_llm_call_count", 0),
                "assignment_count": stats.get("canonical_assignment_count", 0),
                "omitted_term_count": stats.get("canonicalization_omitted_term_count", 0),
                "cache_hit_count": stats.get("canonicalization_cache_hit_count", 0),
                "semantic_threshold": stats.get("semantic_threshold", DEFAULT_SEMANTIC_THRESHOLD),
                "semantic_top_k": stats.get("semantic_top_k", DEFAULT_SEMANTIC_TOP_K),
                "max_group_size": stats.get("max_group_size", DEFAULT_MAX_GROUP_SIZE),
            },
            "execution_validation": stats.get("execution_validation", {}),
            "cache": {
                "skipped_unchanged": stats.get("skipped_unchanged", 0),
                "execution_validation_cache_hits": stats.get("execution_validation", {}).get(
                    "cache_hits",
                    0,
                )
                if isinstance(stats.get("execution_validation", {}), dict)
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


def _config_digest(
    config: BuildConfig,
    *,
    embedding_provider: EmbeddingProvider,
    interface_extractor: SkillInterfaceExtractor,
    canonicalization_provider: CanonicalizationProvider,
    execution_validator: ExecutionFlowValidator,
    llm_job_options: LLMJobOptions,
) -> str:
    payload = {
        "schema_version": "2.0",
        "skill_root": str(config.skill_root),
        "llm_env_path": str(config.llm_env_path),
        "retrieval": {
            "similar_top_k": DEFAULT_SIMILAR_TOP_K,
            "embedding": _provider_fingerprint(embedding_provider),
        },
        "interface_extraction": {
            "extractor": _provider_fingerprint(interface_extractor),
            "prompt_id": INTERFACE_PROMPT_ID,
        },
        "canonicalization": {
            "provider": _provider_fingerprint(canonicalization_provider),
            "prompt_id": CANONICALIZATION_PROMPT_ID,
            "semantic_embedder": _provider_fingerprint(embedding_provider),
            "semantic_threshold": DEFAULT_SEMANTIC_THRESHOLD,
            "semantic_top_k": DEFAULT_SEMANTIC_TOP_K,
            "max_group_size": DEFAULT_MAX_GROUP_SIZE,
        },
        "execution": {
            "validator": _provider_fingerprint(execution_validator),
            "bucket_limit": DEFAULT_EXECUTION_BUCKET_LIMIT,
            "policy_version": EXECUTION_POLICY_VERSION,
            "policy_digest": EXECUTION_POLICY_DIGEST,
            "prompt_id": EXECUTION_PROMPT_ID,
            "compact_prompt_id": COMPACT_EXECUTION_PROMPT_ID,
        },
        "llm_jobs": _llm_job_options_fingerprint(llm_job_options),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _provider_fingerprint(provider: object) -> dict[str, Any]:
    data: dict[str, Any] = {
        "class": provider.__class__.__name__,
        "model_id": str(getattr(provider, "model_id", "")),
    }
    provider_name = getattr(provider, "provider_name", None)
    if provider_name is not None:
        data["provider_name"] = str(provider_name)
    dimension = getattr(provider, "dimension", None)
    if dimension is not None:
        data["dimension"] = int(dimension or 0)
    batch_size = getattr(provider, "batch_size", None)
    if batch_size is not None:
        data["batch_size"] = int(batch_size or 0)
    max_text_chars = getattr(provider, "max_text_chars", None)
    if max_text_chars is not None:
        data["max_text_chars"] = int(max_text_chars or 0)
    timeout = getattr(provider, "timeout", None)
    if timeout is not None:
        data["timeout"] = float(timeout or 0.0)
    return data


def _llm_job_options_fingerprint(options: LLMJobOptions) -> dict[str, int | float | None]:
    normalized = options.normalized()
    return {
        "concurrency": normalized.concurrency,
        "rate_limit_per_minute": normalized.rate_limit_per_minute,
        "max_retries": normalized.max_retries,
        "retry_backoff_seconds": normalized.retry_backoff_seconds,
        "progress_every": normalized.progress_every,
        "batch_size": normalized.batch_size,
    }


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


def _resolve_interface_extractor(
    config: BuildConfig,
    dependencies: _BuildDependencies,
) -> SkillInterfaceExtractor:
    if dependencies.interface_extractor is not None:
        return dependencies.interface_extractor
    return LiteLLMInterfaceExtractor.from_env(env_path=config.llm_env_path)


def _resolve_canonicalization_provider(
    config: BuildConfig,
    dependencies: _BuildDependencies,
) -> CanonicalizationProvider:
    if dependencies.canonicalization_provider is not None:
        return dependencies.canonicalization_provider
    return LiteLLMCanonicalizationProvider.from_env(env_path=config.llm_env_path)


def _resolve_execution_validator(
    config: BuildConfig,
    dependencies: _BuildDependencies,
) -> ExecutionFlowValidator:
    if dependencies.execution_validator is not None:
        return dependencies.execution_validator
    return LiteLLMExecutionFlowValidator.from_env(env_path=config.llm_env_path)


def _llm_job_options(config: BuildConfig) -> LLMJobOptions:
    if config.llm_options is not None:
        return config.llm_options
    return LLMJobOptions.from_env(
        env_path=config.llm_env_path,
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
