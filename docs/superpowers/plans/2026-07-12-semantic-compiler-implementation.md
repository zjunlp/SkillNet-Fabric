# Semantic Compiler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development while implementing each task. This checkout is intentionally dirty; do not reset or overwrite unrelated edits, and do not commit without explicit user approval.

**Goal:** Replace the overlapping graph compilers with one evidence-grounded semantic compiler, then make routing, wiki exploration, planning, CLI, and package metadata consume its schema directly.

**Architecture:** A strict `SkillContract` is extracted once per skill. Contract-aware embeddings retrieve bounded handoff, similarity, and explicit-reference candidate pairs through FAISS; one LLM judge assigns exactly one relation or `none` to each pair; a graph projector enforces identity, direction, evidence, uniqueness, and dependency acyclicity. Runtime routing uses BM25 plus embedding reciprocal-rank fusion and bounded traversal over validated operational edges only.

**Tech Stack:** Python 3.11+, dataclasses, SQLite FTS5, FAISS, LiteLLM, Claude Agent SDK, pytest, Ruff.

---

## File Map

- `skillfabric-ai/src/skillfabric/compiled_graph/contracts/`: strict contract model, prompt, extraction, validation, and cache ownership.
- `skillfabric-ai/src/skillfabric/compiled_graph/semantic/`: candidate records, FAISS retrieval, pair prompt/judge, cycle adjudication, and graph projection.
- `skillfabric-ai/src/skillfabric/compiled_graph/builder.py`: stage orchestration and canonical schema-v2 artifact writing only.
- `skillfabric-ai/src/skillfabric/indexing/`: contract-aware retrieval text, BM25 rank retrieval, embedding persistence, and query embedding.
- `skillfabric-ai/src/skillfabric/router/`: RRF seed retrieval, bounded semantic expansion, bundle assembly, explorer validation, and route serialization.
- `skillfabric-ai/src/skillfabric/wiki/`: direct schema-v2 loading and query-local evidence materialization.
- `skillfabric-ai/src/skillfabric/orchestrator/`: planner package preparation/finalization only.
- `skillfabric-ai/src/skillfabric/api.py`, `skillfabric-ai/src/skillfabric/cli.py`: one public workflow with no business-result fallback or dead parameters.
- `plugins/claude-code/skillfabric/`: one shared prepare/run protocol and no community/legacy schema language.

### Task 1: Strict Skill Contracts

**Files:**
- Create: `skillfabric-ai/src/skillfabric/compiled_graph/contracts/__init__.py`
- Create: `skillfabric-ai/src/skillfabric/compiled_graph/contracts/models.py`
- Create: `skillfabric-ai/src/skillfabric/compiled_graph/contracts/prompts.py`
- Create: `skillfabric-ai/src/skillfabric/compiled_graph/contracts/extraction.py`
- Test: `skillfabric-ai/tests/unit/test_contract_extraction.py`
- Delete after migration: `skillfabric-ai/src/skillfabric/compiled_graph/interface/`

- [ ] **Step 1: Write strict model and extraction tests**

```python
def test_contract_extraction_fails_closed_on_invalid_schema(tmp_path):
    with pytest.raises(ContractExtractionError, match="requires must be a list"):
        extract_skill_contracts([skill], extractor=StaticExtractor({"requires": "bad"}), cache_path=tmp_path / "contracts.json")

def test_contract_evidence_line_must_exist_in_source():
    with pytest.raises(ContractSchemaError, match="outside the skill source"):
        SkillContract.from_extraction(skill, payload_with_out_of_range_line)
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n skillfabric_env python -m pytest skillfabric-ai/tests/unit/test_contract_extraction.py -q`

Expected: import failure because `compiled_graph.contracts` does not exist.

- [ ] **Step 3: Implement one strict contract schema**

```python
@dataclass(frozen=True, slots=True)
class SkillContract:
    skill_id: str
    content_hash: str
    capability: str
    when_to_use: str
    requires: tuple[ContractField, ...]
    produces: tuple[ContractField, ...]
    tools: tuple[ContractField, ...]
    evidence: tuple[EvidenceRef, ...]
```

The extractor must parse one JSON object through `runtime/json_utils.py`, validate exact top-level keys and source evidence, write cache entries only after validation, and raise `ContractExtractionError` on API, JSON, evidence, or schema failure. It must not create an empty deterministic contract.

- [ ] **Step 4: Verify GREEN and migrate direct consumers**

Run: `conda run -n skillfabric_env python -m pytest skillfabric-ai/tests/unit/test_contract_extraction.py skillfabric-ai/tests/unit/test_interface_models.py skillfabric-ai/tests/unit/test_interface_cache.py -q`

Expected: all contract tests pass; obsolete interface tests are migrated or removed in the same patch.

### Task 2: Contract-Aware FAISS Candidate Retrieval

**Files:**
- Create: `skillfabric-ai/src/skillfabric/compiled_graph/semantic/__init__.py`
- Create: `skillfabric-ai/src/skillfabric/compiled_graph/semantic/models.py`
- Create: `skillfabric-ai/src/skillfabric/compiled_graph/semantic/candidates.py`
- Modify: `skillfabric-ai/src/skillfabric/indexing/canonical.py`
- Modify: `skillfabric-ai/src/skillfabric/indexing/embeddings.py`
- Modify: `skillfabric-ai/pyproject.toml`
- Test: `skillfabric-ai/tests/unit/test_semantic_candidates.py`

- [ ] **Step 1: Write candidate-channel and pair-deduplication tests**

```python
def test_handoff_ann_queries_produces_against_requires(fake_embedder):
    pairs = retrieve_candidate_pairs(contracts, skills, provider=fake_embedder, handoff_top_k=2, similarity_top_k=1)
    pair = next(item for item in pairs if item.key == ("skill:consumer", "skill:producer"))
    assert pair.channels == ("handoff",)
    assert pair.hits[0].query_field == "produces:normalized_table"
    assert pair.hits[0].matched_field == "requires:normalized_table"

def test_ann_rank_never_materializes_an_edge(fake_embedder):
    assert all(not hasattr(pair, "relation") for pair in retrieve_candidate_pairs(...))
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n skillfabric_env python -m pytest skillfabric-ai/tests/unit/test_semantic_candidates.py -q`

Expected: import failure because semantic candidates are not implemented.

- [ ] **Step 3: Implement normalized inner-product FAISS retrieval**

```python
index = faiss.IndexFlatIP(matrix.shape[1])
faiss.normalize_L2(matrix)
index.add(matrix)
scores, indices = index.search(query_matrix, min(top_k + 1, len(rows)))
```

Embed complete contract-aware skill documents for similarity and individual `produces`/`requires` fields for handoffs. Merge all hits plus explicit references by unordered pair. Keep channel, rank, field ids, and source evidence; discard ANN scores after ranking. Raise a clear error when FAISS is missing or vectors are empty/malformed; provide no brute-force path.

- [ ] **Step 4: Verify GREEN and scale bounds**

Run: `conda run -n skillfabric_env python -m pytest skillfabric-ai/tests/unit/test_semantic_candidates.py skillfabric-ai/tests/unit/test_embeddings.py -q`

Expected: candidate tests pass and candidate count remains bounded by configured top-k values.

### Task 3: One Pair Judge and Cycle Adjudication

**Files:**
- Create: `skillfabric-ai/src/skillfabric/compiled_graph/semantic/prompts.py`
- Create: `skillfabric-ai/src/skillfabric/compiled_graph/semantic/validation.py`
- Create: `skillfabric-ai/src/skillfabric/compiled_graph/semantic/projection.py`
- Test: `skillfabric-ai/tests/unit/test_semantic_validation.py`
- Test: `skillfabric-ai/tests/unit/test_semantic_projection.py`
- Delete after migration: `skillfabric-ai/src/skillfabric/compiled_graph/execution/`
- Delete after migration: `skillfabric-ai/src/skillfabric/compiled_graph/relations/`
- Delete after migration: `skillfabric-ai/src/skillfabric/compiled_graph/canonicalization/`

- [ ] **Step 1: Write relation semantics, direction, evidence, uniqueness, and cycle tests**

```python
def test_dependency_is_stored_dependent_to_prerequisite():
    decision = judge_decision("depend_on", source="skill:consumer", target="skill:producer")
    edge = project_decisions([decision], skills).edges[0]
    assert (edge.source, edge.target) == ("skill:consumer", "skill:producer")

def test_invalid_judge_output_fails_the_build():
    with pytest.raises(RelationValidationError, match="source evidence"):
        validate_candidate_pairs([pair], skills, contracts, validator=invalid_validator)

def test_unresolved_dependency_cycle_fails_closed():
    with pytest.raises(DependencyCycleError):
        project_decisions(cyclic_decisions, skills, cycle_adjudicator=still_cyclic)
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n skillfabric_env python -m pytest skillfabric-ai/tests/unit/test_semantic_validation.py skillfabric-ai/tests/unit/test_semantic_projection.py -q`

Expected: missing semantic judge/projector imports.

- [ ] **Step 3: Implement the full-source pair prompt and strict judge**

```json
{
  "relation": "depend_on|compose_with|similar_to|none",
  "source_skill": "skill:id",
  "target_skill": "skill:id",
  "confidence": 0.0,
  "reason": "evidence-grounded explanation",
  "evidence": [{"skill": "skill:id", "line": 1}]
}
```

The prompt uses XML-delimited fixed policy and untrusted line-numbered sources, defines the four relations once, includes positive/negative/direction examples, and asks for evidence before classification. Models select evidence by skill id and line number; code resolves exact source text for canonical artifacts. Retrieval selects the pair but contributes no score or semantic proof. Every pair gets one final decision; invalid response is an error, while a valid `none` is a persisted rejection.

- [ ] **Step 4: Implement projection and cycle adjudication**

Project at most one edge per unordered pair, canonicalize symmetric endpoints, and remove `weight` and duplicated validator/candidate metadata from edges. On dependency cycles, call the adjudicator with the complete cycle and evidence; reject unresolved or malformed results.

- [ ] **Step 5: Verify GREEN**

Run: `conda run -n skillfabric_env python -m pytest skillfabric-ai/tests/unit/test_semantic_validation.py skillfabric-ai/tests/unit/test_semantic_projection.py skillfabric-ai/tests/unit/test_edge_safety.py -q`

Expected: all relation and cycle invariants pass without confidence-threshold acceptance heuristics.

### Task 4: Schema-v2 Builder and Canonical Artifacts

**Files:**
- Rewrite: `skillfabric-ai/src/skillfabric/compiled_graph/builder.py`
- Modify: `skillfabric-ai/src/skillfabric/compiled_graph/models.py`
- Modify: `skillfabric-ai/src/skillfabric/storage/workspace.py`
- Test: `skillfabric-ai/tests/unit/test_kg_build.py`
- Test: `skillfabric-ai/tests/unit/test_workspace_layout.py`
- Delete: `skillfabric-ai/src/skillfabric/compiled_graph/community_sidecar.py`
- Delete: `skillfabric-ai/tests/unit/test_community_sidecar.py`

- [ ] **Step 1: Replace build tests with schema-v2 behavior**

```python
def test_build_writes_only_canonical_semantic_artifacts(tmp_path):
    result = build_graph(config, dependencies=fixture_dependencies)
    assert result.graph.schema_version == "2.0"
    assert (workspace / "graph/contracts.jsonl").exists()
    assert (workspace / "graph/relation_decisions.jsonl").exists()
    assert not (workspace / "graph/compiled.json").exists()
    assert not (workspace / "graph/communities.json").exists()
    assert all(edge.type in {"depend_on", "compose_with", "similar_to"} for edge in result.graph.edges)
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n skillfabric_env python -m pytest skillfabric-ai/tests/unit/test_kg_build.py skillfabric-ai/tests/unit/test_workspace_layout.py -q`

Expected: old builder writes schema 1 and obsolete artifacts.

- [ ] **Step 3: Rewrite builder as stage orchestration**

Stages are exactly scan, contracts, indexes, candidates, decisions, projection, artifacts. Write `registry.jsonl`, `contracts.jsonl`, `relation_decisions.jsonl`, `graph.json`, `bm25.sqlite`, `embeddings.json`, caches, `build_summary.json`, `llm_usage.jsonl`, and `status.json`. A failed enabled stage writes non-secret stage/error metadata and raises.

- [ ] **Step 4: Verify GREEN and incompatible-workspace rejection**

Run: `conda run -n skillfabric_env python -m pytest skillfabric-ai/tests/unit/test_kg_build.py skillfabric-ai/tests/unit/test_workspace_layout.py skillfabric-ai/tests/unit/test_compiled_graph_architecture.py -q`

Expected: schema-v2 artifact tests pass and an incompatible existing workspace is rejected before mutation.

### Task 5: RRF Retrieval and Bounded Semantic Expansion

**Files:**
- Rewrite: `skillfabric-ai/src/skillfabric/indexing/bm25.py`
- Rewrite: `skillfabric-ai/src/skillfabric/router/retrieval.py`
- Rewrite: `skillfabric-ai/src/skillfabric/router/expansion.py`
- Simplify: `skillfabric-ai/src/skillfabric/router/models.py`
- Simplify: `skillfabric-ai/src/skillfabric/router/bundle.py`
- Simplify: `skillfabric-ai/src/skillfabric/router/assembly.py`
- Simplify: `skillfabric-ai/src/skillfabric/router/route_edges.py`
- Test: `skillfabric-ai/tests/unit/test_bm25_index.py`
- Test: `skillfabric-ai/tests/unit/test_router_bundle.py`
- Test: `skillfabric-ai/tests/unit/test_router_route.py`

- [ ] **Step 1: Write raw-rank, RRF, expansion-path, and no-fallback tests**

```python
def test_bm25_returns_ordered_rank_not_pseudo_confidence(index):
    hits = search_bm25(index, "extract PDF table", limit=3)
    assert [hit.rank for hit in hits] == [1, 2, 3]

def test_expansion_uses_operational_edges_but_not_similarity():
    expanded = expand_semantic_graph(seeds, graph, max_depth=2, limit=20)
    assert expanded["skill:dependency"].introduced_by[0].edge_type == "depend_on"
    assert "skill:alternative" not in expanded
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n skillfabric_env python -m pytest skillfabric-ai/tests/unit/test_bm25_index.py skillfabric-ai/tests/unit/test_router_bundle.py skillfabric-ai/tests/unit/test_router_route.py -q`

Expected: old pseudo-score/PPR/fallback assumptions fail.

- [ ] **Step 3: Implement two-channel RRF and two-hop traversal**

Use FTS5 BM25 order and query-to-contract embedding order as independent channels. Keep the configured top seed budget with no relative-score gate. Traverse `depend_on` and `compose_with` up to two hops, record every introducing path, and expose `similar_to` neighbors only as alternatives.

- [ ] **Step 4: Enforce explorer-only final selection**

Remove deterministic selected-skill fallback, score-based recovery, `explorer_backend=fallback`, and `use_llm_router=False`. Invalid explorer output or compiled/explorer edge conflicts must raise a structured route error and leave a diagnostic artifact.

- [ ] **Step 5: Verify GREEN**

Run: `conda run -n skillfabric_env python -m pytest skillfabric-ai/tests/unit/test_router_bundle.py skillfabric-ai/tests/unit/test_router_route.py skillfabric-ai/tests/unit/test_wiki_explorer.py -q`

Expected: route tests pass with explicit failures and bounded context.

### Task 6: Query Wiki and Planner Contract Cleanup

**Files:**
- Rewrite: `skillfabric-ai/src/skillfabric/wiki/loader.py`
- Simplify: `skillfabric-ai/src/skillfabric/wiki/query_wiki.py`
- Modify: `skillfabric-ai/src/skillfabric/wiki/materializer.py`
- Modify: `skillfabric-ai/src/skillfabric/wiki/explorer/models.py`
- Modify: `skillfabric-ai/src/skillfabric/wiki/explorer/prompting.py`
- Modify: `skillfabric-ai/src/skillfabric/wiki/explorer/validation.py`
- Simplify: `skillfabric-ai/src/skillfabric/orchestrator/package.py`
- Delete: `skillfabric-ai/src/skillfabric/orchestrator/agent_run_spec.py`
- Test: `skillfabric-ai/tests/unit/test_query_wiki.py`
- Test: `skillfabric-ai/tests/unit/test_skill_package.py`
- Test: `skillfabric-ai/tests/unit/test_orchestrator_package.py`

- [ ] **Step 1: Write canonical loader, route-field, and planner isolation tests**

```python
def test_wiki_loader_rejects_schema_one_workspace(tmp_path):
    with pytest.raises(WorkspaceSchemaError, match="rebuild"):
        load_compiled_workspace(tmp_path)

def test_route_preserves_coverage_gaps_through_planner_package(route):
    result = plan_execution_package(workspace, route, query="task")
    payload = json.loads((result.root / "route.json").read_text())
    assert payload["coverage_gaps"] == list(route.coverage_gaps)
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n skillfabric_env python -m pytest skillfabric-ai/tests/unit/test_query_wiki.py skillfabric-ai/tests/unit/test_skill_package.py skillfabric-ai/tests/unit/test_orchestrator_package.py -q`

Expected: loaders probe legacy files and route/package fields do not match the target contract.

- [ ] **Step 3: Implement direct artifact loading and minimal route schema**

The query wiki contains seeds, introducing paths, validated edges, alternatives, contract cards, and bounded source pages. Route output contains only selected skills, dependencies, composition links, ordered ids, near misses, coverage gaps, pages read, rationale, and warnings.

- [ ] **Step 4: Remove duplicate planner outputs and renderer branching**

Prepare one `route.json` plus selected contract/source cards. Finalization validates package-root isolation, selected ids, dependency completeness, and structured planner output, then writes the execution prompt and workflow plan once.

- [ ] **Step 5: Verify GREEN**

Run: `conda run -n skillfabric_env python -m pytest skillfabric-ai/tests/unit/test_query_wiki.py skillfabric-ai/tests/unit/test_skill_package.py skillfabric-ai/tests/unit/test_orchestrator_package.py -q`

Expected: wiki and planner tests pass without legacy probing or dead renderer behavior.

### Task 7: Public API, CLI, Plugin, Dependencies, and Dead Code

**Files:**
- Modify: `skillfabric-ai/src/skillfabric/api.py`
- Modify: `skillfabric-ai/src/skillfabric/cli.py`
- Modify: `skillfabric-ai/src/skillfabric/runtime/defaults.py`
- Modify: `skillfabric-ai/pyproject.toml`
- Modify: `README.md`
- Modify: `skillfabric-ai/README.md`
- Modify: `plugins/claude-code/skillfabric/`
- Test: `skillfabric-ai/tests/unit/test_cli_build_wiki.py`
- Test: `skillfabric-ai/tests/unit/test_cli_route_plan.py`
- Test: `skillfabric-ai/tests/unit/test_public_package.py`

- [ ] **Step 1: Write public-surface tests**

```python
def test_removed_route_fallback_flags_are_rejected(parser):
    with pytest.raises(SystemExit):
        parser.parse_args(["route", "task", "--skip-llm-router"])

def test_public_facade_exposes_one_planner_entrypoint():
    assert callable(SkillFabric.plan)
    assert not hasattr(SkillFabric, "prepare_plan")
    assert not hasattr(SkillFabric, "finalize_plan")
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n skillfabric_env python -m pytest skillfabric-ai/tests/unit/test_cli_build_wiki.py skillfabric-ai/tests/unit/test_cli_route_plan.py skillfabric-ai/tests/unit/test_public_package.py -q`

Expected: old flags, parameters, dependencies, and methods remain exposed.

- [ ] **Step 3: Remove obsolete public surface and dependencies**

Keep `build`, `route`, and one prompt-only `plan`. Remove fallback/profile/router skip switches,
renderer parameters, prepare/finalize compatibility modes, community options, and dependencies
with no import after the redesign. Add `faiss-cpu>=1.8`; retain only runtime dependencies proven
by `rg` and package tests.

- [ ] **Step 4: Consolidate Claude plugin protocol**

Move shared route/plan instructions into one referenced file. Build/run/prepare skills state the
schema-v2 prerequisites and explicit failure behavior once; remove community and legacy artifact
references.

- [ ] **Step 5: Verify GREEN and package contents**

Run: `conda run -n skillfabric_env python -m pytest skillfabric-ai/tests/unit/test_cli_build_wiki.py skillfabric-ai/tests/unit/test_cli_route_plan.py skillfabric-ai/tests/unit/test_public_package.py -q`

Expected: public-surface and package-data tests pass.

### Task 8: Offline Regression and Real Evaluation

**Files:**
- Create or modify: `skillfabric-ai/tests/evaluation/` only for deterministic datasets/metrics; do not store credentials or raw secret-bearing responses.
- Inspect: fresh workspaces under `/private/tmp/skillfabric-*`.

- [ ] **Step 1: Run static and full offline verification**

Run:

```bash
conda run -n skillfabric_env python -m compileall -q skillfabric-ai/src skillfabric-ai/tests
conda run -n skillfabric_env python -m pytest skillfabric-ai/tests -q
conda run -n skillfabric_test python -m ruff check skillfabric-ai/src skillfabric-ai/tests
conda run -n skillfabric_test python -m ruff format --check skillfabric-ai/src skillfabric-ai/tests
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 2: Run real 50-skill build and route evaluation**

Load `/Users/chenjiang/Documents/SkillFabric/Claude/.env` through project configuration without reading or printing it. Use a fresh `/private/tmp/skillfabric-*` workspace. Inspect `contracts.jsonl`, `relation_decisions.jsonl`, `graph.json`, `build_summary.json`, query wiki, route, execution prompt, and usage log. Review every accepted relation, reject cross-domain `similar_to` edges, and verify dependency direction.

- [ ] **Step 3: Run multiple route scenarios**

Exercise direct selection, dependency closure, composition, and close-alternative queries on the same 50-skill graph. Compare required-skill recall and relation quality against the stable Git baseline before comparing LLM calls and token use.

- [ ] **Step 4: Run real Claude SDK route/planner smoke**

Run the integration test with explicit opt-in environment flags and the same `.env` path. Confirm explorer selection, route serialization, planner package isolation, and finalized prompt behavior.

- [ ] **Step 5: Audit final diff**

Run `git status --short`, `git diff --stat`, `git diff --check`, secret filename scan, and artifact/temp-file scan. Confirm every remaining source module is imported or public, every field is consumed, and no generated workspace or credential file is tracked.
