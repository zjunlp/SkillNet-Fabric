# SkillFabric Semantic Compiler Redesign

Date: 2026-07-12
Status: Implemented; live evaluation pending
Scope: `skillfabric-public`

## Context

The current public pipeline can pass its unit suite while producing a graph dominated by
incorrect `similar_to` edges. A live eight-skill build produced 40 `similar_to` edges and
only four operational edges. The root cause is structural: build-time retrieval scores are
being persisted as graph semantics, while interface canonicalization, execution validation,
generic relation validation, and community metadata each add overlapping representations.

The redesign keeps the existing product flow -- scan, build, route, explore, and prepare --
but replaces the overlapping graph compilers with one evidence-grounded semantic compiler.
Quality and end-to-end task success are release gates. Token reduction is accepted only
after those gates pass.

## Goals

- Build a high-precision operational graph whose primary relations are `depend_on` and
  `compose_with`.
- Keep `similar_to` small and evidence-backed instead of materializing nearest neighbors.
- Improve relation candidate recall without all-pairs LLM comparison.
- Preserve high-recall routing and let the explorer make the final precision decision.
- Remove community support, business-result fallbacks, dead code, duplicate schemas, and
  unused public parameters.
- Make every LLM call auditable and useful to a specific extraction or validation decision.
- Support realistic evaluation on 37, 100, and 300-skill collections without running a
  12,000-skill all-pairs build.

## Non-Goals

- Reintroducing community clustering, labels, nodes, pages, or route context.
- Treating a deterministic threshold or lexical heuristic as semantic proof.
- Preserving obsolete workspace schemas without an explicit rebuild.
- Optimizing for minimum token usage before graph and route quality are established.
- Running a full 12,000-skill LLM build during development.

## Architectural Principles

1. Embeddings and BM25 retrieve candidates; they do not create semantic edges.
2. Original skill evidence and an LLM decision create semantic edges.
3. One unordered skill pair receives one final semantic relation.
4. The strongest operational relation wins: `depend_on`, then `compose_with`, then
   `similar_to`, otherwise `none`.
5. Deterministic code may enforce schema, identity, ordering, and graph invariants. It may
   not infer business semantics.
6. Enabled stages fail closed. Explicitly disabled stages are modes, not fallbacks.
7. Runtime artifacts have one canonical owner and no compatibility filename probing.

## Target Data Flow

```text
SKILL.md files
    -> SkillContract extraction
    -> contract-aware BM25 and skill embeddings
    -> field-level ANN candidate retrieval
    -> pair-level semantic LLM validation
    -> cycle adjudication and graph invariants
    -> core semantic graph
    -> query retrieval and bounded semantic expansion
    -> query wiki
    -> explorer-selected route
    -> planner-authored execution prompt
```

## SkillContract

Each skill is extracted once into a compact contract:

```json
{
  "skill_id": "skill:example",
  "content_hash": "...",
  "capability": "...",
  "when_to_use": "...",
  "requires": [],
  "produces": [],
  "tools": [],
  "evidence": []
}
```

`requires` and `produces` contain reusable cross-skill artifacts or execution states.
Commands, libraries, APIs, and implementation mechanisms belong in `tools`. Cognitive
planning state is not an execution handoff unless it is a concrete reusable artifact.

The extractor receives line-numbered `SKILL.md` as untrusted source data. Its prompt uses
clear XML sections for task, semantics, output schema, examples, metadata, and source. The
prompt contains no benchmark-specific world-state vocabulary. An API, parsing, evidence,
or schema failure records the failed extraction and stops the build; no empty interface is
substituted. The model selects evidence by line number; the compiler copies exact source
text into the canonical contract, avoiding redundant output tokens and copy mismatches.

## Candidate Retrieval

The semantic compiler builds three candidate channels:

- Handoff channel: ANN retrieval from each `produces` field to the nearest `requires`
  fields of other skills.
- Similarity channel: ANN retrieval over complete contract-aware skill documents.
- Explicit-reference channel: canonical `skill:` ids and known names inside Markdown
  inline-code spans.

FAISS is the ANN implementation and an explicit package dependency. If FAISS is unavailable
or an index is invalid, the build fails and records the failed stage in `status.json`. There
is no brute-force or lexical fallback.

ANN top-k values are candidate budgets, not acceptance thresholds. Candidate rows retain
their channel, rank, matching contract fields, and source evidence. Vector scores are used
only to rank nearest neighbors and are not persisted or shown to the relation judge. Rows
are grouped by unordered skill pair before validation, so multiple field hits cause one
useful LLM call.

## Semantic Relation Judge

The pair judge receives both semantic SkillContract payloads and the full line-numbered
source for both skills. Candidate retrieval selects the pair but does not contribute proof
or scores to the prompt. The judge returns exactly one strict object:

```json
{
  "relation": "depend_on|compose_with|similar_to|none",
  "source_skill": "skill:id",
  "target_skill": "skill:id",
  "confidence": 0.0,
  "reason": "evidence-grounded explanation",
  "evidence": [
    {"skill": "skill:id", "line": 1}
  ]
}
```

`accepted` is not a separate field because it would duplicate `relation != none`.
`source_skill` and `target_skill` must be the two candidate ids. Evidence lines must exist;
the compiler resolves their exact source text for canonical artifacts. Invalid output is a
failed validation, not `none`.

### Relation Semantics

`A depend_on B` means A is the dependent and B is the prerequisite. B must produce or
establish a concrete artifact or execution state that A requires for correct execution or
for its core purpose. Route execution order is therefore B before A.

`compose_with` is a symmetric relation between distinct complementary capabilities that
provide material combined workflow value without a strict prerequisite. Endpoints are
stored in canonical id order. It supports candidate expansion and composition context but
does not become a hard before/after constraint.

`similar_to` is a symmetric near-substitute relation. The skills must substantially overlap
in objective, operational capability, and input/output behavior. Shared domain, shared
tools, or both being useful in a broad workflow is insufficient. Similarity edges are used
to compare alternatives and near misses, not to traverse dependency workflows.

`none` covers topical overlap, shared tools, alternatives without sufficient overlap,
generic objects, wrong direction, local-only artifacts, duplicate candidate evidence, and
unsupported inference.

## Graph Invariants

- Only `skill` nodes are persisted.
- Edge types are exactly `depend_on`, `compose_with`, and `similar_to`.
- At most one semantic edge exists per unordered skill pair.
- `compose_with` and `similar_to` endpoints use canonical id order.
- `depend_on` uses dependent -> prerequisite direction.
- Confidence is stored once. The existing duplicate `weight` field is removed.
- Every edge has one validated decision, confidence, reason, and source evidence.
- `depend_on` must be acyclic.

If accepted dependency decisions form a cycle, a cycle adjudication LLM call reviews the
complete cycle and its original evidence. It may reclassify or reject individual decisions.
If the reviewed result remains cyclic or invalid, the build fails and writes the unresolved
cycle artifact. No weakest-edge pruning is used.

## Canonical Artifacts

The target workspace schema is version 2 and contains:

```text
graph/registry.jsonl
graph/contracts.jsonl
graph/relation_decisions.jsonl
graph/graph.json
graph/bm25.sqlite
graph/embeddings.json
cache/contracts.json
cache/relation_decisions.json
reports/build_summary.json
reports/llm_usage.jsonl
status.json
```

`relation_decisions.jsonl` contains validated accepted and rejected pair decisions with
candidate evidence. A failed decision stops the build and is reported in `status.json`.
`graph.json` contains only accepted semantic edges. The following duplicate or obsolete
artifacts are removed:

- canonical objects and canonical assignments
- canonicalization aliases and execution aliases
- raw artifact nodes and raw scenario nodes
- skill-artifact and skill-scenario edges
- execution compatibility index and projected execution records
- community artifacts and caches
- generic relation and execution relation records that represent the same pair twice

The wiki loader reads canonical artifacts directly. `compiled.json` is removed unless a
public compatibility audit finds a documented external consumer before implementation.
Old workspaces are rejected before mutation and require a new workspace; filenames are not
probed for legacy alternatives.

## Retrieval and Routing

Contract extraction completes before retrieval indexing. The indexed skill document combines
name, description, capability, selection conditions, requires, produces, tools, and bounded
source text. Query routing uses two independent channels:

- SQLite FTS5 BM25 rank
- query-to-skill embedding rank

Reciprocal rank fusion combines channel order. BM25 rank is not converted into the current
near-constant pseudo-confidence. The hand-written query stemmer, duplicate stop-word list,
interface term-overlap channel, canonical-object token channel, and execution-object token
channel are removed.

Routing takes the configured top seed budget without the current 45 percent relative-score
gate. It performs bounded two-hop expansion over `depend_on` and `compose_with` and records
the exact path that introduced every candidate. It does not use `similar_to` as workflow
propagation. Similar skills are exposed separately as alternatives. The expanded budget is
a context bound, not semantic rejection.

The explorer remains the final selector. It reads a query-local wiki containing seeds,
semantic expansion paths, validated edges, skill cards, and bounded source pages.

## Route Contract

The route-time structured result contains only fields consumed downstream:

```text
selected_skills
relation_evidence
near_misses
coverage_gaps
wiki_pages_read
rationale
```

Relation evidence projects validated `depend_on` and `compose_with` edges whose endpoints are
both selected. It informs the prompt planner but does not force selection, prerequisite closure,
or execution order. Coverage gaps survive validation, RouteResult serialization, and planner
packaging. No field is generated and then discarded. Public route schemas no longer expose
`artifact_compatibility` or `state_compatibility`.

Invalid explorer output remains a validation error. The router does not synthesize a route or
select top-scoring candidates after explorer failure.

## Failure Semantics

The following business-result fallbacks are removed:

- deterministic route selection
- `explorer_backend=fallback`
- `--skip-llm-router`
- empty deterministic SkillContract generation
- deterministic wiki summary after an enabled LLM summary failure
- unknown expansion mode -> PPR
- invalid explorer package -> score-based route
- legacy artifact filename probing

Contract-derived wiki summaries are an explicit documented mode. If an LLM-backed stage is
enabled, its infrastructure or schema failure stops the workflow and produces a non-secret
error artifact.

## Wiki and Planner

Community content is absent from the main wiki, query wiki, prompts, schemas, CLI, package
data, and tests. Skill pages are grounded in validated SkillContracts and source evidence.
Workflow pages are rendered from accepted semantic relations.

LLM wiki summaries remain enabled during the first correctness comparison because they may
improve route decisions. Their deterministic failure fallback is removed. A contract-only
versus LLM-summary route ablation decides whether the summary stage is useful. The stage is
deleted only if route quality is non-inferior; otherwise it remains a first-class, fail-closed
stage.

The unused `agent_run_spec.py` hierarchy is removed. The `renderer` argument is removed
because Claude Code and Codex currently receive identical output. `SkillFabric.plan()`, which
always raises, is removed. Planner packaging keeps one `route.json` and selected skill cards
instead of writing multiple route subsets. Planner validation checks structure, package path
isolation, selected ids, and dependency completeness; it does not use task-specific keyword
blacklists.

The Claude plugin moves the shared preparation protocol into one reference file used by both
prepare and run skills. Community references and duplicated 17-step instructions are removed.

## Prompt Standard

All semantic prompts follow these rules:

- Direct role and task statement.
- XML separation of fixed instructions and untrusted source data.
- One authoritative relation-semantics definition shared by build, route, and planner code.
- Strict structured output schema.
- A short decision procedure that asks for evidence before classification.
- A small set of positive, negative, and direction examples.
- No benchmark, fixture, project-name, or model-specific heuristics.
- No repeated objective/rules/constraints sections expressing the same policy.
- Prompt changes require behavioral evaluation, not keyword-presence tests alone.

Full-source pair validation is the initial quality baseline. Compact evidence and full-context
escalation may be introduced only through a measured ablation that preserves candidate and
edge recall.

## Code Removal and Consolidation

The implementation removes or replaces:

- dirty community sidecar code and tests
- dirty generic relation code and tests
- `compiled_graph/canonicalization`
- raw execution artifact/scenario models and compiler paths
- materialized KNN similarity builder
- unused AgentRunSpec code
- deterministic interface and route fallbacks
- dead query-wiki parsers and route-edge helpers
- duplicate LLM response and JSON parsers
- unused `graspologic`, `networkx`, `rapidfuzz`, and `python-dotenv` dependencies
- obsolete public flags and parameters

Shared response parsing lives in `runtime/json_utils.py`. API and CLI call one workspace-build
workflow rather than independently assembling graph and wiki stages. Manual edits use the
existing project style; new abstractions are introduced only where they delete duplicated
ownership.

## Evaluation Plan

### Datasets

- 8-skill public fixture for smoke behavior and known PDF, WebShop, and CI chains.
- 37-skill ALFWorld set for multi-step state and ordering behavior.
- 100-skill mixed-domain gold set with positive dependencies, compositions, similarities,
  alternatives, and hard negatives.
- 300-skill scale set for ANN candidate recall, candidate volume, memory, and runtime.
- A sampled collection from `Data/Skills_1w` for cross-domain hard negatives.

### Metrics

- candidate pair recall
- relation precision, recall, and F1 by edge type
- dependency direction accuracy
- false cross-domain `similar_to` count
- route selected-skill precision and recall
- required-dependency accuracy
- end-to-end task success
- LLM calls and tokens per skill, candidate, accepted edge, and route
- build wall time, cache hits, and explicit failure rate

### Release Gates

- Candidate recall is at least 0.95 on the reviewed gold set.
- Semantic edge precision is at least 0.90.
- Dependency direction is correct for every reviewed accepted dependency.
- Known fixture and ALFWorld required chains have no regression.
- Route required-skill recall and end-to-end task success are no worse than the stable Git
  baseline on the same queries.
- No known cross-domain fixture pair is accepted as `similar_to`.
- All enabled-stage failures are explicit and leave diagnostic artifacts.
- Token reductions are reported only after the quality gates pass.

## Verification Sequence

1. Compile and lint changed source and tests.
2. Run focused unit tests after each module replacement.
3. Run the complete offline suite.
4. Build a reviewed 50-skill mixed-domain set through real APIs.
5. Run multiple routes that exercise direct selection, hard dependencies, composition, and
   close alternatives.
6. Review every accepted edge and compare relation quality with the stable Git baseline.
7. Run real Claude Code SDK route and planner flows where the configured SDK endpoint supports
   them.
8. Inspect graph, relation decisions, route, prompt, usage, and failure artifacts.
9. Check package contents, dependencies, secret exclusions, and fresh-install behavior.

Real API runs load configuration from
`/Users/chenjiang/Documents/SkillFabric/Claude/.env` without printing or copying values. Each
run uses a fresh workspace and records exact usage and cost. A full 12,000-skill LLM build is
outside this implementation cycle.

## Compatibility and Rollout

This is a schema-versioned alpha redesign. Existing workspaces must be rebuilt. Public docs,
CLI help, Python API, Claude plugin instructions, package metadata, and tests change in the
same release. There is no hidden compatibility fallback.

Implementation proceeds in vertical slices: contract failure semantics, semantic candidate
retrieval, relation validation, graph projection, router/query wiki, planner cleanup, and
package cleanup. Each slice must preserve a runnable test loop before the next begins.
