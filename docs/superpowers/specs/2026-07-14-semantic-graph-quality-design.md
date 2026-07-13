# Semantic Graph Quality Design

## Status

Approved direction. This document defines the implementation and evaluation contract for the
semantic graph builder before code changes begin.

## Objective

Improve graph candidate recall, relation precision, and downstream routing quality while keeping
the public implementation small, general, and suitable for AgentSkillOS paper experiments.
Token cost is a secondary constraint: it may remain near the current baseline or increase
slightly when that is necessary for quality, but repeated full-source pair prompts must be
eliminated when an equally accurate grounded representation is available.

## Non-goals

- Do not add contract fields such as `accepts`.
- Do not restore communities, canonicalization layers, compilers, or compatibility fallbacks.
- Do not create graph edges from embedding scores, lexical scores, confidence thresholds, quotas,
  or isolated-node repair rules.
- Do not add benchmark-specific vocabulary, skill names, or task co-occurrence rules to public
  code or prompts.
- Do not preserve obsolete prompt APIs after the replacement is validated.
- Do not expose candidate or request-packing tuning through new CLI options.

## Design Principles

1. Read every complete skill source once to extract a complete, nonredundant operational contract.
2. Retrieve candidates through independent artifact, capability, lexical, and reference views.
3. Treat retrieval as recall only; every edge requires a source-grounded LLM decision.
4. Give the relation judge a complete runtime Skill Profile, not artifact fields alone.
5. Keep persisted contract, relation, graph, router, and wiki schemas unchanged.
6. Use one final production decision path. Experimental alternatives are removed before release.
7. Fail closed on malformed LLM output. Retries repeat the same protocol and never switch to a
   weaker fallback.

## Data Flow

```text
SKILL.md
  -> one full-source contract extraction per skill
  -> complete SkillContract
  -> handoff dense retrieval
     + capability dense retrieval
     + compact-profile BM25 retrieval
     + explicit references
  -> reciprocal-rank fusion with a bounded per-skill review budget
  -> globally deduplicated unordered CandidatePair records
  -> pairwise or bounded-batch relation decisions using runtime Skill Profiles
  -> existing projection and dependency-cycle review
  -> existing graph, router indexes, and wiki artifacts
```

## Contract Extraction

The `SkillContract` schema remains:

- `capability`
- `when_to_use`
- `requires`
- `produces`
- `tools`
- `evidence`

The extraction prompt changes from minimal-contract language to complete-but-nonredundant
language.

`requires` includes every distinct, explicitly supported external artifact, data input, resource,
or execution state consumed or transformed by the skill. Direct caller inputs are valid
requirements even when no other skill must produce them.

`produces` includes every distinct, explicitly supported externally usable artifact, data result,
or execution state. Materially different formats such as Markdown, HTML, screenshots, structured
records, and workbooks remain separate when the source supports them.

Tools, credentials, internal reasoning, temporary implementation state, and generic task intent
do not become inputs or outputs. Task intent remains in `when_to_use`; mechanisms remain in
`tools`.

Every retained field requires exact source evidence. The prompt must not impose a fixed field
count and must not invent cross-skill handoffs.

## Runtime Skill Profile

The relation judge receives one runtime-only profile per involved skill:

```json
{
  "id": "skill:example",
  "name": "Example Skill",
  "description": "Original registry description",
  "capability": "Operational capability",
  "when_to_use": "Selection conditions",
  "requires": [],
  "produces": [],
  "tools": [],
  "source_evidence": []
}
```

This is prompt input, not a new model or artifact schema. `name` and `description` come from the
registry. Contract fields come from the validated contract. `source_evidence` is assembled from
validated contract and explicit-reference evidence, with deterministic adjacent source context
and duplicate lines removed. Output evidence still contains only exact skill IDs and line numbers
and is validated against the original source.

The profile supports all relation types:

- `depend_on`: concrete requirements, products, and execution-state handoffs.
- `compose_with`: capability, selection conditions, workflow stages, products, and tools.
- `similar_to`: objective, operational capability, inputs, outputs, and behavior.
- `none`: a complete view for rejecting topical, tool-only, or incidental overlap.

## Candidate Retrieval

Candidate retrieval uses four channels:

1. `handoff`: dense product-field to requirement-field matching.
2. `similarity`: dense whole-profile matching for capability and operational overlap.
3. `lexical`: BM25 matching from a compact profile query against the existing FTS index.
4. `explicit_reference`: exact canonical IDs or unambiguous referenced skill names.

The lexical channel adds a channel value but no new candidate field. Raw dense and BM25 scores are
not persisted or treated as confidence.

The existing reciprocal-rank fusion implementation moves to a shared indexing module and is used
by both router retrieval and graph candidate retrieval. This avoids duplicate ranking code and
keeps raw scores from incompatible channels separate.

Each skill retains the first eight fused review candidates. Explicit references bypass this
budget because they are exact source observations. Unordered pairs are then globally deduplicated.
The budget controls LLM work only; it does not determine edge existence.

For 53 skills this yields at most 424 directed selections plus explicit references, compared with
1,378 exhaustive pairs. For 1,000 skills it yields at most 8,000 directed selections plus explicit
references, compared with 499,500 exhaustive pairs.

## Relation Decision Protocol

The production relation protocol assigns exactly one of:

- `depend_on`
- `compose_with`
- `similar_to`
- `none`

`depend_on` remains strict. Skill A depends on skill B only when B produces or establishes a
concrete artifact or state that A needs for correct execution or its core purpose. Stored direction
is dependent to prerequisite.

`compose_with` means stable, reusable workflow complementarity across preparation, generation,
transformation, refinement, validation, packaging, or presentation. Co-occurrence, shared domain,
shared tools, and hypothetical usefulness are insufficient.

`similar_to` means near substitution with substantial overlap in objective, capability, inputs,
outputs, and operational behavior.

The prompt does not globally bias the model toward `none`. It states positive and negative tests
for each relation, separates untrusted source data from instructions, requires independent
decisions for listed pairs, and provides one authoritative JSON schema. It requests concise
evidence-grounded reasons, not hidden chain-of-thought.

## Prompt Identity And Cache Invalidation

Prompt identifiers use stable semantic names such as `skill_contract` and
`semantic_relation_judge`; manually incremented suffixes are not used.

Cache keys include a deterministic SHA-256 fingerprint of the rendered fixed prompt policy and
output schema. A policy change therefore invalidates affected cache entries without renaming the
prompt. Skill content hashes, model IDs, and relevant validated contracts remain part of cache
identity.

The fingerprint helper is shared by contract extraction, relation decisions, and cycle review
where appropriate. It hashes trusted fixed prompt content, not task-specific skill payloads.

## Pairwise Versus Bounded Batch Selection

Request packing is an empirical implementation choice, not a graph semantic rule. Before the
production path is finalized, evaluate these protocols on the same fixed labeled pairs and model:

1. Current full-source pairwise judge.
2. Skill Profile pairwise judge.
3. Skill Profile judge with four pairs per request.
4. Skill Profile judge with eight pairs per request.

The production repository retains only the least expensive protocol whose relation precision,
positive recall, and exact-label accuracy do not regress from the strongest pairwise result.
Unused experimental branches, flags, prompt builders, and fixtures are deleted before commit.

Batch output, when selected, is a request-level `decisions` wrapper that is immediately validated
and decomposed into existing pair-level `RelationDecision` records. Every requested pair must
appear exactly once; missing, duplicate, or extra decisions invalidate the response. Pair-level
cache identity remains independent of request packing.

## Error Handling And Resumption

- Invalid contracts, unknown skills, invalid vectors, malformed decisions, incorrect pair
  identities, and unsupported evidence lines fail closed.
- Provider retries repeat the same request with the configured backoff.
- No deterministic edge, alternate prompt, full-source retry, pairwise retry, or reduced-context
  fallback is allowed.
- Valid decisions remain pair-cacheable so a restarted build judges only uncached pairs.
- Cache files retain only schema-valid records and are atomically written.

## Code Scope

Expected production changes are limited to:

- contract prompt and prompt fingerprinting
- compact Skill Profile text construction
- shared reciprocal-rank fusion
- candidate channel fusion and bounded review selection
- relation prompt construction and validation
- builder wiring and metrics already needed to explain build quality
- focused unit and integration tests

The CLI command, graph projection, router expansion, wiki generation, graph edge schema, contract
schema, and public orchestration interfaces remain unchanged. Existing unrelated user changes in
runtime usage accounting are not modified.

## Verification

### Unit and static verification

- Contract prompt retains distinct supported inputs and output formats without inventing fields.
- Lexical retrieval recovers an exact-format candidate missed by dense fixtures.
- Fusion is deterministic, bounded, score-agnostic across channels, and preserves references.
- Candidate pairs are globally unique.
- Relation prompts include complete profiles and grounded source evidence.
- Every requested pair is returned exactly once.
- Invalid evidence, missing pairs, duplicate pairs, and extra pairs fail closed.
- Pair cache reuse is independent of retrieval rank and request packing.
- Existing projection, dependency direction, cycle review, router, and artifact tests pass.
- `compileall`, focused `pytest`, full `pytest`, Ruff, and `git diff --check` pass.

### Real Seeds comparison

Run a fresh build with the same 53 skills, model, embedding provider, concurrency, and wiki mode as
the current baseline. Do not reuse contract or relation caches across prompt protocols.

The fixed evaluation set is stored outside public code and contains source-reviewed positive
relations and hard negatives. Benchmark co-occurrence may identify pairs for human review but is
never treated as a graph label.

Release gates:

- candidate recall on adjudicated positives is at least 90 percent
- accepted-edge precision is at least 90 percent after reviewing every accepted Seeds edge
- hard negatives are not connected
- existing route and execution-prompt checks do not regress
- total build tokens do not exceed the current 8,923,070-token Seeds baseline
- no relation protocol is selected solely because it produces more edges or fewer isolated nodes

### Scale verification

After Seeds passes, freeze prompts and retrieval parameters. Build the 500-skill and 1,000-skill
pools without benchmark-specific retuning. Inspect candidate growth, edge samples by type,
isolated skills, graph connectivity, usage artifacts, and route outputs before running the official
AgentSkillOS SkillNet-Fabric experiment and BT evaluation.

## Release Criteria

The implementation is ready for the public repository and paper experiments only when:

1. The final repository contains one contract path, one candidate path, and one relation path.
2. No compatibility aliases, dead experimental modes, fallback branches, or unused fields remain.
3. All public tests and static checks pass.
4. Real Seeds gates pass with reproducible artifacts and usage accounting.
5. The 500-skill and 1,000-skill builds use the frozen Seeds-approved configuration.
6. The final diff contains no secrets, local caches, temporary outputs, or unrelated changes.
