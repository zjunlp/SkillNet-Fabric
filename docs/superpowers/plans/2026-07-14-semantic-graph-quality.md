# Semantic Graph Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a high-recall, source-grounded semantic graph pipeline that preserves relation accuracy, uses stable prompt names, and is ready for reproducible AgentSkillOS experiments.

**Architecture:** Extract a complete nonredundant contract once per skill, retrieve bounded candidates through dense handoff, dense capability, BM25, and explicit-reference views, then judge candidates from complete runtime Skill Profiles. Prompt fingerprints invalidate caches without manually versioned names; retrieval never creates edges, and only one validated relation path remains in the public code.

**Tech Stack:** Python 3.11, dataclasses, SQLite FTS5, FAISS, LiteLLM, pytest, Ruff.

---

## File Structure

- Create `skillfabric-ai/src/skillfabric/runtime/prompting.py` for prompt fingerprints.
- Create `skillfabric-ai/src/skillfabric/indexing/ranking.py` for shared rank fusion.
- Modify contract prompt and extraction modules for complete contracts and cache identity.
- Modify canonical indexing text, router retrieval, and semantic candidates for hybrid retrieval.
- Modify semantic prompts and validation for Skill Profiles and exact multi-pair decisions.
- Modify graph builder wiring and focused tests.
- Rename remaining numeric prompt constants in wiki and orchestration modules.

### Task 1: Prompt Policy Fingerprints

**Files:**
- Create: `skillfabric-ai/src/skillfabric/runtime/prompting.py`
- Create: `skillfabric-ai/tests/unit/test_prompting.py`

- [ ] **Step 1: Write failing tests**

```python
from skillfabric.runtime.prompting import prompt_fingerprint


def test_prompt_fingerprint_is_stable_for_mapping_order() -> None:
    first = prompt_fingerprint("skill_contract", {"b": 2, "a": 1})
    second = prompt_fingerprint("skill_contract", {"a": 1, "b": 2})
    assert first == second
    assert len(first) == 64


def test_prompt_fingerprint_changes_with_policy() -> None:
    assert prompt_fingerprint("judge", "strict") != prompt_fingerprint("judge", "complete")
```

- [ ] **Step 2: Verify failure**

Run from `skillfabric-ai`:

```bash
/Users/chenjiang/Documents/Develop/Miniforge/envs/skillfabric_env/bin/python -m pytest tests/unit/test_prompting.py -q
```

Expected: collection fails because the module does not exist.

- [ ] **Step 3: Implement the helper**

```python
"""Stable identity for trusted prompt policy and output schemas."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def prompt_fingerprint(prompt_name: str, *policy: Any) -> str:
    """Hash a stable prompt name and its trusted policy content."""

    if not prompt_name.strip():
        raise ValueError("prompt_name must not be empty")
    encoded = json.dumps(
        [prompt_name, *policy],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
```

- [ ] **Step 4: Verify and commit**

```bash
/Users/chenjiang/Documents/Develop/Miniforge/envs/skillfabric_env/bin/python -m pytest tests/unit/test_prompting.py -q
/Users/chenjiang/Documents/Develop/Miniforge/envs/skillfabric_env/bin/python -m ruff check src/skillfabric/runtime/prompting.py tests/unit/test_prompting.py
git add skillfabric-ai/src/skillfabric/runtime/prompting.py skillfabric-ai/tests/unit/test_prompting.py
git commit -m "refactor: fingerprint prompt policies"
```

Expected: tests and Ruff pass before commit.

### Task 2: Complete Skill Contracts

**Files:**
- Modify: `skillfabric-ai/src/skillfabric/compiled_graph/contracts/prompts.py`
- Modify: `skillfabric-ai/src/skillfabric/compiled_graph/contracts/extraction.py`
- Modify: `skillfabric-ai/tests/unit/test_contract_extraction.py`

- [ ] **Step 1: Add failing policy and cache tests**

Require the stable name and complete extraction policy:

```python
assert CONTRACT_PROMPT_ID == "skill_contract"
assert "complete but nonredundant" in user
assert "consumes or transforms" in user
assert "Direct caller inputs are valid requirements" in user
assert "materially distinct externally usable" in user
```

Monkeypatch `CONTRACT_PROMPT_FINGERPRINT` between two runs and assert the extractor is called again.

- [ ] **Step 2: Verify failure**

```bash
/Users/chenjiang/Documents/Develop/Miniforge/envs/skillfabric_env/bin/python -m pytest tests/unit/test_contract_extraction.py -q
```

Expected: old minimal policy and cache key fail.

- [ ] **Step 3: Implement complete extraction semantics**

Use:

```python
CONTRACT_PROMPT_ID = "skill_contract"
CONTRACT_PROMPT_FINGERPRINT = prompt_fingerprint(
    CONTRACT_PROMPT_ID,
    _OUTPUT_SCHEMA,
    _CONTRACT_SEMANTICS,
    _DECISION_PROCESS,
    _EXAMPLES,
)
```

Define `requires` as all distinct source-supported external inputs, artifacts, resources, and
states consumed or transformed. Define `produces` as all materially distinct externally usable
outputs and states. Reject tools, credentials, internal state, inferred handoffs, fixed field
counts, and duplicate concepts. Keep exact line evidence and one authoritative JSON schema.

Replace the cache `prompt_id` value with both `prompt_name` and `prompt_fingerprint`.

- [ ] **Step 4: Verify and commit**

```bash
/Users/chenjiang/Documents/Develop/Miniforge/envs/skillfabric_env/bin/python -m pytest tests/unit/test_contract_extraction.py -q
git add skillfabric-ai/src/skillfabric/compiled_graph/contracts/prompts.py skillfabric-ai/src/skillfabric/compiled_graph/contracts/extraction.py skillfabric-ai/tests/unit/test_contract_extraction.py
git commit -m "feat: preserve complete skill contracts"
```

Expected: all contract tests pass with unchanged serialized contract keys.

### Task 3: Shared Ranking And Compact Profile Text

**Files:**
- Create: `skillfabric-ai/src/skillfabric/indexing/ranking.py`
- Modify: `skillfabric-ai/src/skillfabric/indexing/canonical.py`
- Modify: `skillfabric-ai/src/skillfabric/indexing/__init__.py`
- Modify: `skillfabric-ai/src/skillfabric/router/retrieval.py`
- Modify: `skillfabric-ai/tests/unit/test_router_bundle.py`
- Modify: `skillfabric-ai/tests/unit/test_bm25_index.py`

- [ ] **Step 1: Move tests to the shared API and add compact text coverage**

```python
from skillfabric.indexing.ranking import reciprocal_rank_fusion

text = compact_contract_text(skill, contract)
assert "Capability:" in text
assert "Requires:" in text
assert "Produces:" in text
assert "full source-only marker" not in text
```

- [ ] **Step 2: Verify failure**

```bash
/Users/chenjiang/Documents/Develop/Miniforge/envs/skillfabric_env/bin/python -m pytest tests/unit/test_router_bundle.py::test_reciprocal_rank_fusion_uses_channel_order_only tests/unit/test_bm25_index.py -q
```

Expected: shared ranking and compact text imports fail.

- [ ] **Step 3: Implement without changing router behavior**

Move the existing `RRF_K`, `FusedRank`, and `reciprocal_rank_fusion` definitions to
`indexing/ranking.py`. Import them in router retrieval. Refactor canonical text as:

```python
def compact_contract_text(skill: SkillNode, contract: SkillContract) -> str:
    return "\n".join(_contract_sections(skill, contract)).strip()


def contract_skill_text(skill, contract, *, source_char_limit: int = 2_000) -> str:
    sections = _contract_sections(skill, contract)
    source = _bounded_source(skill.raw_text, source_char_limit)
    if source:
        sections.append(f"Source:\n{source}")
    return "\n".join(sections).strip()
```

- [ ] **Step 4: Verify and commit**

```bash
/Users/chenjiang/Documents/Develop/Miniforge/envs/skillfabric_env/bin/python -m pytest tests/unit/test_router_bundle.py tests/unit/test_bm25_index.py -q
/Users/chenjiang/Documents/Develop/Miniforge/envs/skillfabric_env/bin/python -m ruff check src/skillfabric/indexing src/skillfabric/router/retrieval.py
git add skillfabric-ai/src/skillfabric/indexing skillfabric-ai/src/skillfabric/router/retrieval.py skillfabric-ai/tests/unit/test_router_bundle.py skillfabric-ai/tests/unit/test_bm25_index.py
git commit -m "refactor: share reciprocal rank fusion"
```

Expected: router ranking is behaviorally unchanged.

### Task 4: Hybrid Candidate Retrieval

**Files:**
- Modify: `skillfabric-ai/src/skillfabric/compiled_graph/semantic/models.py`
- Modify: `skillfabric-ai/src/skillfabric/compiled_graph/semantic/candidates.py`
- Modify: `skillfabric-ai/src/skillfabric/compiled_graph/builder.py`
- Modify: `skillfabric-ai/tests/unit/test_semantic_candidates.py`
- Modify: `skillfabric-ai/tests/unit/test_semantic_builder.py`

- [ ] **Step 1: Add failing lexical, fusion, and budget tests**

Build a temporary BM25 index where an exact format term retrieves a pair ranked outside the fake
dense budget. Assert:

```python
assert "lexical" in recovered_pair.channels
assert len({pair.key for pair in result.pairs}) == len(result.pairs)
assert explicit_reference_pair in result.pairs
assert first_run.pairs == second_run.pairs
```

Also assert no query skill retains more than `DEFAULT_CANDIDATE_TOP_K` fused matches except exact
references.

- [ ] **Step 2: Verify failure**

```bash
/Users/chenjiang/Documents/Develop/Miniforge/envs/skillfabric_env/bin/python -m pytest tests/unit/test_semantic_candidates.py tests/unit/test_semantic_builder.py -q
```

Expected: lexical channel and fused budget tests fail.

- [ ] **Step 3: Implement four-channel bounded retrieval**

Use one budget and four channel values:

```python
DEFAULT_CANDIDATE_TOP_K = 8
CandidateChannel = Literal["handoff", "explicit_reference", "similarity", "lexical"]
```

Query BM25 with `compact_contract_text`. Fuse per-query handoff, similarity, and lexical rank lists
with shared RRF, retain eight matched skills, preserve exact references, and globally deduplicate
unordered pairs. Do not persist RRF scores. Replace the two old top-k arguments with one
`candidate_top_k`, pass the BM25 path from the builder, and add lexical counts to existing metrics.

- [ ] **Step 4: Verify and commit**

```bash
/Users/chenjiang/Documents/Develop/Miniforge/envs/skillfabric_env/bin/python -m pytest tests/unit/test_semantic_candidates.py tests/unit/test_semantic_builder.py tests/unit/test_cli_build_wiki.py -q
git add skillfabric-ai/src/skillfabric/compiled_graph/semantic/models.py skillfabric-ai/src/skillfabric/compiled_graph/semantic/candidates.py skillfabric-ai/src/skillfabric/compiled_graph/builder.py skillfabric-ai/tests/unit/test_semantic_candidates.py skillfabric-ai/tests/unit/test_semantic_builder.py
git commit -m "feat: fuse semantic graph candidates"
```

Expected: candidate order is deterministic and graph schemas are unchanged apart from the useful
lexical channel value.

### Task 5: Runtime Skill Profiles And Relation Policy

**Files:**
- Modify: `skillfabric-ai/src/skillfabric/compiled_graph/semantic/prompts.py`
- Modify: `skillfabric-ai/src/skillfabric/compiled_graph/semantic/validation.py`
- Modify: `skillfabric-ai/tests/unit/test_semantic_validation.py`

- [ ] **Step 1: Add failing profile and cache tests**

Require name, description, capability, selection conditions, inputs, outputs, tools, exact evidence,
and adjacent evidence context. Assert unrelated source text is absent, `Prefer none` is absent, and:

```python
assert RELATION_PROMPT_ID == "semantic_relation_judge"
assert CYCLE_PROMPT_ID == "dependency_cycle_adjudicator"
assert "stable, reusable workflow progression" in rendered
```

Change `RELATION_PROMPT_FINGERPRINT` between two cached runs and assert the second run calls the
judge again.

- [ ] **Step 2: Verify failure**

```bash
/Users/chenjiang/Documents/Develop/Miniforge/envs/skillfabric_env/bin/python -m pytest tests/unit/test_semantic_validation.py -q
```

Expected: old full-source prompt and cache identity fail.

- [ ] **Step 3: Implement complete runtime profiles**

Create a private prompt payload containing registry metadata and the complete validated contract.
Collect contract and explicit-reference evidence, add immediate adjacent source lines, drop blanks,
deduplicate by line number, and sort. This payload is not persisted. Define positive and negative
tests for all relation types, keep hard dependency strict, broaden compose semantics to stable
workflow progression, and remove the global `none` bias.

Compute relation and cycle prompt fingerprints from trusted policy sections and schemas. Include
the relation fingerprint in pair cache keys while retaining exact source-line output validation.

- [ ] **Step 4: Verify and commit**

```bash
/Users/chenjiang/Documents/Develop/Miniforge/envs/skillfabric_env/bin/python -m pytest tests/unit/test_semantic_validation.py tests/unit/test_semantic_projection.py -q
git add skillfabric-ai/src/skillfabric/compiled_graph/semantic/prompts.py skillfabric-ai/src/skillfabric/compiled_graph/semantic/validation.py skillfabric-ai/tests/unit/test_semantic_validation.py
git commit -m "feat: judge relations from complete skill profiles"
```

Expected: profile, relation, evidence, cache, and cycle tests pass.

### Task 6: Exact Multi-pair Requests

**Files:**
- Modify: `skillfabric-ai/src/skillfabric/compiled_graph/semantic/prompts.py`
- Modify: `skillfabric-ai/src/skillfabric/compiled_graph/semantic/validation.py`
- Modify: `skillfabric-ai/tests/unit/semantic_fixtures.py`
- Modify: `skillfabric-ai/tests/unit/test_semantic_validation.py`
- Modify: `skillfabric-ai/tests/unit/test_semantic_builder.py`

- [ ] **Step 1: Add failing exact-response tests**

For two requested pairs, reject these payloads:

```python
{"decisions": [first_decision]}
{"decisions": [first_decision, first_decision]}
{"decisions": [first_decision, second_decision, extra_decision]}
```

Assert cached pairs are removed before request packing and every pending pair is submitted once.

- [ ] **Step 2: Verify failure**

```bash
/Users/chenjiang/Documents/Develop/Miniforge/envs/skillfabric_env/bin/python -m pytest tests/unit/test_semantic_validation.py tests/unit/test_semantic_builder.py -q
```

Expected: the single-pair protocol fails the new tests.

- [ ] **Step 3: Implement one request protocol**

Replace the judge signature with:

```python
def judge(
    self,
    pairs: tuple[CandidatePair, ...],
    skills: dict[str, SkillNode],
    contracts: dict[str, SkillContract],
) -> dict[str, Any]:
    """Return one raw decision for every requested pair."""
```

Use one internal `RELATION_PAIRS_PER_REQUEST` constant. Pack pairs deterministically around shared
endpoints. Require a wrapper with exactly `decisions`, validate every item with the existing
pair-level validator, and compare returned pair keys exactly with requested pair keys. Decompose
valid results into pair-level cache entries. Do not implement pairwise or full-source fallback.

- [ ] **Step 4: Verify and commit**

```bash
/Users/chenjiang/Documents/Develop/Miniforge/envs/skillfabric_env/bin/python -m pytest tests/unit/test_semantic_validation.py tests/unit/test_semantic_builder.py tests/unit/test_semantic_projection.py -q
git add skillfabric-ai/src/skillfabric/compiled_graph/semantic skillfabric-ai/tests/unit/semantic_fixtures.py skillfabric-ai/tests/unit/test_semantic_validation.py skillfabric-ai/tests/unit/test_semantic_builder.py
git commit -m "feat: validate relation requests exactly"
```

Expected: exact pair coverage and pair-granular cache reuse pass.

### Task 7: Stable Names For Remaining Prompts

**Files:**
- Modify: `skillfabric-ai/src/skillfabric/wiki/summarizer.py`
- Modify: `skillfabric-ai/src/skillfabric/wiki/explorer/prompting.py`
- Modify: `skillfabric-ai/src/skillfabric/orchestrator/package.py`
- Modify: `skillfabric-ai/tests/unit/test_wiki_summary_cache.py`
- Modify: `skillfabric-ai/tests/unit/test_wiki_explorer_prompting.py`
- Modify: `skillfabric-ai/tests/unit/test_orchestrator_package.py`
- Modify: `skillfabric-ai/tests/integration/test_real_claude_sdk_route.py`

- [ ] **Step 1: Add failing stable-name assertions**

```python
assert WIKI_SUMMARY_PROMPT_ID == "wiki_summary"
assert EXPLORER_PROMPT_ID == "query_wiki_explorer_semantic"
assert PLANNER_PROMPT_ID == "skillfabric_execution_planner"
```

- [ ] **Step 2: Rename constants and fingerprint summary cache policy**

Remove numeric suffixes without aliases. Add the trusted summary policy fingerprint to summary
cache keys so later policy edits invalidate cache without renaming the prompt.

- [ ] **Step 3: Verify and commit**

```bash
rg -n 'PROMPT_ID\s*=.*_v[0-9]' src tests
/Users/chenjiang/Documents/Develop/Miniforge/envs/skillfabric_env/bin/python -m pytest tests/unit/test_wiki_summary_cache.py tests/unit/test_wiki_explorer_prompting.py tests/unit/test_orchestrator_package.py tests/integration/test_real_claude_sdk_route.py -q
git add skillfabric-ai/src/skillfabric/wiki skillfabric-ai/src/skillfabric/orchestrator/package.py skillfabric-ai/tests
git commit -m "refactor: stabilize prompt identities"
```

Expected: search returns no prompt-name suffixes and tests pass.

### Task 8: Full Local Verification

**Files:** Review all production and test changes.

- [ ] **Step 1: Compile, test, and lint**

```bash
/Users/chenjiang/Documents/Develop/Miniforge/envs/skillfabric_env/bin/python -m compileall -q src tests
/Users/chenjiang/Documents/Develop/Miniforge/envs/skillfabric_env/bin/python -m pytest tests/unit -q
/Users/chenjiang/Documents/Develop/Miniforge/envs/skillfabric_env/bin/python -m ruff check src tests
git diff --check
```

Expected: every command exits zero.

- [ ] **Step 2: Audit discarded paths and worktree scope**

```bash
rg -n 'fallback|legacy|compat|full_source|pairwise_mode|batch_mode|PROMPT_ID\s*=.*_v[0-9]' src tests
git status --short
```

Expected: no experimental or compatibility path remains; status contains only intentional files
and the pre-existing usage-accounting edits.

### Task 9: Real Seeds Selection And Build

**Files:** Write only to a new timestamped experiment workspace under
`Skill_Fabric/experiments/AgentSkillOS`.

- [ ] **Step 1: Create a fixed source-reviewed positive and hard-negative pair set**

Use current candidates, historical edges, and benchmark co-occurrence only to discover pairs for
manual source review. Never treat co-occurrence as a graph label and never place benchmark skill
names in public code.

- [ ] **Step 2: Compare request sizes with the same model and labels**

Measure current full-source pairwise, Skill Profile pairwise, four-pair, and eight-pair requests.
Record exact-label accuracy, positive recall, accepted precision, calls, tokens, and cost. Set
`RELATION_PAIRS_PER_REQUEST` to the least expensive size that does not regress from the strongest
pairwise result, then remove temporary experiment code.

- [ ] **Step 3: Run a fresh Seeds build and release checks**

Use the same 53 skills, `gpt-5.4-mini`, BGE-M3, concurrency 32, retry controls, and wiki mode as the
baseline, with no copied graph caches. Require at least 90 percent candidate recall, at least 90
percent manually reviewed accepted-edge precision, no hard-negative edges, no route regression,
and no more than 8,923,070 total tokens.

- [ ] **Step 4: Freeze and commit the production implementation**

```bash
git add skillfabric-ai/src skillfabric-ai/tests
git commit -m "feat: finalize semantic graph quality pipeline"
git status --short
git log -1 --oneline
```

Expected: no environment files, caches, logs, experiment outputs, or unrelated files are staged.

### Task 10: Scale And Paper Readiness

- [ ] **Step 1: Build the 500-skill pool with frozen prompts and parameters**

Record candidate counts, edges by type, isolated nodes, usage by operation, wall time, and sampled
edge precision.

- [ ] **Step 2: Build the normalized top-level 1,000-skill pool identically**

Exclude nested `SKILL.md` files and retain all standard graph, route, status, and usage artifacts.

- [ ] **Step 3: Run SkillNet-Fabric AgentSkillOS and official BT evaluation**

Use official-equivalent harness behavior and the accepted judge configuration. Audit commit,
models, task set, skill pool, build status, routes, usage, and BT artifacts without modifying
official benchmark code.
