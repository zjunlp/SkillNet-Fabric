# Selection And Prompt Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Explorer selection-only and replace the intermediate workflow DAG with one context-bounded LLM call that writes `execution_prompt.md`.

**Architecture:** Preserve query-wiki SDK exploration, but treat graph relations as planner evidence rather than route authority. Simplify the public route and planner schemas so every retained field affects selection or final prompt generation.

**Tech Stack:** Python 3.11+, dataclasses, Claude Agent SDK explorer, LiteLLM planner, pytest, Ruff.

---

### Task 1: Selection-Only SkillPackage

**Files:**
- Modify: `skillfabric-ai/src/skillfabric/wiki/explorer/skill_package.py`
- Modify: `skillfabric-ai/src/skillfabric/wiki/explorer/validation.py`
- Modify: `skillfabric-ai/src/skillfabric/wiki/explorer/prompting.py`
- Test: `skillfabric-ai/tests/unit/test_skill_package.py`
- Test: `skillfabric-ai/tests/unit/test_wiki_explorer_prompting.py`

- [ ] Write tests whose schema accepts only `selected_skills`, `near_misses`, `coverage_gaps`, `wiki_pages_read`, and `rationale`.
- [ ] Run the focused tests and confirm they fail on dependency/order fields and prerequisite closure.
- [ ] Remove relation declarations and ordering from `SkillPackage` and its prompt.
- [ ] Retain manifest, evidence, path, duplicate, limit, and empty-selection validation.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Route Relation Evidence

**Files:**
- Modify: `skillfabric-ai/src/skillfabric/router/models.py`
- Modify: `skillfabric-ai/src/skillfabric/wiki/explorer/validation.py`
- Test: `skillfabric-ai/tests/unit/test_router_route.py`
- Test: `skillfabric-ai/tests/unit/test_skill_package.py`

- [ ] Write tests proving one selected dependent does not force its graph prerequisite.
- [ ] Write tests proving relations among selected skills are projected as evidence only.
- [ ] Run the tests and confirm the current hard-closure behavior fails.
- [ ] Replace dependency/composition/order fields with a single typed relation-evidence collection.
- [ ] Run the focused tests and confirm they pass.

### Task 3: Prompt-Only Planner

**Files:**
- Modify: `skillfabric-ai/src/skillfabric/orchestrator/package.py`
- Modify: `skillfabric-ai/src/skillfabric/orchestrator/__init__.py`
- Modify: `skillfabric-ai/src/skillfabric/api.py`
- Modify: `skillfabric-ai/src/skillfabric/cli.py`
- Test: `skillfabric-ai/tests/unit/test_orchestrator_package.py`
- Test: `skillfabric-ai/tests/unit/test_cli_route_plan.py`

- [ ] Write tests for the exact `{"execution_prompt": string}` output schema and absence of `workflow_plan.json`.
- [ ] Write tests for one planner call and explicit context-budget failure.
- [ ] Run the tests and confirm they fail against the current DAG planner.
- [ ] Delete workflow-step validation, dependency preservation, topological helpers, and workflow artifacts.
- [ ] Add one LiteLLM planner call with complete selected context and pre-call token estimation.
- [ ] Wire `SkillFabric.plan()` and CLI planning to the single-call path without fallback.
- [ ] Run the focused tests and confirm they pass.

### Task 4: Prompt And Documentation Alignment

**Files:**
- Modify: `skillfabric-ai/src/skillfabric/compiled_graph/contracts/prompts.py`
- Modify: `skillfabric-ai/src/skillfabric/compiled_graph/semantic/prompts.py`
- Modify: `skillfabric-ai/src/skillfabric/wiki/summarizer.py`
- Modify: `skillfabric-ai/README.md`
- Modify: `README.md`
- Modify: `plugins/claude-code/skillfabric/**`
- Test: prompt-focused unit tests and public package tests

- [ ] Audit every LLM prompt for role, untrusted-data isolation, semantics, decision process, exact output contract, and concise wording.
- [ ] Remove stale workflow-DAG and prepare/finalize compatibility documentation.
- [ ] Run prompt and package tests.

### Task 5: Verification And Commit

**Files:**
- Verify all changed source, tests, docs, and plugin files.

- [ ] Run `python -m compileall -q skillfabric-ai/src skillfabric-ai/tests`.
- [ ] Run the complete pytest suite.
- [ ] Run `python -m ruff check skillfabric-ai/src skillfabric-ai/tests`.
- [ ] Run `python -m ruff format --check skillfabric-ai/src skillfabric-ai/tests`.
- [ ] Run `git diff --check`.
- [ ] Run real Route and Planner queries against the completed 50-skill workspace and inspect selected skills, relation evidence, prompt usefulness, usage, and artifacts.
- [ ] Confirm no secret, `.env`, cache, temporary artifact, or unrelated file is staged.
- [ ] Commit only the reviewed `skillfabric-public` changes and verify `git status --short` plus `git log -1 --oneline`.
