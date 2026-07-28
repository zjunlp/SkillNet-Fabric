# Planner Prompt Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the shared SkillFabric Planner produce shorter, source-grounded execution prompts that improve downstream task completion without changing its output schema.

**Architecture:** Keep the existing `SkillFabric.plan()` data flow, untrusted-input boundary, and single-field JSON response. Revise only the system prompt contract and its prompt ID, then lock the intended behavior with unit tests that inspect the rendered Planner messages.

**Tech Stack:** Python 3.11+, pytest, LiteLLM-compatible message construction, JSON Schema

---

### Task 1: Lock the New Planner Contract with Failing Tests

**Files:**
- Modify: `skillfabric-ai/tests/unit/test_orchestrator_package.py:91-151`
- Test: `skillfabric-ai/tests/unit/test_orchestrator_package.py`

- [x] **Step 1: Update the length and workflow assertions**

Replace the existing length assertions in
`test_planner_builds_a_short_end_to_end_handoff_with_targeted_checks` with:

```python
assert "normally 150-350 words" in prompt
assert "up to 500 words" in prompt
assert "three to six ordered steps" in prompt
assert "normally 300-700 words" not in prompt
```

- [x] **Step 2: Add source-grounding and simplicity tests**

Add:

```python
def test_planner_uses_one_source_grounded_primary_path() -> None:
    prompt = _planner_contract_prompt()

    assert "one primary execution path" in prompt
    assert "traceable to the original task or selected Skill context" in prompt
    assert "Do not invent thresholds, algorithms, libraries, commands, parameters" in prompt
    assert "Prefer the simplest method that fully satisfies the task" in prompt
    assert "Do not present alternatives" in prompt


def test_planner_uses_skills_concisely_without_repeating_sources() -> None:
    prompt = _planner_contract_prompt()

    assert "one clear role" in prompt
    assert "mention its exact `skill_id` once" in prompt
    assert "Do not enumerate selected Skills" in prompt
    assert "Do not restate Skill instructions" in prompt
```

- [x] **Step 3: Add positive-example tests**

Add:

```python
def test_planner_contract_contains_a_short_positive_example() -> None:
    prompt = _planner_contract_prompt()

    assert "<example>" in prompt
    assert "<example_execution_prompt>" in prompt
    assert "Produce `output.json` from `input.dat`" in prompt
    assert "This example demonstrates shape, not a mandatory template" in prompt
```

- [x] **Step 4: Update the prompt identity assertion**

Change the expected value in `test_plan_calls_llm_once_with_complete_selected_context` to:

```python
assert PLANNER_PROMPT_ID == "skillfabric_execution_planner_task_grounded_handoff_v2"
```

- [x] **Step 5: Run the focused tests and confirm failure**

Run:

```bash
cd skillfabric-ai
pytest tests/unit/test_orchestrator_package.py -q
```

Expected: failures for the new prompt ID and missing contract phrases; existing functional tests remain passing.

### Task 2: Rewrite the Shared Planner Prompt Contract

**Files:**
- Modify: `skillfabric-ai/src/skillfabric/orchestrator/package.py:30`
- Modify: `skillfabric-ai/src/skillfabric/orchestrator/package.py:243-334`
- Test: `skillfabric-ai/tests/unit/test_orchestrator_package.py`

- [x] **Step 1: Version the prompt identity**

Set:

```python
PLANNER_PROMPT_ID = "skillfabric_execution_planner_task_grounded_handoff_v2"
```

- [x] **Step 2: Replace the Planner reasoning and quality rules**

Keep the existing `<role>`, `<trust_boundary>`, `<objective>`, and `<output_contract>` boundaries,
but make the role explicitly outcome-first and replace the current planning rules with:

```text
<planning_process>
Before writing, internally extract the hard output contract, the few constraints most likely to
cause failure, and the selected Skill guidance that directly addresses them. Assign each useful
Skill one clear role, identify real producer-consumer handoffs, and choose one primary execution
path. Do not expose this analysis.
</planning_process>

<quality_rules>
1. Preserve every literal deliverable, path, filename, format, quantity, ordering rule, and
   acceptance constraint from the original task.
2. Prefer the simplest method that fully satisfies the task. Do not present alternatives,
   optional enhancements, or extra deliverables unless the task explicitly requests them.
3. Include an implementation detail only when it is traceable to the original task or selected
   Skill context. Do not invent thresholds, algorithms, libraries, commands, parameters,
   dependencies, or environmental assumptions.
4. Use a selected Skill only when it materially improves the workflow. Give it one clear role,
   mention its exact `skill_id` once, and integrate its decisive guidance at the relevant step.
   Do not enumerate selected Skills or restate Skill instructions.
5. Treat graph relations as evidence. Use a directed relation only for a concrete source-before-
   target handoff; do not turn `compose_with` adjacency or a coverage gap into required work.
6. Leave reversible choices to the executor. Resolve genuine underspecification with conservative,
   internally consistent assumptions only when needed to complete the requested artifact.
7. End with only the task-specific checks that can catch likely defects in the actual deliverables.
   Generic runtime, security, fallback, and file-existence guidance is already supplied elsewhere.
</quality_rules>
```

- [x] **Step 3: Tighten the execution-prompt contract and add one example**

Use:

```text
<execution_prompt_contract>
Write a concise, directly executable handoff for one capable executor.

- Lead with the target artifact or outcome and its exact output contract.
- Give one primary execution path, normally as three to six ordered steps.
- Preserve decisive task constraints and Skill-informed methods without retelling the task.
- Close with a short, task-specific definition of done or final check.

Use the structure that best fits the task; headings and numbered steps are optional. Write normally
150-350 words, shorter for simple tasks and up to 500 words only for genuinely complex,
multi-artifact work.
</execution_prompt_contract>

<example>
<example_execution_prompt>
Produce `output.json` from `input.dat` with the exact fields and ordering required by the task.

1. Use `skill:domain-parser-example` once to parse the source records while preserving source IDs.
2. Apply the task's stated normalization rules and write only the requested records.
3. Reload `output.json` and check its schema, record coverage, ordering, and source-ID preservation.
</example_execution_prompt>
This example demonstrates shape, not a mandatory template. Adapt detail and structure to the task.
</example>
```

- [x] **Step 4: Run the focused tests**

Run:

```bash
cd skillfabric-ai
pytest tests/unit/test_orchestrator_package.py -q
```

Expected: all tests pass.

### Task 3: Verify the Public Planner Surface

**Files:**
- Verify: `skillfabric-ai/src/skillfabric/orchestrator/package.py`
- Verify: `skillfabric-ai/tests/unit/test_orchestrator_package.py`
- Verify: `docs/superpowers/specs/2026-07-29-planner-prompt-quality-design.md`
- Verify: `docs/superpowers/plans/2026-07-29-planner-prompt-quality.md`

- [x] **Step 1: Run prompt-adjacent unit tests**

Run:

```bash
cd skillfabric-ai
pytest tests/unit/test_orchestrator_package.py tests/unit/test_public_package.py tests/unit/test_cli_route_plan.py -q
```

Expected: all tests pass without network access.

- [x] **Step 2: Run the complete unit suite**

Run:

```bash
cd skillfabric-ai
pytest tests/unit -q
```

Expected: all tests pass. If an unrelated pre-existing failure occurs, record the exact failing test
and keep the focused Planner tests as the acceptance gate.

- [x] **Step 3: Check the diff**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only the Planner source, its unit tests, and this plan are changed
after the already committed design spec.

- [x] **Step 4: Commit the implementation**

Run:

```bash
git add skillfabric-ai/src/skillfabric/orchestrator/package.py \
  skillfabric-ai/tests/unit/test_orchestrator_package.py \
  docs/superpowers/plans/2026-07-29-planner-prompt-quality.md
git commit -m "improve planner execution prompt quality"
```

Expected: one implementation commit containing the public Planner contract, tests, and execution
plan.
