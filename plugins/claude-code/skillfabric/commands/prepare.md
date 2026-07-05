---
description: Route a task and prepare a validated execution package without running it.
argument-hint: "[task] [--workspace path] [--skill-root path]"
---

# Command Contract

Prepare a validated SkillFabric execution package for one user task, without
executing that task.

## Inputs

- Treat all non-option text in `$ARGUMENTS` as the task query.
- Preserve the user's wording when passing the task query to SkillFabric.
- Treat options such as `--workspace`, `--skill-root`, and `--env-file` as
  workflow flags, not as task text.
- Use the `skillfabric-prepare` skill as the authoritative workflow.

## Required Workflow

1. Load and follow the `skillfabric-prepare` skill instructions.
2. Resolve workspace, env file, optional skill root, and forwarded flags.
3. Build the workspace only when the skill instructions say it is needed.
4. Run SkillFabric route prepare and route finalize.
5. Run SkillFabric plan prepare and plan finalize.
6. Confirm that the finalized package contains `execution_prompt.md`.

## Boundaries

- Never reveal env-file contents, API keys, tokens, or shell secret values.
- Do not answer or perform the user's task directly.
- Do not stop after only loading the skill or only creating route artifacts.
- Do not execute `execution_prompt.md`.

## Completion Criteria

Finish only after reporting the execution package root, `execution_prompt.md`,
planner validation path, selected skills, warnings, coverage gaps, and blockers.
Loading the skill, restating the task, or producing route artifacts alone is not
completion; route finalization and plan finalization must both have run.
