---
description: Route, plan, and execute a task with SkillFabric-selected native skills.
argument-hint: "[task] [--workspace path] [--skill-root path]"
---

# Command Contract

Route, plan, and execute one user task through SkillFabric-selected native
skills.

## Inputs

- Treat all non-option text in `$ARGUMENTS` as the task query.
- Preserve the user's wording when passing the task query to SkillFabric.
- Treat options such as `--workspace`, `--skill-root`, and `--env-file` as
  workflow flags, not as task text.
- Use the `skillfabric-run` skill as the authoritative workflow.

## Required Workflow

1. Load and follow the `skillfabric-run` skill instructions.
2. Resolve workspace, env file, optional skill root, and forwarded flags.
3. Build the workspace only when the skill instructions say it is needed.
4. Run SkillFabric route prepare and route finalize.
5. Run SkillFabric plan prepare and plan finalize.
6. Confirm that the finalized package contains `execution_prompt.md`.
7. Read `execution_prompt.md` and execute the final task in the active workspace.

## Boundaries

- Never reveal env-file contents, API keys, tokens, or shell secret values.
- Do not answer or perform the user's task before route and plan finalization.
- Do not stop after only loading the skill or only creating route artifacts.
- Do not call SkillFabric again during final task execution.

## Completion Criteria

Finish only after reporting deliverables changed or produced, verification
evidence, native skills actually used, coverage gaps, and blockers.
