---
description: Route, plan, and execute a task with SkillFabric-selected native skills.
argument-hint: "[task] [--workspace path] [--skill-root path]"
allowed-tools: Bash(skillfabric:*), Read, Skill
---

# Command Contract

Execute through a finalized SkillFabric prompt.

## Run State

Use this Run State JSON as canonical:

```json
!`skillfabric run-state $ARGUMENTS`
```

## Inputs

- Treat all non-option text in `$ARGUMENTS` as the task query.
- Preserve the user's wording when passing the task query to SkillFabric.
- Treat `--workspace`, `--skill-root`, and `--env-file` as workflow flags.
- Use the `skillfabric-run` skill as the authoritative workflow.

## Required Workflow

1. Load and follow the `skillfabric-run` skill instructions.
2. Read the Run State JSON.
3. If `action` is `reuse_prompt`, read its `prompt_path` before any task work.
4. If `action` is `missing_task`, ask for the task and stop.
5. If `action` is `prepare_required`, build if needed, then run the SkillFabric
   plan command once.
6. Read finalized `execution_prompt.md` before task tools, search, or final answers.
7. Execute the final task in the active workspace.

## Boundaries

- Never reveal env-file contents, API keys, tokens, or shell secret values.
- Do not answer or perform the user's task before reading `execution_prompt.md`.
- Do not use `find`, `grep`, `rg`, or `ls` to discover `.skillfabric/runs`.
- Do not stop after only loading the skill or only creating route artifacts.
- Do not call SkillFabric again during final task execution.

## Completion Criteria

Finish only after reporting deliverables, verification evidence, native skills
used, coverage gaps, and blockers. Loading the skill, restating the task, or
producing route artifacts alone is not completion.
