---
description: Check SkillFabric CLI, API configuration, and workspace readiness.
argument-hint: "[--env-file path] [--workspace path]"
allowed-tools: Bash(skillfabric:*), Skill
---

# Command Contract

Report SkillFabric readiness from CLI state.

## Doctor State

Use this Doctor State JSON as canonical:

```json
!`skillfabric doctor-state $ARGUMENTS`
```

## Inputs

- Treat `$ARGUMENTS` as optional diagnostic flags.
- Use the `skillfabric-doctor` skill as the authoritative workflow.

## Required Workflow

1. Load and follow the `skillfabric-doctor` skill instructions.
2. Read the Doctor State JSON.
3. Report readiness from that JSON only.
4. If JSON is missing or invalid, run `skillfabric doctor-state $ARGUMENTS`.

## Boundaries

- Never reveal env-file contents, API keys, tokens, or shell secret values.
- Do not use `find`, `grep`, `rg`, `sed`, `cat`, or directory scans.
- Do not build, route, plan, or execute a task.

## Completion Criteria

Finish only after reporting CLI availability, API configuration status, workspace
readiness, and the next useful SkillFabric command.
