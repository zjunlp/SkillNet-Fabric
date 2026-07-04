---
description: Check SkillFabric CLI, API configuration, and workspace readiness.
argument-hint: "[--env-file path] [--workspace path]"
---

# Command Contract

Run the SkillFabric readiness check for the current Claude Code workspace.

## Inputs

- Treat `$ARGUMENTS` as optional diagnostic flags.
- Use the `skillfabric-doctor` skill as the authoritative workflow.

## Required Workflow

1. Load and follow the `skillfabric-doctor` skill instructions.
2. Resolve the env file and workspace defaults from that skill.
3. Run the non-secret CLI checks required by the skill.
4. Treat CLI JSON as the source of truth for status.

## Boundaries

- Never reveal env-file contents, API keys, tokens, or shell secret values.
- Do not build, route, plan, or execute a task.

## Completion Criteria

Finish only after reporting CLI availability, API configuration status, workspace
readiness, and the next useful SkillFabric command.
