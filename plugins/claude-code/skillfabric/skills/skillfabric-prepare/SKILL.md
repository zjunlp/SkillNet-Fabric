---
name: skillfabric-prepare
description: Use when the user asks Claude Code to route and plan a task with SkillFabric without executing the task.
license: MIT
disable-model-invocation: true
---

# SkillFabric Prepare

## Purpose

Create a CLI-validated route and execution package, then stop before task
execution.

## Input Contract

Treat `$ARGUMENTS` as task text plus the small set of recognized SkillFabric
workflow flags.

- The task is the user's natural-language request after removing explicit
  workflow flags. Preserve the user's wording.
- Treat only `--workspace`, `--env-file`, and `--skill-root` as workflow
  configuration. Keep route and planner implementation limits at their stable
  runtime defaults.
- Use `.skillfabric` for the workspace and `.env` for the env file when omitted.
- Use `--skill-root` only when a build is required.

## Safety Boundaries

- Do not reveal secret values or env-file contents.
- Do not solve the user's task or inspect the active project during route
  selection.
- Do not execute `execution_prompt.md`.
- Do not invent route or planner output after CLI validation fails.

Treat CLI JSON as canonical for paths, selected skills, validation, and final
artifacts.

## Workflow

1. Resolve task, workspace, env file, optional skill root, and supported flags.
2. If the workspace is not ready, require a valid skill root, run
   `skillfabric init --check --json --env-file $env_file`, then run the supported
   `skillfabric build` command. Stop on configuration or build failure.
3. Read and follow the bundled `references/route-plan.md` exactly once.
4. Confirm the returned package contains `route.json`, `execution_prompt.md`,
   and a successful planner validation artifact.

## Failure Handling

- Stop on any build, route, schema, or plan validation error.
- Report only non-secret errors and the relevant workspace, trace, or package
  path.
- Missing final artifacts mean preparation is incomplete.

## Final Response

Report the package root, selected skills, coverage gaps, execution prompt,
validation path, and blockers. State that the final task was not
executed.
