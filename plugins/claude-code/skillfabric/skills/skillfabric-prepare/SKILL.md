---
name: skillfabric-prepare
description: Use when the user wants Claude Code to select relevant skills and create a SkillFabric execution package, handoff prompt, or implementation plan without executing the final task.
license: MIT
---

# SkillFabric Prepare

## Purpose

Prepare a complete SkillFabric handoff for one task. Build the workspace when
needed, route the task through query-wiki exploration, create a validated
execution package, and stop before final task execution.

## Input Contract

Treat `$ARGUMENTS` as task text plus optional workspace/build flags.

- The task is all non-option text in `$ARGUMENTS`.
- Use `.skillfabric` when `--workspace` is omitted.
- Use `.env` when `--env-file` is omitted.
- `--skill-root` is optional when the workspace already contains `status.json`.
- `--skill-root` is required only when the workspace has not been built and
  `.claude/skills` does not exist.
- If the workspace is missing and `.claude/skills` exists, use `.claude/skills`
  as the default skill root for the build step.
- Forward `--progress-json`, `--quiet`, and supported embedding flags to build.
- Forward `--embedding-provider disabled` only for explicit local smoke checks.
- If the task is missing, ask for it before running anything.

## Safety Boundaries

- Do not reveal secret values or env-file contents.
- Do not run the final task automatically.
- Do not use `.skillfabric` as a runtime skill directory.
- Do not skip route or plan finalization through the CLI.
- Do not introduce skills outside the finalized route.
- Do not continue after missing API configuration unless this is an explicit
  disabled-embedding local smoke check.

Treat CLI JSON as canonical for status, trace ids, package roots, prompt paths,
and warnings.

## Workflow

1. Resolve task, workspace, env file, optional skill root, and build flags.
2. Check workspace status with `test -f $workspace/status.json`.
3. If workspace is missing, resolve a skill root: provided `--skill-root`,
   `.claude/skills`, or ask the user and stop.
4. Before API-backed build, run `skillfabric init --check --json --env-file $env_file`.
5. If configuration is incomplete, stop and give
   `skillfabric init --env-file $env_file`.
6. Run `skillfabric build` when the workspace needs building.
7. Generate `task_atoms.json` in a temporary file under the workspace or trace
   parent, using the TaskAtoms schema below. Do this yourself as the current
   Claude agent; do not ask SkillFabric to infer atoms unless the CLI is used
   outside the plugin.
8. Run `skillfabric route --agent-mode prepare --task-atoms-file $task_atoms_json`.
9. Launch `skillfabric-query-wiki-explorer`.
10. Run `skillfabric route --agent-mode finalize --skill-package-file -`.
11. Run `skillfabric plan --agent-mode prepare --route-file $final_route_json`.
12. Launch `skillfabric-workflow-planner`.
13. Run `skillfabric plan --agent-mode finalize --package-root $package_root --planner-output-file -`.
14. Stop after finalization. Do not read and execute `execution_prompt.md`.

## TaskAtoms Schema

Write strict JSON only:

```json
{
  "schema_version": "1.0",
  "atoms": [
    {
      "id": "a1",
      "kind": "action",
      "text": "short atomic requirement",
      "evidence": "exact short substring from the user task",
      "required": true,
      "depends_on": []
    }
  ]
}
```

Rules:

- `kind` must be exactly `action`, `artifact`, or `constraint`.
- Create at most 12 atoms and merge duplicates.
- Every `evidence` value must be copied from the user's task text.
- Do not output `skill_id`, `intent`, `domain_hints`, `deliverable`, or graph vocabulary.
- Do not recommend, rank, or name skills.
- Do not map vague words to fixed formats. For example, do not turn
  `slides` into `.pptx` unless the task explicitly says `.pptx` or PowerPoint.
- Use `depends_on` only when the user states ordering or a clear prerequisite.

## Failure Handling

- If build fails, report CLI error and status artifact path when available.
- If route finalization rejects the SkillPackage, report validation errors and
  trace directory.
- If plan finalization rejects planner JSON, report validation errors and
  package root.
- If selected skills are insufficient, report coverage notes and stop at
  handoff artifacts.

## Final Response

Return:

- Handoff prompt path.
- Execution package root.
- `workflow_plan.json`.
- `execution_prompt.md`.
- Renderer and entry prompt label.
- Selected skills, warnings, coverage gaps, and blockers.

State that the final task has not been executed.
