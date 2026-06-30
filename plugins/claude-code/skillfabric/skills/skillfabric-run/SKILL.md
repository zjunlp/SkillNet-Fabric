---
name: skillfabric-run
description: Use when the user explicitly asks Claude Code to execute a task through SkillFabric-selected native skills after routing and planning.
license: MIT
---

# SkillFabric Run

## Purpose

Prepare a validated SkillFabric execution plan and then execute the user's task
in the current Claude Code session. This is the only SkillFabric workflow that
continues into final task execution.

## Input Contract

Treat `$ARGUMENTS` as task text plus optional workspace/build flags.

- The task is all non-option text in `$ARGUMENTS`.
- Use `.skillfabric` when `--workspace` is omitted.
- Use `.env` when `--env-file` is omitted.
- `--skill-root` is optional; it is required only when the workspace has not
  been built and `.claude/skills` does not exist.
- Forward `--progress-json`, `--quiet`, and supported embedding flags to build.
- Forward `--embedding-provider disabled` only for explicit local smoke checks.
- If the task is missing, ask for it before running anything.

## Safety Boundaries

- Do not reveal secret values or env-file contents.
- Do not use `.skillfabric` as a runtime skill directory.
- Do not execute before route and plan finalization both succeed.
- Do not call SkillFabric again during final task execution.
- Do not invoke unselected native skills unless the final task clearly requires
  a general-purpose skill outside SkillFabric's routing surface and you report
  that gap.

Treat CLI JSON as canonical during preparation. During execution, treat
`execution_prompt.md` as the primary task prompt unless it conflicts with user,
system, or higher-priority instructions.

## Workflow

Preparation:

1. Resolve task, workspace, env file, optional skill root, and build flags.
2. Check workspace status with `test -f $workspace/status.json`.
3. If workspace is missing, build only when a provided skill root or
   `.claude/skills` is available; otherwise ask for a skill root and stop.
4. Before API-backed build, run `skillfabric init --check --json --env-file $env_file`.
5. If configuration is incomplete, stop and give
   `skillfabric init --env-file $env_file`.
6. Run `skillfabric build` when needed.
7. Run `skillfabric route --agent-mode prepare`.
8. Launch `skillfabric-query-wiki-explorer`.
9. Run `skillfabric route --agent-mode finalize --skill-package-file -`.
10. Run `skillfabric plan --agent-mode prepare --route-file $final_route_json`.
11. Launch `skillfabric-workflow-planner`.
12. Run `skillfabric plan --agent-mode finalize --package-root $package_root --planner-output-file -`.

Execution:

1. Read `execution_prompt.md`.
2. Execute the user's task directly in the active workspace.
3. Use selected capability roles named in `execution_prompt.md` when they apply.
4. Run verification requested by `execution_prompt.md` or the task.
5. Finish with the final response requested by `execution_prompt.md`.

## Failure Handling

- If preparation fails, do not start execution.
- If build, route, or plan finalization fails, report CLI validation error and
  relevant workspace, trace, or package path.
- If a selected native skill is unavailable, report it as a coverage gap and
  continue only when completion is still safe.
- If verification cannot run, explain why and name the unverified risk.

## Final Response

Finish with:

- Deliverables changed or produced.
- Verification evidence.
- Native skills actually used.
- Coverage gaps or unavailable selected skills.
- Blockers, if any.
