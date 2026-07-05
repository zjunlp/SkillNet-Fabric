---
name: skillfabric-run
description: Use when the user explicitly asks Claude Code to execute a task through SkillFabric-selected native skills after routing and planning.
license: MIT
disable-model-invocation: true
---

# SkillFabric Run

## Purpose

Execute a task from a finalized SkillFabric execution prompt. Reuse the most
recent prepared prompt when available; otherwise prepare one first from the
provided task. This is the only SkillFabric workflow that continues into final
task execution.

## Input Contract

Treat `$ARGUMENTS` as task text plus optional workspace/build flags.

- The task is the user's natural-language request after removing explicit
  SkillFabric flags. Preserve the user's wording and do not rewrite it into a
  narrower task.
- Treat only recognized flags as workflow configuration. Do not infer API keys,
  models, or workspaces from free-form prose.
- Use `.skillfabric` when `--workspace` is omitted.
- Use `.env` when `--env-file` is omitted.
- `--skill-root` is optional; it is required only when the workspace has not
  been built and `.claude/skills` does not exist.
- Forward `--progress-json`, `--quiet`, and supported embedding flags to build.
- Forward `--embedding-provider disabled` only for explicit local smoke checks.
- If the task is missing and no finalized execution prompt exists, ask for the
  task before running anything.

## Safety Boundaries

- Do not reveal secret values or env-file contents.
- Do not use `.skillfabric` as a runtime skill directory.
- Do not execute before reading a finalized `execution_prompt.md`.
- Do not call SkillFabric again during final task execution.
- Do not invoke unselected native skills unless the final task clearly requires
  a general-purpose skill outside SkillFabric's routing surface and you report
  that gap.
- Before reading `execution_prompt.md`, do not use Web Search, Fetch, native
  task skills, project-editing tools, or direct answers to work on the user's
  task. Only inspect SkillFabric status/artifacts and run the required
  SkillFabric CLI steps.
- Before reading `execution_prompt.md`, do not search `.skillfabric` with
  `find`, `grep`, `rg`, or directory listings to discover package paths. The
  only valid source for run state is `skillfabric run-state`.
- Never inspect env files with commands that can print secret values. To check
  whether a task-specific variable exists, use a presence-only check that prints
  `SET` or `MISSING`.

Treat CLI JSON as canonical during preparation. During execution, treat
`execution_prompt.md` as the primary task prompt unless it conflicts with user,
system, or higher-priority instructions.

## Workflow

Run-state gate:

1. Resolve task, workspace, env file, optional skill root, and build flags.
2. Prefer the Run State JSON injected by the slash command. If it is absent,
   run `skillfabric run-state "$task" --workspace "$workspace" --env-file "$env_file"`.
3. If `action` is `reuse_prompt`, set `$execution_prompt` to `prompt_path`.
   Read that exact file before loading native skills or running task commands,
   then continue to Execution.
4. If `action` is `missing_task`, ask the user for a task and stop.
5. If `action` is `prepare_required`, use its `task`, `workspace`, and
   `env_file`, then continue to Preparation path.
6. Do not substitute shell path discovery for `run-state`.

Preparation path:

1. Check workspace status with `test -f $workspace/status.json`.
2. If workspace is missing, build only when a provided skill root or
   `.claude/skills` is available; otherwise ask for a skill root and stop.
3. Before API-backed build, run `skillfabric init --check --json --env-file $env_file`.
4. If configuration is incomplete, stop and give
   `skillfabric init --env-file $env_file`.
5. Run `skillfabric build` when needed.
6. Run
   `skillfabric route "$task" --workspace "$workspace" --env-file "$env_file" --agent-mode prepare`
   with forwarded route flags.
7. Read the returned JSON fields: `trace_id`, `query_wiki_root`,
   `agent_route_request`, and `skill_package_file`.
8. In the main Claude Code session, read `agent_route_request.json`,
   `query_wiki/EXPLORER.md`, and `query_wiki/index.md`; then read only the
   query-wiki skill, community, workflow, or edge pages needed to select an
   evidence-backed SkillPackage. Do not inspect the active project workspace
   during this route-selection step. Route selection is about choosing skills
   from the query wiki, not solving the user's task.
9. Write a single raw SkillPackage JSON object matching
   `agent_route_request.json.expected_schema` to the returned
   `skill_package_file`. Do not wrap it in Markdown fences, comments, or
   explanatory text. Then run
   `skillfabric route "$task" --workspace "$workspace" --env-file "$env_file" --trace-id "$trace_id" --agent-mode finalize --skill-package-file "$skill_package_file"`.
10. Read the finalized route JSON and use its `trace_dir/route.json` as
    `$route_json`.
11. Run
    `skillfabric plan --workspace "$workspace" --agent-mode prepare --route-file "$route_json"`.
12. Read the returned JSON fields: `root`, `planner_request_path`,
    `planner_prompt_path`, and `planner_output_path`.
13. In the main Claude Code session, read `planner_request.json`, `PLANNER.md`,
    `route.json`, `evidence/required_edges.json`,
    `evidence/selected_skill_evidence.json`, `evidence/route_summary.json`,
    and only the selected skill pages needed to understand capability boundaries.
14. Optionally perform bounded active-workspace inspection when it improves task
    execution: read non-secret README/project metadata, file maps, git status,
    relevant source or tests, and obvious verification commands. Do not read
    `.env`, tokens, credentials, large caches, historical run directories, or
    unrelated generated artifacts. Use these observations to ground the
    execution plan before finalization.
15. Write a single raw planner JSON object matching
    `planner_request.json.expected_schema` to the returned
    `planner_output_path`. Do not wrap it in Markdown fences, comments, or
    explanatory text. Use real newline characters inside `execution_prompt`,
    not literal `\n` text. Then run
    `skillfabric plan --workspace "$workspace" --agent-mode finalize --package-root "$package_root" --planner-output-file "$planner_output_path"`.
16. Confirm that the finalized payload points to an existing
    `execution_prompt.md`; set `$execution_prompt` to that path.

Execution:

1. Read `$execution_prompt` exactly once from the path returned by the gate or
   finalization payload.
2. Execute the user's task directly in the active workspace.
3. Use selected capability roles named in `execution_prompt.md` when they apply.
4. Run verification requested by `execution_prompt.md` or the task.
5. Finish with the final response requested by `execution_prompt.md`.

## Failure Handling

- If preparation fails, do not start execution.
- If build, route, or plan finalization fails, report CLI validation error and
  relevant workspace, trace, or package path.
- If no reusable prompt exists and no task was provided, ask for a task and do
  not start execution.
- If `execution_prompt.md` is missing or unreadable, report preparation as
  failed and do not start task execution.
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
