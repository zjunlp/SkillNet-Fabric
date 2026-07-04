---
name: skillfabric-prepare
description: Use when the user wants Claude Code to select relevant skills and create a SkillFabric execution package and prompt-only handoff without executing the final task.
license: MIT
---

# SkillFabric Prepare

## Purpose

Prepare a complete SkillFabric handoff for one task. Build the workspace when
needed, route the task through query-wiki exploration, create a validated
prompt-only execution package, and stop before final task execution.

## Input Contract

Treat `$ARGUMENTS` as task text plus optional workspace/build flags.

- The task is all non-option text in `$ARGUMENTS`; preserve the user's wording.
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
7. Run
   `skillfabric route "$task" --workspace "$workspace" --env-file "$env_file" --agent-mode prepare`
   with forwarded route flags.
8. Read the returned JSON fields: `trace_id`, `query_wiki_root`,
   `agent_route_request`, and `skill_package_file`.
9. In the main Claude Code session, read `agent_route_request.json`,
   `query_wiki/EXPLORER.md`, and `query_wiki/index.md`; then read only the
   query-wiki skill, community, workflow, or edge pages needed to select an
   evidence-backed SkillPackage. Do not inspect the active project workspace
   during this route-selection step.
10. Produce a single raw SkillPackage JSON object matching
   `agent_route_request.json.expected_schema`; do not wrap it in Markdown fences,
   comments, or explanatory text. Pass that JSON to
   `skillfabric route "$task" --workspace "$workspace" --env-file "$env_file" --trace-id "$trace_id" --agent-mode finalize --skill-package-file -`.
11. Read the finalized route JSON and use its `trace_dir/route.json` as
    `$route_json`.
12. Run
    `skillfabric plan --workspace "$workspace" --agent-mode prepare --route-file "$route_json"`.
13. Read the returned JSON fields: `root`, `planner_request_path`,
    `planner_prompt_path`, and `planner_output_path`.
14. In the main Claude Code session, read `planner_request.json`, `PLANNER.md`,
    `route.json`, `evidence/required_edges.json`,
    `evidence/selected_skill_evidence.json`, `evidence/route_summary.json`,
    and only the selected skill pages needed to understand capability boundaries.
15. Optionally perform bounded active-workspace inspection when it improves the
    final handoff: read non-secret README/project metadata, file maps, git status,
    relevant source or tests, and obvious verification commands. Do not read
    `.env`, tokens, credentials, large caches, historical run directories, or
    unrelated generated artifacts.
16. Produce a single raw planner JSON object matching
    `planner_request.json.expected_schema`; do not wrap it in Markdown fences,
    comments, or explanatory text. Pass that JSON to
    `skillfabric plan --workspace "$workspace" --agent-mode finalize --package-root "$package_root" --planner-output-file -`.
17. Confirm that the finalized payload points to an existing `execution_prompt.md`.
18. Stop after finalization. Do not execute `execution_prompt.md`.

## Failure Handling

- If build fails, report CLI error and status artifact path when available.
- If route finalization rejects the SkillPackage, report validation errors and
  trace directory.
- If plan finalization rejects planner JSON, report validation errors and
  package root.
- If `execution_prompt.md` is not created, report prepare as failed or incomplete;
  do not summarize it as a completed prepare.
- If selected skills are insufficient, report coverage notes and stop at
  handoff artifacts.

## Final Response

Return:

- Execution package root.
- `execution_prompt.md`.
- Planner validation path.
- Selected skills, warnings, coverage gaps, and blockers.

State that the final task has not been executed.
