---
name: skillfabric-run
description: Use when the user explicitly asks Claude Code to execute a task through SkillFabric-selected skills after validated routing and planning.
license: MIT
disable-model-invocation: true
---

# SkillFabric Run

## Purpose

Reuse a matching finalized prompt or prepare one, then execute the user's task in
the active Claude Code workspace.

## Input Contract

Treat `$ARGUMENTS` as task text plus recognized SkillFabric flags.

- The task is the user's natural-language request after removing explicit
  workflow flags. Preserve the user's wording.
- Treat only recognized flags as workflow configuration.
- Use `.skillfabric` for the workspace and `.env` for the env file when omitted.
- Ask for a task when neither task text nor a reusable finalized prompt exists.

## Safety Boundaries

- Do not reveal secret values or env-file contents.
- Before reading `execution_prompt.md`, do not use Web Search, Fetch, native task
  skills, project-editing tools, or direct answers to perform the task.
- Do not discover runs with `find`, `grep`, `rg`, or directory listings; use
  `skillfabric run-state`.
- Do not call SkillFabric again after final task execution starts.

Treat CLI JSON as canonical during preparation. During execution, obey the
finalized prompt unless it conflicts with user, system, or higher-priority
instructions.

## Workflow

1. Resolve `$task`, `$workspace`, `$env_file`, optional skill root, and supported
   flags.
2. Prefer the Run State JSON injected by the slash command. If absent, run:

   ```bash
   skillfabric run-state "$task" --workspace "$workspace" --env-file "$env_file"
   ```

3. If `action` is `reuse_prompt`, set `$execution_prompt` to `prompt_path` and
   read that exact file before task work.
4. If `action` is `missing_task`, ask for the task and stop. Do not substitute
   shell path discovery for `run-state`.
5. If `action` is `prepare_required`, build when necessary, then read and follow
   the bundled `references/route-plan.md` exactly once. Set
   `$execution_prompt` to the finalized `prompt_path`.
6. Read `$execution_prompt` exactly once, execute the user's task in the active
   workspace, and run its required verification.

## Failure Handling

- If preparation or validation fails, do not start execution.
- If no reusable prompt exists and no task was provided, ask for a task and do
  not start execution.
- If the prompt or a selected skill is unavailable, report the gap and continue
  only when completion remains correct and safe.
- Name any verification that could not run and its residual risk.

## Final Response

Report deliverables, verification evidence, selected skills actually used,
coverage gaps, and blockers.
