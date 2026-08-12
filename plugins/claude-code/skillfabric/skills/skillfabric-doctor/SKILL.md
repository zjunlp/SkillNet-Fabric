---
name: skillfabric-doctor
description: Use when the user asks to check, diagnose, validate, or troubleshoot SkillFabric CLI availability, API configuration, plugin wiring, or workspace readiness in Claude Code.
license: MIT
disable-model-invocation: true
---

# SkillFabric Doctor

## Purpose

Check whether the current Claude Code session is ready to use SkillFabric. This
skill reports CLI availability, non-secret API configuration status, and
workspace readiness. It does not build, route, plan, or execute a task.

## Input Contract

Treat `$ARGUMENTS` as optional diagnostic flags.

- Use `.env` when `--env-file` is omitted.
- Use `.skillfabric` when `--workspace` is omitted.
- Forward only the resolved `--env-file` value to `skillfabric init`.
- Do not treat free-form text as an API key, base URL, model, or workspace
  value unless it is attached to a supported flag.

## Safety Boundaries

- Do not reveal secret values or env-file contents.
- Do not ask the user to paste API credentials into the conversation.
- Do not run `skillfabric init` in interactive write mode.
- Do not build or mutate `.skillfabric`.
- Do not use `find`, `grep`, `rg`, `sed`, `cat`, or directory scans to inspect
  the workspace.
- Do not display shell environment values or raw config payloads beyond
  present/missing field names.

Treat CLI JSON as canonical for configuration status.

## Workflow

1. Prefer the Doctor State JSON injected by the slash command. If it is absent,
   run `skillfabric doctor-state --json $ARGUMENTS`.
2. Treat the returned JSON as the only status source.
3. Consume exactly `api_configured`, `missing_configuration`, `workspace_ready`,
   `build_id`, `skill_count`, and `next_action`.
4. If `api_configured` is false, report the names in `missing_configuration`
   and the exact terminal command: `skillfabric init --env-file <env_file>`.
5. If `workspace_ready` is false and `build_id` is empty, report "not built
   yet" and say this is normal before the first build. If `build_id` is
   present, report that the workspace is not reusable and must be rebuilt.
6. Use `next_action` to report the next command: `init` maps to
   `skillfabric init`, `build` maps to `/skillfabric:build`, and `ready` maps
   to `/skillfabric:prepare` or `/skillfabric:run`.

## Failure Handling

- If `skillfabric` is missing, tell the user to install `skillfabric-ai[claude]`
  and verify with `which skillfabric` plus `skillfabric --help`.
- If config check returns non-JSON output, report that the check failed and
  include only non-secret error text.
- If any required state field is missing or has the wrong type, report a
  plugin/CLI contract mismatch. Rerun the canonical doctor-state command
  without inspecting workspace files directly.

## Final Response

Write a concise readiness summary, not a raw JSON field dump.

Preferred format:

```text
SkillFabric ready.

- CLI: available
- API config: complete via <env_file>
- Workspace: ready, <skill_count> skills, build <build_id>
- Next: /skillfabric:prepare or /skillfabric:run
```

If incomplete, keep the same shape but replace the affected line with the
missing field names or "Workspace: not built yet". Do not dump the raw JSON
unless the user asks for details.
