---
name: skillfabric-doctor
description: Use when the user asks to check, diagnose, validate, or troubleshoot SkillFabric CLI availability, API configuration, plugin wiring, or workspace readiness in Claude Code.
license: MIT
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
- Do not display shell environment values or raw config payloads beyond
  present/missing field names.

Treat CLI JSON as canonical for configuration status.

## Workflow

1. Resolve `env_file` and `workspace`.
2. Run `which skillfabric` or equivalent command lookup.
3. Run `skillfabric --help` only far enough to confirm the CLI is callable.
4. Run `skillfabric init --check --json --env-file $env_file`.
5. Check whether `$workspace/status.json` exists. If it is missing, report the
   workspace as "not built yet", not as a CLI or API failure.
6. If the workspace status file exists, read only non-secret status fields
   needed to summarize readiness.
7. If configuration is incomplete, report missing field names and the exact
   terminal command: `skillfabric init --env-file $env_file`.

## Failure Handling

- If `skillfabric` is missing, tell the user to install `skillfabric-ai[claude]`
  and verify with `which skillfabric` plus `skillfabric --help`.
- If config check returns non-JSON output, report that the check failed and
  include only non-secret error text.
- If the workspace status file is missing, report that this is normal before
  the first build and that `/skillfabric:build` is needed.
- If the workspace status says `failed`, report the status path and the
  non-secret error summary when present.

## Final Response

Return:

- CLI availability.
- Env file checked.
- Whether API-backed builds are configured.
- Missing field names when incomplete.
- Workspace status path and readiness.
- Next command: `/skillfabric:build`, `/skillfabric:prepare`, or
  `/skillfabric:run`.
