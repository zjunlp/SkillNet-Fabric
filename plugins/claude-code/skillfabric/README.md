# SkillFabric Claude Code Plugin

SkillFabric routes a user task through a graph-backed workspace of native
skills, then produces validated route, planner validation, and execution prompt
artifacts for Claude Code.

The `skillfabric` CLI is the artifact source of truth. Claude Code commands and
the main Claude Code session prepare inputs and perform bounded reasoning, but
the CLI owns writes to `.skillfabric`: registry, graph, wiki, runs, route
finalization, planner validation metadata, and `execution_prompt.md`.

## Requirements

- Claude Code with plugin support.
- Python package installed in the shell where Claude Code runs:

```bash
pip install "skillfabric-ai[claude]"
```

- The `skillfabric` CLI must be on `PATH`:

```bash
which skillfabric
skillfabric --help
```

- For API-backed builds, configure a private env file:

```bash
skillfabric init --env-file .env
skillfabric init --check --json --env-file .env
```

Do not paste API keys into Claude Code conversation context. Keep secrets in a
private shell, keychain, or untracked `.env`.

## Installation

For one session, load the plugin directly:

```bash
claude --plugin-dir /path/to/plugins/claude-code/skillfabric
```

For a user-level install, copy or symlink this directory under
`~/.claude/skills/skillfabric`, then restart Claude Code:

```bash
mkdir -p ~/.claude/skills
cp -R /path/to/plugins/claude-code/skillfabric ~/.claude/skills/skillfabric
claude plugin validate --strict ~/.claude/skills/skillfabric
claude plugin list --json
```

The plugin should appear as `skillfabric@skills-dir` and be enabled.

## Quickstart

Start Claude Code in a project, then run:

```text
/skillfabric:doctor --workspace .skillfabric
/skillfabric:build .claude/skills --workspace .skillfabric
/skillfabric:prepare "summarize this repo and identify release risks" --workspace .skillfabric
```

Use `/skillfabric:prepare` when you want a complete handoff package but do not
want Claude Code to execute the task:

```text
/skillfabric:prepare "write a migration plan for the auth module" --skill-root .claude/skills --workspace .skillfabric
```

Use `/skillfabric:run` only when you want Claude Code to execute a finalized
SkillFabric prompt. It reuses the latest prepared prompt when available, or
routes and plans the provided task first.

## Commands

- `/skillfabric:doctor` checks CLI availability, API configuration, plugin
  wiring, and workspace readiness without exposing secret values.
- `/skillfabric:build` builds graph, registry, wiki, and status artifacts from
  a skill root.
- `/skillfabric:prepare` builds when needed, routes, plans, and returns a
  handoff package without executing the final task.
- `/skillfabric:run` reads the latest finalized prompt when available; otherwise
  it builds when needed, routes, plans, reads the finalized prompt, and executes
  the task in the current Claude Code session.

Slash commands are thin wrappers under `commands/*.md`; command logic lives in
`skills/skillfabric-<command>/SKILL.md` using Claude Code's modern skill-backed
layout. The public command surface is intentionally small: `doctor`, `build`,
`prepare`, and `run`.

## Local Smoke Test

Use disabled embeddings and skip LLM validation when you want to verify local
plugin and CLI wiring without API calls:

```text
/skillfabric:build .claude/skills --workspace .skillfabric-smoke --skip-llm-validation --embedding-provider disabled --wiki-summary-mode off
```

The expected result is a JSON summary with:

- workspace path,
- skill count,
- registry and graph artifact paths,
- wiki artifact paths,
- warnings, if any.

Outside Claude Code, the same smoke check can be run directly:

```bash
skillfabric build \
  --skill-root .claude/skills \
  --workspace .skillfabric-smoke \
  --skip-llm-validation \
  --embedding-provider disabled \
  --wiki-summary-mode off
```

## Security Model

- The CLI owns canonical writes under `.skillfabric`.
- The plugin does not install hooks, start MCP servers, write Claude Code
  settings, or run background automation.
- Route exploration and prompt planning run inline in the main Claude Code
  session. They return JSON that must be accepted by CLI finalization before it
  is canonical.
- `.skillfabric` is an artifact store, not a runtime skill directory.
- Packaged route evidence is planner input only. Final task execution should
  follow `execution_prompt.md` and the active user request.
- Do not paste API keys, `.env` contents, or shell secrets into the conversation.
- Only `/skillfabric:run` continues into final task execution.

## Troubleshooting

If Claude Code does not show the plugin:

```bash
claude plugin validate --strict ~/.claude/skills/skillfabric
claude plugin list --json
```

If the plugin loads but commands fail:

```bash
which skillfabric
skillfabric --help
skillfabric init --check --json --env-file .env
```

Common failures:

- `skillfabric: command not found`: install `skillfabric-ai[claude]` in the
  environment used by Claude Code.
- Missing API fields: run `skillfabric init --env-file .env`.
- Build fails with provider or model errors: rerun the local smoke test with
  `--embedding-provider disabled --skip-llm-validation --wiki-summary-mode off`
  to separate plugin wiring from provider configuration.
- Route finalization fails: inspect the trace directory returned by
  `route --agent-mode prepare`; the CLI writes validation diagnostics there.
- Plan finalization fails: inspect the execution package root returned by
  `plan --agent-mode prepare`.

## Uninstall

For a user-level install:

```bash
rm -rf ~/.claude/skills/skillfabric
```

Restart Claude Code, then verify:

```bash
claude plugin list --json
```
