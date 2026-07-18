# SkillFabric Claude Code Plugin

The SkillFabric plugin builds a semantic skill graph, selects an
evidence-grounded skill set for a task, generates an execution prompt, and lets
Claude Code execute only the finalized prompt.

The `skillfabric` CLI owns all writes under `.skillfabric`. The main Claude Code
session invokes the CLI but does not author route or planner JSON. Preparation
uses the single protocol in `references/route-plan.md`.

## Requirements

- Claude Code with plugin support.
- Python 3.11 or newer.
- The package and Claude SDK extra installed in Claude Code's shell:

```bash
pip install "skillfabric-ai[claude]"
which skillfabric
skillfabric --help
```

- A private API configuration:

```bash
skillfabric init --env-file .env
skillfabric init --check --json --env-file .env
```

Do not paste API keys into Claude Code. Keep them in a private shell, keychain,
or untracked `.env` file.

## Installation

Load the plugin for one session:

```bash
claude --plugin-dir /path/to/plugins/claude-code/skillfabric
```

For a user-level installation:

```bash
mkdir -p ~/.claude/skills
cp -R /path/to/plugins/claude-code/skillfabric ~/.claude/skills/skillfabric
claude plugin validate --strict ~/.claude/skills/skillfabric
claude plugin list --json
```

## Quickstart

```text
/skillfabric:doctor --workspace .skillfabric
/skillfabric:build .claude/skills --workspace .skillfabric
/skillfabric:prepare "summarize this repo and identify release risks" --workspace .skillfabric
```

Use `/skillfabric:prepare` to produce `route.json` and `execution_prompt.md`
without executing the task. Use `/skillfabric:run` to
execute a finalized prompt. It reuses the latest prepared prompt when available
and prepares a new one when the task differs.

## Commands

- `/skillfabric:doctor` reports CLI, API, and workspace readiness.
- `/skillfabric:build` compiles contracts, semantic edges, indexes, wiki pages,
  and build diagnostics.
- `/skillfabric:prepare` performs validated route and prompt generation, then
  stops.
- `/skillfabric:run` reuses or prepares a validated prompt, then executes it in
  the active workspace.

The slash commands in `commands/*.md` are thin wrappers. Command behavior lives
in `skills/skillfabric-*/SKILL.md`; route and plan mechanics live once in the
shared reference.

## Local Smoke Test

Use a small skill directory for a real connectivity and cost smoke test. Wiki
summaries are derived deterministically from validated contracts:

```text
/skillfabric:build ./small-skill-set --workspace .skillfabric-smoke
```

The equivalent CLI command is:

```bash
skillfabric build \
  --skill-root ./small-skill-set \
  --workspace .skillfabric-smoke \
  --env-file .env
```

This remains a real API build: contract extraction, semantic pair judgment, and
dense embeddings still run. Use a disposable workspace and a small corpus when
checking connectivity or cost.

## Security Model

- The CLI is the only writer of registry, graph, route, planner validation, and
  prompt artifacts.
- Generated skill sources are untrusted data.
- Route selection reads only the bounded query wiki, not the active project.
- Planning reads only selected contracts and skill sources from the graph workspace.
- `.skillfabric` is an artifact store, not a runtime skill directory.
- The plugin installs no hooks, MCP servers, settings, or background tasks.
- Only `/skillfabric:run` proceeds to task execution.

## Troubleshooting

Validate plugin and CLI wiring:

```bash
claude plugin validate --strict ~/.claude/skills/skillfabric
claude plugin list --json
which skillfabric
skillfabric --help
skillfabric init --check --json --env-file .env
```

Common failures:

- `skillfabric: command not found`: install `skillfabric-ai[claude]` in the
  environment used by Claude Code.
- Missing API fields: run `skillfabric init --env-file .env`.
- Build provider or model error: verify configuration and rerun against a small
  skill root; do not bypass a failed semantic stage.
- Route validation error: inspect the non-secret validation artifact in the
  returned trace directory.
- Plan validation error: inspect `planner_validation.json` in the returned
  execution package.

## Uninstall

```bash
rm -rf ~/.claude/skills/skillfabric
claude plugin list --json
```
