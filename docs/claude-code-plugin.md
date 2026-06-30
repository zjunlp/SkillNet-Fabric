# Claude Code Plugin

The Claude Code plugin lives at `plugins/claude-code/skillfabric`.

```bash
claude --plugin-dir plugins/claude-code/skillfabric
```

The plugin assumes `skillfabric` is already installed in the active shell:

```bash
pip install "skillfabric-ai[claude]"
```

Commands:

- `/skillfabric:doctor` checks CLI availability, API configuration, plugin
  wiring, and workspace readiness without exposing secret values.
- `/skillfabric:build` builds a workspace from a skill root.
- `/skillfabric:prepare` chains build-if-needed, route, and plan, then stops at
  a validated handoff package.
- `/skillfabric:run` chains build-if-needed, route, plan, and then executes the
  task from the generated `execution_prompt.md`.

Build requires API configuration because SkillFabric uses LLM validation,
embeddings, wiki summaries, and graph/KG artifacts. Configure it outside the
conversation:

```bash
skillfabric init --env-file .env
```

Route and plan commands use the current Claude Code session's subagent
support by default. The Python CLI prepares query-local wiki context, validates
the explorer's SkillPackage JSON, prepares selected-skill execution context,
validates the planner's workflow/prompt JSON, and writes stable route/plan
artifacts. The plugin passes subagent JSON back to the CLI through stdin; the
CLI performs bounded writes under `.skillfabric/runs/<trace_id>/`. The subagents
do not execute the final user task.
Only the main agent executes the final task, and only when the user invokes
`/skillfabric:run`.

The plugin has no hooks and no MCP server in v1. It stays explicit so
SkillFabric does not add background cost, automatic context injection, or
long-running tool servers unless the user asks for routing.

Future versions may add an opt-in hook installer for lightweight workspace
reminders and an MCP server for structured graph/wiki queries. Those should be
separate user-enabled surfaces, not default plugin behavior.
