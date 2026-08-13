# SkillFabric for Claude Code

SkillFabric helps Claude Code route tasks through a large local collection of native
`SKILL.md` files. It compiles the collection into reusable graph and index artifacts,
then lets Claude inspect a task specific Wiki before forming the final skill set.

## Commands

- `/skillfabric:doctor` checks CLI configuration and workspace readiness.
- `/skillfabric:build <skill-root>` builds or incrementally refreshes the default
  `.skillfabric` workspace.
- `/skillfabric:route <task>` selects relevant skills and reports evidence, near misses,
  and coverage gaps.

The plugin stops after selection. It does not execute the task, invoke selected skills,
or generate a plan. Claude Code remains responsible for using the selected skills.

## Requirements

Install the `skillfabric` CLI from the `skillfabric-ai` package and configure an LLM
endpoint plus an OpenAI-compatible embedding endpoint. The first build uses both APIs;
subsequent routes still use the embedding endpoint for the task query and the Claude
Agent SDK for Task Wiki exploration.

## Install

For local development, load this directory with:

```bash
claude --plugin-dir /path/to/SkillFabric/plugins/claude-code/skillfabric
```

For a persistent installation, add the repository marketplace and install the plugin:

```bash
claude plugin marketplace add /path/to/SkillFabric/plugins/claude-code
claude plugin install skillfabric@skillfabric --scope user
```

Do not put API keys in Claude Code prompts. The CLI owns all workspace writes, and the
plugin only consumes its JSON output.
