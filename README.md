<div align="center">

# SkillFabric

**Task Specific Wikis for Agentic Skill Routing**

[![GitHub stars](https://img.shields.io/github/stars/zjunlp/SkillNet-Fabric?style=social)](https://github.com/zjunlp/SkillNet-Fabric)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv)](https://arxiv.org/abs/2603.04448)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-FFD21E)](https://huggingface.co/blog/xzwnlp/skillnet)
[![Website](https://img.shields.io/badge/Website-skillnet.openkg.cn-0078D4.svg)](http://skillnet.openkg.cn/)

[Website](http://skillnet.openkg.cn/) · [Quick Start](#quick-start) · [Architecture](#architecture) · [CLI](#cli) · [Agent Integrations](#agent-integrations) · [Paper](https://arxiv.org/abs/2603.04448)

</div>

---

## Why SkillFabric?

Finding the right Skills becomes harder as a local library grows. A task may need several
complementary capabilities, and useful connections are often spread across different `SKILL.md`
files. SkillFabric organizes that library once, then prepares a focused Task Wiki for each task.

SkillFabric provides a Python SDK and CLI for Wiki-based Skill routing:

- **Build:** compile native `SKILL.md` files into source grounded contracts, a typed graph,
  retrieval indexes, and a reusable Full Wiki.
- **Update:** rebuild incrementally as Skills are added, edited, or removed.
- **Route:** retrieve relevant candidates, expand useful relations, and return a compact Skill set
  with reasons and evidence.
- **Plan:** optionally turn a validated route into an execution prompt for a downstream agent.

The package is `skillfabric-ai`; the command-line entry point is `skillfabric`. Agent integrations
use the same build and routing pipeline described below.

---

## Quick Start

### Requirements

- Python 3.11 or newer.
- An LLM endpoint supported by [LiteLLM](https://docs.litellm.ai/) for `build` and `plan`.
- An OpenAI-compatible embedding endpoint for `build` and `route`.
- The [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python) for Claude
  exploration, or the official Codex SDK for Codex exploration.

### Install From Source

Install the current checkout:

```bash
git clone https://github.com/zjunlp/SkillNet-Fabric.git SkillFabric
cd SkillFabric/skillfabric-ai
python -m pip install -e ".[claude]"
```

Add the Codex Explorer from the same checkout when needed:

```bash
python -m pip install -e ".[codex]"
```

### Prepare a Skill Library

SkillFabric reads one native `SKILL.md` from each immediate child directory of the skill root. A
single `SKILL.md` file can also be passed directly as `--skill-root`.

```text
skills/
|-- data-inspection/
|   `-- SKILL.md
|-- visualization/
|   `-- SKILL.md
`-- slide-creation/
    `-- SKILL.md
```

Each file contains YAML frontmatter with a non-empty `name` and `description`, followed by the
Skill instructions. Source text remains available for evidence checks throughout routing.

### Configure Providers

Keep credentials in a private `.env` file. CLI summaries contain provider status and artifact paths;
secret values stay in the environment file.

```bash
skillfabric init --env-file .env
skillfabric init --check --json --env-file .env
```

The smallest common configuration is:

```text
API_KEY=...
BASE_URL=https://api.openai.com/v1
MODEL=openai/responses/gpt-5.4-mini
EMBEDDING_MODEL=openai/text-embedding-3-small
```

Use `EMBEDDING_API_KEY` and `EMBEDDING_BASE_URL` for a separate embedding service. Provider
specific gateways can use the environment variables supported by LiteLLM and the selected agent
SDK.

### Build and Route

```bash
skillfabric build \
  --skill-root ./skills \
  --workspace .skillfabric \
  --env-file .env

skillfabric route \
  "Analyze the dataset and prepare a presentation" \
  --workspace .skillfabric \
  --env-file .env
```

Builds are incremental. After adding, editing, or removing `SKILL.md` files, run the same `build`
command again. Unchanged contracts, embeddings, and relation judgments are reused; affected
Skills and relationships are refreshed, followed by a consistent Full Wiki publication.

Human-readable summaries are printed by default. Add `--json` for a stable result that can be
consumed by scripts and plugins:

```bash
skillfabric route --json "Analyze the dataset and prepare a presentation"
```

## News

- **[2026-08-20] SkillNet update.** The updated technical report introduces
  **SkillNet-Gym** for evaluating skill construction, retrieval, and composition, and
  **SkillNet-Fabric** for task time routing through a task specific Wiki.
  [Read the report](https://arxiv.org/abs/2603.04448).

## Architecture

SkillFabric has two phases: prepare the reusable Skill library, then localize it for each task.

### Build Time

The build stage converts the local Skill library into four reusable layers:

- **Skill Contracts** summarize each Skill's capability, use conditions, inputs, and outputs. Each
  field remains linked to supporting source lines.
- **Typed Relations** connect candidate pairs as `depend_on`, `compose_with`, or `similar_to` after
  source-grounded validation. Directed relations preserve producer and consumer roles; similarity
  identifies alternatives.
- **Canonical Graph and Indexes** store the machine-facing graph, BM25 index, dense embeddings,
  and relation evidence used during task time retrieval.
- **Full Wiki** gives the local library a navigable form. Each Skill page contains a compact card,
  the complete source, and links to related evidence.

### Task Time Localization

For a task query, hybrid retrieval combines lexical BM25 matches with dense embedding matches using
reciprocal rank fusion. The highest-ranked seeds are expanded through typed graph relations. The
retained candidates and the evidence that admitted them are projected into a task specific Wiki.

This projection keeps the routing space focused while preserving the complete sources needed for
final comparison:

```text
Full Wiki + graph/indexes
          |
          v
   retrieval + relation expansion
          |
          v
      Task Wiki
          |
          v
   Claude or Codex Explorer
          |
          v
      RouteResult
```

### Wiki Exploration

The Explorer starts from the Task Wiki index and contract cards, then reads complete sources for
the candidates under consideration. It records a concrete role and source evidence for every
selected Skill, reports near misses and coverage gaps, and returns at most the configured number
of Skills.

The optional Planner combines the original task with the validated route and produces one
execution prompt for a downstream agent. It operates after selection and keeps the route evidence
attached to the handoff.

![SkillFabric overview](images/overview.png)

### Relations Are Evidence

| Relation | Meaning | Direction |
| :-- | :-- | :-- |
| `depend_on` | A concrete artifact, data, or state handoff | Producer to consumer |
| `compose_with` | Complementary capabilities that fit together | Predecessor to successor |
| `similar_to` | Alternative Skills for a shared subproblem | Symmetric |

Relations support retrieval and candidate comparison. The Explorer forms the final Skill set from
the Task Wiki and its source evidence.

### Workspace

Generated state lives in the workspace passed to the CLI or Python API:

```text
.skillfabric/
|-- graph/                  # canonical graph and retrieval indexes
|-- wiki/                   # reusable Full Wiki
|-- runs/<trace-id>/        # Task Wiki and route or plan artifacts
`-- status.json
```

The CLI owns generated artifacts. Rebuild the workspace after changing the source library so the
graph, indexes, Full Wiki, and status remain aligned.

## Python SDK

```python
from skillfabric import SkillFabric

fabric = SkillFabric(workspace=".skillfabric", env_file=".env")
fabric.build("./skills")

route = fabric.route(
    "Analyze the dataset and prepare a presentation",
    max_selected_skills=5,
)
print([skill.skill_id for skill in route.selected_skills])
```

Select Codex as the Explorer when the Codex extra is installed:

```python
route = fabric.route(
    "Extract KPIs from the supplied report",
    backend="codex",
    max_selected_skills=5,
)
```

Create an execution package from a route when a downstream agent needs one:

```python
package = fabric.plan("Analyze the dataset and prepare a presentation", route=route)
print(package.prompt_path)
```

Provider settings come from the environment file passed to `SkillFabric`. The public facade keeps
the routine workflow small: build, route, and optional plan.

## CLI

The command-line surface is intentionally compact:

| Command | Purpose | Example |
| :-- | :-- | :-- |
| `init` | Create or inspect provider settings | `skillfabric init --check --json --env-file .env` |
| `build` | Build or incrementally refresh the graph and Full Wiki | `skillfabric build --skill-root ./skills` |
| `route` | Select Skills through a task specific Wiki | `skillfabric route "your task"` |
| `plan` | Create one execution prompt from a route | `skillfabric plan "your task"` |

Use `--backend codex` to select the Codex Explorer, `--max-selected-skills` to set the selection
limit, and `--json` when another program consumes the result. Run `skillfabric <command> --help`
for the command-specific options.

## Agent Integrations

### Claude Code

The bundled Claude Code plugin exposes three commands:

- `/skillfabric:doctor` checks provider configuration and workspace readiness.
- `/skillfabric:build <skill-root>` builds or incrementally refreshes the default workspace.
- `/skillfabric:route <task>` routes one task and reports selected Skills and evidence.

Load a local checkout while developing:

```bash
claude --plugin-dir /path/to/SkillFabric/plugins/claude-code/skillfabric
```

Install it through the repository marketplace for a persistent setup:

```bash
claude plugin marketplace add /path/to/SkillFabric/plugins/claude-code
claude plugin install skillfabric@skillfabric --scope user
```

Inside Claude Code:

```text
/skillfabric:doctor
/skillfabric:build .claude/skills
/skillfabric:route Analyze the dataset and prepare a presentation
```

The plugin calls the CLI with JSON output and leaves Skill execution to Claude Code after routing.

### Codex

Codex is available as the alternative Explorer backend. Build and Task Wiki artifacts are shared
across both backends; choose the Explorer that fits your environment:

```bash
cd SkillFabric/skillfabric-ai
python -m pip install -e ".[codex]"
skillfabric route \
  "Extract KPIs from the supplied report" \
  --backend codex \
  --workspace .skillfabric \
  --env-file .env
```

## Development

Run the deterministic test and lint suite from `skillfabric-ai`:

```bash
cd skillfabric-ai
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check src tests
python -m ruff format --check src tests
python -m compileall -q src
```

Provider integrations can be checked separately with a disposable Skill library and private
credentials. Keep credentials, generated workspaces, raw provider logs, and experiment outputs
outside commits.

## Contributing

Keep pull requests focused and include tests for behavior changes. Documentation should describe
the public workflow and keep implementation details in the architecture section.

## License

SkillFabric is released under the [MIT License](LICENSE).
