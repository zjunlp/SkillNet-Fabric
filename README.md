<div align="center">

# SkillFabric

**Compile native agent skills into an evidence-grounded graph, select the right skills for each task, and produce one execution-ready prompt.**

[![GitHub stars](https://img.shields.io/github/stars/zjunlp/SkillNet-Fabric?style=social)](https://github.com/zjunlp/SkillNet-Fabric)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Status: Alpha](https://img.shields.io/badge/Status-Alpha-orange.svg)](https://github.com/zjunlp/SkillNet-Fabric)

[Quick Start](#quick-start) · [How It Works](#how-it-works) · [CLI](#cli) · [Python API](#python-api) · [Claude Code](#claude-code-plugin)

</div>

---

## Why SkillFabric?

Large skill collections create two practical problems for agents: the model cannot read every skill for every task, and file-level similarity alone does not explain how skills work together.

SkillFabric turns a local collection of native `SKILL.md` files into a reusable orchestration substrate:

- **Source-grounded compilation:** extract a strict contract from every skill and retain evidence for every accepted relationship.
- **Semantic skill graph:** distinguish prerequisites, useful compositions, close alternatives, and unrelated pairs.
- **Bounded retrieval:** combine BM25, dense embeddings, and validated operational edges without placing the full corpus in the route context.
- **Agentic selection:** let an explorer inspect a task-specific query wiki and choose the skills needed for the current task.
- **Constraint-preserving planning:** give the planner the selected contracts and complete skill sources, then combine its complete execution plan with the original task in one validated `execution_prompt.md`.
- **Explicit failure:** malformed model output, incompatible artifacts, missing credentials, and context overflow stop the workflow instead of silently generating a fallback.

The project is named **SkillFabric**. Its installable Python package is `skillfabric-ai`, and its CLI command is `skillfabric`.

---

## Quick Start

### Requirements

- Python 3.11 or newer.
- An LLM endpoint supported by LiteLLM for graph compilation and prompt planning.
- An OpenAI-compatible embedding endpoint.
- A Claude Agent SDK-compatible gateway for query-wiki exploration.

### Install

From the repository:

```bash
git clone https://github.com/zjunlp/SkillNet-Fabric.git SkillFabric
cd SkillFabric/skillfabric-ai
python -m pip install -e ".[claude]"
```

Or install the packaged distribution:

```bash
python -m pip install "skillfabric-ai[claude]"
```

### Configure

Create a private, untracked environment file and verify that the required fields are available:

```bash
skillfabric init --env-file .env
skillfabric init --check --json --env-file .env
```

The interactive command does not echo secret input. Do not commit `.env` files.

### Build and Route

```bash
skillfabric build \
  --skill-root /path/to/skills \
  --workspace .skillfabric \
  --env-file .env

skillfabric route \
  "Summarize this repository and identify release risks" \
  --workspace .skillfabric \
  --env-file .env
```

The shortest routing workflow is `init`, `build`, then `route`. `route` returns the selected
Skills and their evidence; it does not execute the task. The optional `plan` CLI remains available
for downstream runtimes that need a generated execution prompt. The bundled Claude Code plugin
provides only `doctor`, `build`, and `route`; it does not call `init` automatically and keeps
workspace and environment options at their CLI defaults.

Builds are incremental by default. Re-run the same `build` command after adding, editing, or
removing `SKILL.md` files: unchanged contracts, embeddings, and relation judgments are reused,
while affected skills and relationships are refreshed. The final graph, indexes, and Full Wiki
are still validated and published as one consistent workspace.

> [!NOTE]
> Build, route, and plan use external model services. Start with a small skill corpus when validating a new endpoint or its service limits.

---

## How It Works

```mermaid
flowchart LR
    A["Native SKILL.md files"] --> B["Registry and SkillContracts"]
    B --> C["BM25 and dense indexes"]
    B --> D["Candidate skill pairs"]
    D --> E["Semantic relation judge"]
    E --> F["Validated skill graph"]
    C --> G["Hybrid task retrieval"]
    F --> G
    G --> H["Bounded query wiki"]
    H --> I["Explorer selects skills"]
    I --> J["Validated planner"]
    J --> K["execution_prompt.md"]
```

### 1. Compile the Skill Corpus

The build pipeline:

1. Scans native skill directories and parses `SKILL.md` files.
2. Extracts one strict `SkillContract` per skill from the complete source.
3. Builds BM25 and contract-aware dense indexes.
4. Retrieves bounded candidate pairs from handoff intent, similarity, and explicit references.
5. Judges each candidate once and requires source evidence for every accepted edge.
6. Validates relation direction, uniqueness, and dependency acyclicity.
7. Materializes the canonical graph and the reusable Full Wiki.

Full Wiki cards are rendered deterministically from validated contracts. Wiki materialization makes no additional model calls and does not alter contract extraction, semantic relation judgment, or embeddings.

LLM-backed build jobs retry within the compiler. SkillFabric writes canonical graph and Wiki
artifacts without producing a separate billing or usage report.

### 2. Model Skill Relationships

| Relation | Meaning | Direction |
| :-- | :-- | :-- |
| `depend_on` | A concrete artifact, data, or state handoff | Producer → consumer |
| `compose_with` | Adjacent complementary stages in a reusable workflow | Workflow predecessor → successor |
| `similar_to` | Independent near alternatives for one shared subproblem; implementation may differ | Symmetric; canonical id order |

Candidate pairs without sufficient evidence are rejected and do not become graph edges. The graph supplies evidence to routing and planning; it does not force prerequisite closure, automatically enlarge the selected set, or dictate the final execution order.

Workspaces built with the previous relation direction are rejected and must be rebuilt; the runtime does not reinterpret legacy edges.

### 3. Route Through a Query Wiki

For each task, the router fuses BM25 and dense ranks, expands a bounded neighborhood over
`depend_on`, `compose_with`, and `similar_to`, and writes a task-specific query wiki. Dependency
propagation is bidirectional, workflow propagation favors predecessor-to-successor traversal, and
similarity edges use a lower symmetric traversal weight so alternatives can enter the candidate
set without dominating operational handoffs. The explorer can inspect only this bounded workspace
and returns a strict selection-only `SkillPackage` with cited pages.

The query wiki is materialized once. The explorer then receives at most two attempts by default; SDK failures, malformed responses, and packages that fail exact validation are retried without repeating retrieval or graph expansion.

When both endpoints of a `similar_to` edge are present, the Wiki also marks the lower-ranked
endpoint as an explicit alternative. The relation informs exploration but never forces either
endpoint into the final selected set.

### 4. Generate the Execution Prompt

Planning is an optional Python/CLI stage for downstream runtimes. The bundled Claude Code plugin
ends after route selection and does not create or consume this package.

The planner receives:

- the original task,
- the explorer's selected skills,
- relevant graph evidence,
- each selected `SkillContract`, and
- the complete source of every selected skill.

It produces a complete task-specific execution plan. SkillFabric places the original task first and the plan second in one `execution_prompt.md`. It does not create an intermediate workflow DAG or execute the task from the CLI.

Planner messages, selected skill context, and the token estimate are constructed once. The planner receives at most two attempts by default.

---

## CLI

| Command | Purpose | Example |
| :-- | :-- | :-- |
| `init` | Configure or inspect API settings | `skillfabric init --check --json --env-file .env` |
| `build` | Compile graph, indexes, and wiki artifacts | `skillfabric build --skill-root ./skills` |
| `route` | Retrieve candidates and select skills for a task | `skillfabric route "your task"` |
| `plan` | Generate a validated execution prompt | `skillfabric plan "your task"` |
| `query-wiki card` | Print one manifest-listed query-wiki card | `skillfabric query-wiki card <wiki> <skill-id>` |
| `doctor-state` | Report plugin and workspace readiness | `skillfabric doctor-state --workspace .skillfabric` |
| `run-state` | Resolve the latest finalized execution package | `skillfabric run-state --workspace .skillfabric` |

Use `skillfabric help workflow` for the short workflow or `skillfabric <command> --help` for the
documented options. Build, route, and plan print concise terminal summaries by default. Add
`--json` when a script or plugin needs the stable machine-readable payload. Batch progress is
quiet by default; pass `--llm-progress-every N` to `build` when observing a large build.

---

## Python API

The public facade exposes the same `build`, `route`, and `plan` stages:

```python
from skillfabric import SkillFabric

task = "Summarize this repository and identify release risks"
fabric = SkillFabric(workspace=".skillfabric", env_file=".env")

fabric.build("/path/to/skills")
route = fabric.route(task, max_selected_skills=8)
package = fabric.plan(task, route=route)

print([skill.skill_id for skill in route.selected_skills])
print(package.prompt_path)
```

Planning checks its estimated prompt size before calling the model. The planner
context and recovery policy use stable runtime defaults.

Routing uses the Claude explorer by default. An isolated Codex app-server backend can be injected
through the same public API:

```bash
python -m pip install "skillfabric-ai[codex]"
```

This extra installs the stable official `openai-codex` SDK and its matching Codex runtime.

```python
from skillfabric import SkillFabric
from skillfabric.wiki.explorer.backends import CodexWikiExplorerBackend

task = "extract KPIs from the supplied report"
backend = CodexWikiExplorerBackend(
    env_file=".env",
    max_selected_skills=5,
    model="your-codex-model",
    reasoning_effort="medium",
)
route = SkillFabric(workspace=".skillfabric").route(
    task,
    max_selected_skills=5,
    explorer_backend=backend,
)
```

The backend creates a fresh ephemeral thread per attempt, exposes the query wiki read-only, and
accepts results only when the event stream stays within its declared command policy. Shared
explorer validation owns bounded recovery, so the backend does not add a nested retry loop.

---

## Claude Code Plugin

The bundled plugin provides three user-facing commands:

- `/skillfabric:doctor` checks CLI, API, and workspace readiness.
- `/skillfabric:build` compiles a skill corpus.
- `/skillfabric:route` selects relevant native Skills for one task and reports the evidence.

The plugin stops after routing. It does not call the optional `plan` CLI, generate an execution
prompt, execute the task, or manage historical run state. Claude Code remains responsible for
loading and using the selected native Skills.

Load the plugin directly from a clone:

```bash
claude --plugin-dir /path/to/SkillFabric/plugins/claude-code/skillfabric
```

Then run:

```text
/skillfabric:doctor
/skillfabric:build .claude/skills
/skillfabric:route "Summarize this repository and identify release risks"
```

For a user-level installation:

```bash
claude plugin marketplace add /path/to/SkillFabric/plugins/claude-code
claude plugin install skillfabric@skillfabric --scope user
claude plugin list --json
```

The marketplace source is `plugins/claude-code/.claude-plugin/marketplace.json`. For local
development without a persistent installation, use `claude --plugin-dir` as shown above.

Do not paste API keys into Claude Code. The CLI is the only writer of registry, graph, route, and
wiki artifacts. Generated skill sources are treated as untrusted data, and routing reads only the
bounded query Wiki. The plugin installs no hooks, MCP servers, settings, or background tasks. It
only reports a validated Skill selection; it does not execute the task.

To diagnose installation or configuration:

```bash
which skillfabric
skillfabric --help
skillfabric init --check --json --env-file .env
claude plugin validate /path/to/SkillFabric/plugins/claude-code/skillfabric
claude plugin list --json
```

Common failures are explicit:

- `skillfabric: command not found`: install `skillfabric-ai[claude]` in the environment used by
  Claude Code.
- Missing API fields: run `skillfabric init --env-file .env`. `doctor` reports LLM and
  embedding readiness separately; the embedding key may fall back to `API_KEY`.
- Build provider or model failure: verify the endpoint against a small disposable skill root; do
  not bypass a failed semantic stage.
- Route validation failure: report the non-secret route error and rebuild the workspace when its
  status is not ready.

To uninstall the user-level plugin, run:

```bash
claude plugin uninstall skillfabric@skillfabric --scope user
claude plugin list --json
```

---

## Configuration

| Variable | Purpose | Default |
| :-- | :-- | :-- |
| `API_KEY` | LLM authentication | unset |
| `BASE_URL` | LLM endpoint used by LiteLLM and the SDK adapter | `https://api.openai.com/v1` |
| `MODEL` | Contract, relation, and planner model | `openai/responses/gpt-5.4-mini` |
| `EMBEDDING_API_KEY` | Embedding authentication; falls back to `API_KEY` | unset |
| `EMBEDDING_BASE_URL` | OpenAI-compatible embedding endpoint; falls back to `BASE_URL` | provider default |
| `EMBEDDING_MODEL` | Embedding model identifier | `openai/text-embedding-3-small` |
| `EMBEDDING_DIMENSION` | Embedding vector dimension | `1536` |
| `EMBEDDING_BATCH_SIZE` | Maximum texts sent in each embedding request | `64` |
| `EMBEDDING_CONCURRENCY` | Concurrent embedding batch requests | `1` |
| `EMBEDDING_TIMEOUT` | Per-request embedding timeout in seconds | `120` |
| `EMBEDDING_TEXT_CHARS` | Maximum characters retained per embedding input; `0` disables truncation | `4000` |
| `EMBEDDING_MAX_RETRIES` | Retries for each embedding batch | `2` |
| `SKILLFABRIC_MAX_SELECTED_SKILLS` | Maximum explorer selection size | `8` |

Project-specific LLM aliases such as `SKILLFABRIC_LLM_API_KEY`, `SKILLFABRIC_LLM_API_BASE`, and
`SKILLFABRIC_LLM_MODEL` are also supported. Build concurrency, retries, rate limits, batching,
checkpointing, circuit breaking, graph expansion, and explorer recovery use stable internal
defaults; they are intentionally omitted from the routine configuration surface.

---

## Workspace Artifacts

SkillFabric writes generated state under the configured workspace:

```text
.skillfabric/
├── graph/
│   ├── registry.jsonl
│   ├── contracts.jsonl
│   ├── relation_decisions.jsonl
│   ├── graph.json
│   ├── bm25.sqlite
│   └── embeddings.json
├── wiki/
│   ├── index.md
│   ├── manifest.json
│   ├── health.md
│   └── skills/
├── runs/<trace-id>/
│   ├── query.json
│   ├── query_wiki/
│   ├── route.json
│   └── execution_package/
│       ├── planner_validation.json
│       └── execution_prompt.md
└── status.json
```

Artifacts use exact validated fields. Rebuild a workspace when canonical artifact validation fails instead of mutating generated files by hand.

The `runs/<trace-id>/execution_package` subtree is created only by the optional `plan` stage; it is
not required for routing or used by the bundled plugin.

---

## Repository Layout

```text
SkillFabric/
├── skillfabric-ai/                  # Python package, CLI, and tests
├── plugins/claude-code/skillfabric/ # Claude Code plugin
├── README.md
└── LICENSE
```

---

## Development

```bash
git clone https://github.com/zjunlp/SkillNet-Fabric.git SkillFabric
cd SkillFabric/skillfabric-ai
python -m pip install -e ".[dev,claude]"

python -m compileall -q src tests
python -m pytest -q
python -m ruff check src tests
python -m ruff format --check src tests
python -m build
```

The published test suite is deterministic and does not invoke real model or SDK services.
Validate provider integrations separately in a controlled environment with a disposable skill corpus.

The suite covers native skill parsing, contract extraction, semantic projection, graph validation,
hybrid retrieval, all three relation types, query-wiki generation, explorer package validation,
planner finalization, public CLI and Python APIs, artifact caching, plugin
behavior, and secret handling.

Contributions should keep public interfaces stable, include focused tests, and avoid committing credentials, generated workspaces, or run artifacts.

---

## License

SkillFabric is released under the [MIT License](LICENSE).
