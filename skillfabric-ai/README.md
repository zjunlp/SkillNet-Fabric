# skillfabric-ai

`skillfabric-ai` provides the SkillFabric compiler, retrieval router,
query wiki, and execution-prompt planner.

```bash
pip install "skillfabric-ai[claude]"
skillfabric init --env-file .env
skillfabric init --check --json --env-file .env
```

## Configuration

Project-prefixed settings are preferred; common OpenAI-compatible aliases are
also accepted:

```text
SKILLFABRIC_LLM_API_BASE=<api-base>
SKILLFABRIC_LLM_API_KEY=<api-key>
SKILLFABRIC_LLM_MODEL=<model>
SKILLFABRIC_LLM_REASONING_EFFORT=medium
EMBEDDING_MODEL=<embedding-model>
EMBEDDING_BASE_URL=<embedding-api-base>
EMBEDDING_API_KEY=<embedding-api-key>
EMBEDDING_DIMENSION=<vector-dimension>
EMBEDDING_BATCH_SIZE=64
EMBEDDING_CONCURRENCY=1
EMBEDDING_TIMEOUT=120
EMBEDDING_TEXT_CHARS=4000
EMBEDDING_MAX_RETRIES=2
```

Keep credentials in a private shell or untracked env file. CLI diagnostics
report only presence and source, never values.

## Build

```bash
skillfabric build \
  --skill-root /path/to/skills \
  --workspace .skillfabric \
  --env-file .env
```

The compiler performs these stages:

1. Scan and parse native `SKILL.md` files.
2. Extract strict, source-grounded skill contracts.
3. Build BM25 and dense indexes.
4. Retrieve bounded handoff, similarity, and explicit-reference candidate pairs.
5. Judge each candidate once, accepting `depend_on`, `compose_with`, or
   `similar_to`, or rejecting the pair when evidence is insufficient.
6. Validate evidence, relation direction, uniqueness, and dependency acyclicity.
7. Write canonical graph and build artifacts.

Directed relations use execution order. `depend_on` is producer to consumer;
`compose_with` is workflow predecessor to successor. `similar_to` is a symmetric
near-alternative relation for skills that independently solve the same explicit
subproblem with equivalent task-level results, even when providers, runtimes, or
toolchains differ. It is stored in canonical id order.

Wiki summaries are derived deterministically from validated contracts without an
additional model call. Contract extraction, pair judgment, and embeddings remain
LLM- or provider-backed build stages as configured.

## Route And Plan

```bash
skillfabric route "your task" --workspace .skillfabric --env-file .env
skillfabric plan "your task" --workspace .skillfabric --env-file .env
```

Routing uses reciprocal-rank fusion over BM25 and dense retrieval, then bounded
traversal over `depend_on` and `compose_with`, with relation-aware direction
weights. The explorer must return the exact
selection-only SkillPackage schema and cite query-wiki paths it read. Graph
relations remain evidence: they neither force skill selection nor impose final
execution order. Retrieval and query-wiki materialization run once; the explorer
retries only SDK, response, or package-validation failures. Planning constructs
its context once and retries only invalid returned content, then places the
original task and resulting execution plan in `execution_prompt.md`.

Each LiteLLM completion has one initial request and at most two retries for the
provider's retryable transport, timeout, rate-limit, and server failures. An
exhausted request is not submitted again by Build or Planner validation loops,
so network and response-validation retries cannot multiply each other.

## Python API

The public facade exposes `build`, `route`, and `plan`. `plan` requires the
original task even when a route is supplied:

```python
from skillfabric import SkillFabric

task = "extract KPIs from the supplied report"
sf = SkillFabric(workspace=".skillfabric", env_file=".env")
route = sf.route(task)
result = sf.plan(task, route=route, planner_context_max_tokens=100_000)
print(result.prompt_path)
```

Pass `llm_timeout_seconds=0` to disable the Planner request deadline. Omitting the option preserves
the timeout configured by `SKILLFABRIC_LLM_TIMEOUT` or the runtime default.

Routing uses the Claude Code explorer by default. Install the optional Codex
dependency and inject its backend when an isolated Codex app-server is required:

```python
from skillfabric import SkillFabric
from skillfabric.wiki.explorer.backends import CodexWikiExplorerBackend

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

The Codex backend creates a fresh ephemeral thread for each attempt, exposes the
query wiki read-only, disables network, Web, MCP, plugins, and Skills, and accepts
results only when the observed event stream stays within the declared
`exec_command` policy. The shared explorer owns package validation and bounded
recovery; the backend does not add another retry loop.

The context limit is checked before the Planner call. Each route and planner has
at most two attempts by default, and token/cost summaries include only accepted
attempts. Overflow, missing credentials, and exhausted provider or validation
failures stop explicitly; SkillFabric does not truncate context or generate a
fallback prompt.
