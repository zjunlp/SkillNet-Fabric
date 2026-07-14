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
near-alternative relation and is stored in canonical id order.

Use `--wiki-summary-mode off` to derive wiki summaries directly from validated
contracts. This does not skip contract extraction, pair judgment, or embeddings.

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
execution order. Planning uses one bounded LLM call over the selected contracts
and full sources, then writes only `execution_prompt.md`.

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

The context limit is checked before the Planner call. Overflow, malformed model
output, missing credentials, and provider failures stop explicitly; SkillFabric
does not truncate context or generate a fallback prompt.
