# SkillFabric

SkillFabric compiles native agent skills into an evidence-grounded semantic
graph, retrieves a bounded candidate set for each task, and generates one
validated execution prompt.

```bash
pip install "skillfabric-ai[claude]"
skillfabric init --env-file .env
```

## Workflow

```bash
skillfabric build \
  --skill-root /path/to/skills \
  --workspace .skillfabric \
  --env-file .env

skillfabric route \
  "summarize this repository and identify release risks" \
  --workspace .skillfabric \
  --env-file .env

skillfabric plan \
  "summarize this repository and identify release risks" \
  --workspace .skillfabric \
  --env-file .env
```

Build extracts one strict `SkillContract` per skill, retrieves bounded candidate
pairs with contract-aware embeddings, and asks one semantic judge to assign
exactly one of `depend_on`, `compose_with`, `similar_to`, or `none`. Every
accepted edge requires source evidence. `depend_on` is stored from dependent to
prerequisite; `compose_with` is symmetric operational context; `similar_to` is
used only to present close alternatives.

The build writes schema-v2 artifacts under `.skillfabric`:

- `graph/registry.jsonl`
- `graph/contracts.jsonl`
- `graph/relation_decisions.jsonl`
- `graph/graph.json`
- `graph/bm25.sqlite`
- `graph/embeddings.json`
- `reports/build_summary.json`
- `reports/llm_usage.jsonl`
- `status.json`

Routing fuses BM25 and dense ranks, expands only validated operational edges,
materializes a bounded query wiki, and lets the explorer return one strict
selection-only SkillPackage. Graph relations provide task-time evidence; they do
not force prerequisite closure or expand the selected set. Invalid explorer or
planner output stops the workflow.

## Python SDK

```python
from skillfabric import SkillFabric

task = "summarize this repository and identify release risks"
sf = SkillFabric(workspace=".skillfabric", env_file=".env")
sf.build("/path/to/skills")
route = sf.route(task)
result = sf.plan(task, route=route)
print(result.prompt_path)
```

The Planner receives every selected contract and full skill source in one
bounded LLM call. It decides whether graph relation evidence matters for the
task and writes only `execution_prompt.md`; no intermediate workflow DAG is
created.

## Claude Code Plugin

```bash
claude --plugin-dir plugins/claude-code/skillfabric
```

Use `/skillfabric:prepare` to stop after a validated execution prompt. Use
`/skillfabric:run` to continue into task execution. The CLI remains the sole
writer of SkillFabric artifacts.

## Package Layout

```text
skillfabric-ai/                  # Python package and CLI
plugins/claude-code/skillfabric/ # Claude Code plugin
```
