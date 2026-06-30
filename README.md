# SkillFabric

SkillFabric turns a directory of agent skills into a graph-backed routing
workspace. It scans `SKILL.md` files, builds retrieval indexes, compiles skill
relationships, materializes a query wiki, routes user tasks to the right skill
set, and generates execution prompt packages for Claude Code or Codex.

```bash
pip install "skillfabric-ai[claude]"
```

```bash
skillfabric init --env-file .env
skillfabric help workflow
skillfabric help config
```

```bash
skillfabric build \
  --skill-root examples/skills \
  --workspace .skillfabric \
  --env-file .env

skillfabric route \
  "extract financial KPIs from a PDF report" \
  --workspace .skillfabric \
  --env-file .env

skillfabric plan \
  "extract financial KPIs from a PDF report" \
  --workspace .skillfabric \
  --env-file .env
```

The public defaults keep LLM-backed skill contracts, wiki summaries, and
community metadata, while relation and execution validation use selective
interface-first checks to avoid sending every candidate pair through full
`SKILL.md` prompts. Plain route/plan uses the Claude Code explorer by default;
pass `--skip-llm-router --explorer-backend fallback` only for deterministic
local smoke checks.

```bash
skillfabric build --skill-root examples/skills --skip-llm-validation --embedding-provider disabled --wiki-summary-mode off
skillfabric build --skill-root examples/skills --embedding-provider local --embedding-model-path /path/to/bge-large-en-v1.5
```

## Python SDK

```python
from skillfabric import SkillFabric

sf = SkillFabric(workspace=".skillfabric", env_file=".env")
sf.build("examples/skills")
route = sf.route("extract financial KPIs from a PDF report")
plan = sf.plan(route=route)
print(plan.prompt_path)
```

## Claude Code Plugin

The Claude Code plugin uses the installed `skillfabric` CLI for stable
artifacts, then uses Claude Code subagents for route-time wiki exploration and
plan-time workflow/prompt planning. The CLI remains the only component that
writes SkillFabric workspace artifacts.

```bash
claude --plugin-dir plugins/claude-code/skillfabric
```

Use `/skillfabric:prepare` to stop at a plan, or `/skillfabric:run` to route,
plan, and continue with the final task in the current Claude Code session. See
`docs/claude-code-plugin.md` for command details.

## Package Layout

```text
skillfabric-ai/                  # PyPI package
plugins/claude-code/skillfabric/ # Claude Code plugin skeleton
docs/                            # User and architecture docs
examples/                        # Tiny runnable skill pool
```

The CLI does not run a background executor. The Claude Code plugin can
explicitly continue into task execution through `/skillfabric:run`.
