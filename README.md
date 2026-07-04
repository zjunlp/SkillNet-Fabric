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

The public defaults keep LLM-backed skill contracts, wiki summaries, and
community metadata, while relation and execution validation use selective
interface-first checks to avoid sending every candidate pair through full
`SKILL.md` prompts. Plain route/plan uses the Claude Code explorer by default;
pass `--skip-llm-router --explorer-backend fallback` only for deterministic
local smoke checks.

```bash
skillfabric build --skill-root /path/to/skills --skip-llm-validation --embedding-provider disabled --wiki-summary-mode off
```

## Python SDK

```python
from skillfabric import SkillFabric

sf = SkillFabric(workspace=".skillfabric", env_file=".env")
sf.build("/path/to/skills")
route = sf.route("summarize this repository and identify release risks")
package = sf.prepare_plan(route=route)
plan = sf.finalize_plan(
    package_root=package.root,
    planner_output={"execution_prompt": "Summarize the repository and verify the result."},
)
print(plan.prompt_path)
```

## Claude Code Plugin

The Claude Code plugin uses the installed `skillfabric` CLI for stable
artifacts, then uses the main Claude Code session for bounded route-time wiki
exploration and plan-time prompt planning. The CLI remains the only component
that writes SkillFabric workspace artifacts.

```bash
claude --plugin-dir plugins/claude-code/skillfabric
```

Use `/skillfabric:prepare` to stop at a plan, or `/skillfabric:run` to route,
plan, and continue with the final task in the current Claude Code session.

## Package Layout

```text
skillfabric-ai/                  # PyPI package
plugins/claude-code/skillfabric/ # Claude Code plugin skeleton
```

The CLI does not run a background executor. The Claude Code plugin can
explicitly continue into task execution through `/skillfabric:run`.
