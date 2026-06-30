# skillfabric-ai

`skillfabric-ai` is the Python package behind SkillFabric. It compiles agent
skill documents into a graph-backed workspace, builds query-local routing
context, routes tasks to selected skills, and emits execution prompt packages
for external coding agents.

```bash
pip install "skillfabric-ai[claude]"
```

```bash
skillfabric init --env-file .env
skillfabric help workflow
skillfabric help config
```

SkillFabric accepts the project-level API namespace used by the internal
experiments, while keeping the shorter public aliases compatible:

```text
SKILLFABRIC_LLM_API_BASE=<openai-compatible-base-url>
SKILLFABRIC_LLM_API_KEY=<api-key>
SKILLFABRIC_LLM_MODEL=openai/responses/gpt-5.4-mini
SKILLFABRIC_LLM_REASONING_EFFORT=medium
```

These values are bridged to the OpenAI-compatible LiteLLM path and to Claude
Code SDK `ANTHROPIC_*` runtime aliases. Keep real keys in a private shell,
conda, or untracked `.env` configuration.

Public builds keep LLM-backed skill contracts, wiki summaries, and community
metadata, while relation and execution validation use selective interface-first
checks to avoid sending every candidate pair through full `SKILL.md` prompts.
Plain route/plan uses fallback routing unless you explicitly request the Claude
Code explorer.

```bash
skillfabric build --skill-root examples/skills --skip-llm-validation --embedding-provider disabled --wiki-summary-mode off
skillfabric build --skill-root examples/skills --embedding-provider local --embedding-model-path /path/to/bge-large-en-v1.5
```

```python
from skillfabric import SkillFabric

sf = SkillFabric(workspace=".skillfabric", env_file=".env")
sf.build("examples/skills")
route = sf.route("extract financial KPIs from a PDF report")
plan = sf.plan(route=route)
print(plan.prompt_path)
```

The package does not include embedding model weights. API embeddings are the
default. Local SentenceTransformer embeddings are available through the
`local-embeddings` extra with `--embedding-provider local` and
`--embedding-model-path`.
