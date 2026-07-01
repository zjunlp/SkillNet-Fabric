# Configuration

SkillFabric reads API configuration from the shell environment and, for CLI
commands, from `--env-file`.

```bash
skillfabric init --env-file .env
```

The generated env file uses generic names that are safe to keep in a private,
untracked configuration file:

```bash
API_KEY=sk-...
BASE_URL=https://api.openai.com/v1
MODEL=openai/gpt-4o-mini
EMBEDDING_MODEL=openai/text-embedding-3-small
```

Shell environment values take precedence over values loaded from `--env-file`.

SkillFabric v1 supports OpenAI-compatible chat/completion and embedding APIs
through LiteLLM. Vendor-specific SDKs such as Azure, Gemini, and Bedrock are
not part of the public API contract; they may work through LiteLLM-compatible
model strings and environment variables, but they are not documented as stable.

Do not paste API keys into Claude Code chat. Use `skillfabric init` in a
terminal or provide values through the shell environment.

## Embeddings

API embeddings are the default and share `API_KEY` and `BASE_URL` unless
embedding-specific values are set:

- `EMBEDDING_PROVIDER=api`
- `EMBEDDING_MODEL=openai/text-embedding-3-small`
- `EMBEDDING_API_KEY`
- `EMBEDDING_BASE_URL`

OpenAI-compatible fallback variables are also supported:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` or `OPENAI_API_BASE`

The Python CLI uses the `claude-agent-sdk` route explorer by default for
non-plugin route and plan workflows. Pass `--skip-llm-router --explorer-backend
fallback` only when you need deterministic local routing. The Claude Code plugin
uses current-session subagents instead of the SDK explorer path.

For local SentenceTransformer embeddings:

```bash
pip install "skillfabric-ai[local-embeddings]"
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
EMBEDDING_MODEL_PATH=/path/to/models/BAAI/bge-large-en-v1.5
```

Use this mode when your chat/completion endpoint is OpenAI-compatible but does
not provide an embeddings endpoint. The local model path is read during both
`build` and route-time query embedding, so keep it in the same `.env` passed to
`--env-file`.

Set `DISABLE_DENSE_EMBEDDINGS=1` or use `--embedding-provider disabled` only
for deterministic local smoke checks. The public build workflow expects
API-backed LLM validation, embeddings, wiki summaries, and graph/KG artifacts.
