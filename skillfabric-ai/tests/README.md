# Tests

This directory covers offline unit tests, fixture skill pools, and small
integration checks. Real quality claims still need a fresh AgentSkillOS run
through the current route and execution prompt package mainline.

```text
tests/
├── unit/         # unit tests
├── integration/  # gated integration tests
└── fixtures/     # test skills, tasks, and graph fixtures
```

Current unit tests focus on:

- Task understanding and deliverable coverage.
- Artifact ontology constraints.
- Interface extraction, canonicalization, and execution compatibility.
- Community assignment repair, semantic relocation, rebalance, and graph health.
- Router coverage diagnostics, query_wiki assembly, SkillPackage validation,
  and Claude SDK prompt contracts.
- AgentRunSpec, execution prompt package generation, and eval aggregation
  across build, graph, execution, wiki, route, and orchestrator status.

Common local checks:

```bash
source "$HOME/Documents/Develop/Miniforge/etc/profile.d/conda.sh"
conda activate skillfabric_env
cd "$HOME/Documents/SkillFabric/skillfabric-public/skillfabric-ai"

python -m compileall -q src tests
python -m pytest tests/unit -q
python -m ruff check .
```
