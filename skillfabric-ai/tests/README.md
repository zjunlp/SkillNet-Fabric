# Tests

The suite covers offline fixture builds and opt-in real integration checks.

```text
tests/
├── unit/         # deterministic contracts, graph, routing, planner, CLI, package
├── integration/  # explicitly gated real SDK/API checks
└── fixtures/     # small native skill corpus
```

Unit coverage includes:

- Native skill parsing and strict contract extraction.
- Contract-aware candidate retrieval and semantic pair validation.
- Relation projection, dependency cycles, and canonical artifacts.
- BM25/dense rank fusion and bounded operational-edge expansion.
- Query-wiki generation, explorer SkillPackage validation, and route artifacts.
- Planner schema, dependency preservation, and prompt-package finalization.
- CLI, Python facade, Claude Code plugin, caching, usage, and secret handling.

Common local checks:

```bash
conda activate skillfabric_env
cd SkillNet-Fabric/skillfabric-ai

python -m compileall -q src tests
python -m pytest tests/unit -q
```

Use an existing environment that includes Ruff for linting; do not install or
modify environments solely for a test run.
