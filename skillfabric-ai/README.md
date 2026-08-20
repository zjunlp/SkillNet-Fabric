# skillfabric-ai

SkillFabric helps Claude Code and Codex choose Skills from a local library of native `SKILL.md`
files. Build the library once, route each task through a focused Task Wiki, and receive a small
set of Skills with source evidence.

## Install

```bash
python -m pip install "skillfabric-ai[claude]"
```

Add the Codex Explorer when needed:

```bash
python -m pip install "skillfabric-ai[codex]"
```

## Quick Start

Create a private `.env` file with an LLM and embedding configuration, then run:

```bash
skillfabric build --skill-root ./skills --workspace .skillfabric --env-file .env
skillfabric route "Analyze the dataset and prepare a presentation"
```

Builds reuse unchanged contracts, embeddings, and relation judgments when the Skill library is
updated. Add `--json` when another program consumes the result.

## Python API

```python
from skillfabric import SkillFabric

fabric = SkillFabric(workspace=".skillfabric", env_file=".env")
fabric.build("./skills")
route = fabric.route("Analyze the dataset and prepare a presentation")
print([skill.skill_id for skill in route.selected_skills])
```

The full project documentation and Claude Code plugin are available at:

<https://github.com/zjunlp/SkillNet-Fabric>
