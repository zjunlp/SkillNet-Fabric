---
name: skillfabric
description: Use when the user asks what SkillFabric is, how to use the SkillFabric Claude Code plugin, or needs help choosing between SkillFabric doctor, build, prepare, and run workflows.
license: MIT
---

# SkillFabric

SkillFabric builds a graph-backed workspace from native Claude Code skills,
then uses that workspace to prepare or run task-specific skill workflows.

## Command Choice

- `/skillfabric:doctor`: check CLI availability, API configuration, plugin
  wiring, and workspace readiness.
- `/skillfabric:build`: build or refresh `.skillfabric` from a skill root.
- `/skillfabric:prepare`: select skills and generate a handoff package without
  executing the final task.
- `/skillfabric:run`: select skills, generate a plan, and execute the task in
  the current Claude Code session.

## Boundaries

- Prefer the four slash commands for real work.
- Do not duplicate command workflows in this overview skill.
- Do not expose `.env` contents or secret values.
- Treat `.skillfabric` as an artifact store, not a runtime skill directory.
