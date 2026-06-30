# Architecture

SkillFabric keeps the public interface small and the build artifacts auditable.

```text
SKILL.md pool
  -> registry
  -> BM25 + embedding indexes
  -> compiled skill graph
  -> interface and execution sidecars
  -> Markdown wiki
  -> query-local RouterBundle
  -> RouteResult
  -> external-agent execution package
```

The core boundary is deliberate:

- SkillFabric builds routing knowledge and execution prompt packages.
- It does not run the final user task.
- Claude Code and Codex consume the generated package as external agents.
- The Claude plugin calls the Python CLI instead of duplicating logic in
  prompt files.

Workspace outputs live under the configured `--workspace` directory and are
safe to delete and rebuild.
