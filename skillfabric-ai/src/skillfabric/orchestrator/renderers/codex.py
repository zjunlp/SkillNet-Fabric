"""Codex execution prompt renderer."""

from __future__ import annotations

from pathlib import Path

from skillfabric.orchestrator.agent_run_spec import AgentRunSpec
from skillfabric.orchestrator.renderers.claude_code import AgentEntryPrompt


def render_codex_entry_prompt(
    spec: AgentRunSpec,
    *,
    execution_package_root: Path,
) -> AgentEntryPrompt:
    """Render a Codex-specific entry prompt."""

    prompt = (
        "Codex entry prompt for SkillFabric execution.\n\n"
        f"Open `{execution_package_root / 'execution_prompt.md'}` and follow it as the primary task prompt.\n"
        "Use the selected capability roles included in that prompt as task guidance.\n"
        "Complete the user objective and finish with the final response requested in the brief.\n"
    )
    return AgentEntryPrompt(label="Codex SkillFabric execution", prompt=prompt)
