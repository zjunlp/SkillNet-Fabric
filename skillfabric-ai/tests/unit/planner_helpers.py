from __future__ import annotations

import json
from pathlib import Path


def valid_planner_output(package_root: Path) -> dict[str, str]:
    route = json.loads((package_root / "route.json").read_text(encoding="utf-8"))
    selected_ids = [
        str(item["skill_id"])
        for item in route.get("selected_skills", [])
        if isinstance(item, dict) and item.get("skill_id")
    ]
    required_edges = [
        f"{item['before_skill']} -> {item['after_skill']}"
        for item in route.get("required_edges", [])
        if isinstance(item, dict) and item.get("before_skill") and item.get("after_skill")
    ]
    selected_text = "\n".join(f"- {skill_id}" for skill_id in selected_ids) or "- No selected skills."
    strategy_text = "\n".join(f"- {edge}" for edge in required_edges) or "- No hard dependencies."
    return {
        "execution_prompt": (
            "# Execution Prompt\n\n"
            f"## Objective\n{route.get('query', 'Execute the task.')}\n\n"
            f"## Selected Skills\n{selected_text}\n\n"
            f"## Execution Strategy\n{strategy_text}\n\n"
            "## Verification\nVerify the requested deliverables.\n\n"
            "## Final Report\nSummarize deliverables, checks, deviations, and blockers."
        )
    }
