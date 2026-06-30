"""Claude Code execution prompt renderer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from skillfabric.orchestrator.agent_run_spec import AgentRunSpec


@dataclass(slots=True)
class AgentEntryPrompt:
    """Rendered target-agent entry prompt."""

    label: str
    prompt: str

    def to_dict(self) -> dict[str, str]:
        return {"label": self.label, "prompt": self.prompt}


def render_execution_prompt(spec: AgentRunSpec, *, renderer: str = "claude-code") -> str:
    """Render the package's primary agent-facing execution prompt."""

    lines = [
        "# SkillFabric Execution Prompt",
        "",
        f"Target agent: {renderer}",
        "",
        "## TODO",
        "Execute the user's objective end-to-end in the active workspace. "
        "Use the selected capability roles below as task guidance, produce the requested deliverables, "
        "verify them with task-appropriate checks, and finish with a concise final response.",
        "",
        "## Input",
        "- objective: The concrete user task to execute.",
        "- selected capability roles: The task capabilities selected during planning.",
        "- execution_strategy: A semantic plan of allowed operations, dependencies, parallelization, delegation, verification, and reporting.",
        "",
        "## Objective",
        spec.objective,
        "",
        "## Output",
        "- Primary deliverables: satisfy the objective and acceptance criteria in the active workspace.",
        "- Completion state: continue through implementation, verification, and reporting until the requested deliverables exist in the active workspace.",
        "- Final response: concise summary of completed work, deliverables, verification evidence, skills actually used, coverage gaps, and blockers.",
        "- If a requested deliverable cannot be completed, preserve partial artifacts when useful and report the exact blocker.",
        "",
        "## Target Agent Guidance",
        *_renderer_guidance(renderer),
        "",
        "## Workflow",
        "Step 1: Orient. Map the objective to concrete deliverables, inputs, operations, constraints, support capabilities, and verification signals.",
        "Step 2: Inspect. Read the relevant workspace files and task inputs.",
        "Step 3: Confirm applicability. For each selected capability role, identify the task stage it covers, its prerequisites, tools, outputs, boundaries, and failure modes.",
        "Step 4: Plan execution internally. Use the execution strategy below to decide serial work, parallel work, delegation, aggregation, production, verification, and reporting.",
        "Step 5: Apply selected capability roles to their assigned stages.",
        "Step 6: Handle gaps. If selected roles do not cover part of the objective, solve that part directly with normal agent capabilities and record the coverage gap.",
        "Step 7: Produce deliverables directly in the active workspace. Keep generated artifacts and final reports in the active workspace unless the objective says otherwise.",
        "Step 8: Aggregate. Resolve overlaps, redundant intermediate outputs, and conflicts between skill-guided results before final verification.",
        "Step 9: Verify. Run relevant tests, builds, lint checks, schema checks, file inspections, rendering checks, metadata checks, or artifact-specific validators.",
        "Step 10: Report. Final response should include concrete paths, checks run, unresolved risks, and blocked verification.",
        "",
        "## Rules",
        "- Use this prompt as the primary task instruction.",
        "- Treat the objective as executable user work and selected roles as capability guidance.",
        "- Do not call back into SkillFabric for execution, step-level validation, automatic repair, or additional routing.",
        "- Do not perform a separate capability search, install unrelated capabilities, or add unsupported capabilities just because they sound useful.",
        "- Do not copy long reference text into outputs. Apply the operational guidance to the task.",
        "- Base task decisions on inspected inputs, role guidance, tool outputs, and artifact checks.",
        "- Do not assume a role completed work until the workspace shows the requested artifact, state, or verification result.",
        "",
        "## Constraints",
        "- Keep file edits and generated artifacts scoped to the active workspace unless the objective explicitly says otherwise.",
        "- Preserve unrelated user changes. Do not revert files you did not need to touch.",
        "- Keep delegated or parallel work bounded, independent, and non-overlapping in write scope.",
        "- Report exact blockers for missing tools, credentials, network, permissions, unavailable SDKs, or environment constraints.",
        "- If verification is partial, say what remains unverified and why.",
        "",
        "## Action Type Definitions",
        "- orient: Understand objective, selected skills, deliverables, constraints, and verification signals before changing files.",
        "- inspect: Read task inputs and workspace context before applying selected capability roles.",
        "- apply_skill: Apply one selected capability role to its assigned task stage after confirming relevance.",
        "- produce: Create or modify the concrete requested deliverables.",
        "- verify: Check deliverables and code/artifacts with task-appropriate commands or inspections.",
        "- delegate: Use a bounded subagent/task only for independent exploration, verification, or disjoint implementation work.",
        "- parallelize: Run independent reads, checks, exploration, or disjoint work concurrently when there is no ordering dependency.",
        "- aggregate: Merge independent results into coherent final deliverables and remove redundant intermediate artifacts when appropriate.",
        "- report: Write the concise final response with evidence.",
        "",
        "## Skill Use Protocol",
        "- Treat this section as skill-use guidance for executing the task, not as a search trace or explanation to the end user.",
        "- Start by decomposing the objective into capability facets: task domain, input artifacts, output artifacts, required operations, constraints, support capabilities, and verification signals.",
        "- For each selected role, identify the specific stage or facet it covers and the boundary of what it does not cover.",
        "- Compare similar selected roles by their listed responsibilities and use the sharper one for each stage unless both are complementary.",
        "- Combine roles only when their responsibilities are complementary for this task. Avoid redundant work when two roles overlap.",
        "- If a selected role is not actually applicable after inspection, skip it and record the reason in the final response.",
        "- If the selected roles leave a coverage gap, handle that part directly with normal agent capabilities and record the gap.",
        "- Do not copy long reference text into outputs. Apply the operational guidance to the task.",
        "",
        "## Selected Skills",
    ]
    for skill in spec.selected_skills:
        lines.append(f"- {skill.skill_id} (`{skill.native_skill_name}`): {skill.role}")
    lines.extend(["", "## Execution Strategy"])
    for operation in spec.execution_strategy.operations:
        lines.append(f"### {operation.id}: {operation.operation}")
        lines.append(f"- Control: {operation.control}")
        lines.append(f"- Goal: {operation.goal}")
        if operation.skill_ids:
            lines.append(f"- Skills: {', '.join(operation.skill_ids)}")
        if operation.depends_on:
            lines.append(f"- Depends on: {', '.join(operation.depends_on)}")
        if operation.guidance:
            lines.append(f"- Guidance: {operation.guidance}")
        if operation.expected_outputs:
            lines.append(f"- Expected outputs: {', '.join(operation.expected_outputs)}")
    lines.extend(
        [
            "",
            "## Strategy Policies",
            f"- Parallelization: {spec.execution_strategy.parallelization_policy}",
            f"- Delegation: {spec.execution_strategy.delegation_policy}",
            f"- Verification: {spec.execution_strategy.verification_policy}",
            "- Evidence discipline: Base task decisions on inspected inputs, role guidance, tool outputs, and artifact checks. Do not assume a selected role completed work until the workspace shows it.",
        ]
    )
    lines.extend(["", "## Phases"])
    for phase in spec.phases:
        lines.append(f"### {phase.id}")
        lines.append(f"- Goal: {phase.goal}")
        lines.append(f"- Skills: {', '.join(phase.skill_ids)}")
        if phase.depends_on:
            lines.append(f"- Depends on: {', '.join(phase.depends_on)}")
        if phase.expected_outputs:
            lines.append(f"- Expected outputs: {', '.join(phase.expected_outputs)}")
        if phase.guidance:
            lines.append(f"- Guidance: {phase.guidance}")
    lines.extend(["", "## Required Order"])
    if spec.required_order:
        for edge in spec.required_order:
            lines.append(f"- {edge.before_skill} before {edge.after_skill}: {edge.reason}")
    else:
        lines.append("- None")
    lines.extend(["", "## Acceptance Criteria"])
    if spec.acceptance_criteria:
        for criterion in spec.acceptance_criteria:
            artifacts = ", ".join(criterion.expected_artifacts) or "final task output"
            covered_by = ", ".join(criterion.covered_by) or "selected skills"
            lines.append(f"- {criterion.id}: {criterion.description} Expected: {artifacts}. Covered by: {covered_by}.")
    else:
        lines.append("- Complete the objective and report the final deliverables.")
    lines.extend(
        [
            "",
            "## Verification Protocol",
            "- Verify every requested deliverable exists at the expected path or explain the deviation.",
            "- Inspect generated artifacts with format-appropriate checks when possible, such as opening metadata, counting pages/slides/sheets, rendering images, running tests, or validating schemas.",
            "- Run relevant commands from the workspace after edits or generation. Prefer task-specific checks over generic success claims.",
            "- If verification is blocked by missing tools, credentials, network, or environment constraints, report the exact blocker and what remains unverified.",
        ]
    )
    lines.extend(["", "## Constraints"])
    for constraint in spec.constraints:
        lines.append(f"- {constraint}")
    lines.extend(
        [
            "",
            "## Final Report",
            "After producing and verifying the requested deliverables, respond with a concise summary of changed or produced artifacts, verification evidence, unresolved risks, and blockers.",
            "Do not call back into SkillFabric for step-level validation or automatic repair.",
            "",
        ]
    )
    return "\n".join(lines)


def _renderer_guidance(renderer: str) -> list[str]:
    if renderer == "codex":
        return [
            "- Start by using `update_plan` for visible progress tracking.",
            "- Inspect with `exec_command` and standard commands such as `rg`, `sed`, `ls`, and targeted test commands.",
            "- Prefer `apply_patch` for file edits.",
            "- Use parallel tool calls for independent reads or checks when practical.",
            "- Use `spawn_agent` only for bounded, non-blocking, independent exploration, verification, or disjoint implementation work; use `wait_agent` only when the result is needed.",
            "- Keep generated artifacts and final reports in the active workspace unless the objective says otherwise.",
        ]
    if renderer == "claude-code":
        return [
            "- Start by using `TodoWrite` for visible progress tracking.",
            "- Inspect with `Read / Grep / Glob / LS`; use `Bash` for tests, builds, and shell checks.",
            "- Edit with `Edit / MultiEdit / Write`; use `Bash` only when it is the right tool.",
            "- Use `Task` only for bounded, independent work that can run separately from the critical path.",
            "- Launch independent `Task` agents in parallel only when their scopes do not overlap.",
            "- Keep generated artifacts and final reports in the active workspace unless the objective says otherwise.",
        ]
    return [f"- Unknown renderer `{renderer}`; follow the semantic execution strategy below."]


def render_claude_code_entry_prompt(
    spec: AgentRunSpec,
    *,
    execution_package_root: Path,
) -> AgentEntryPrompt:
    """Render a Claude Code-specific entry prompt."""

    prompt = (
        "Claude Code entry prompt for SkillFabric execution.\n\n"
        f"Read `{execution_package_root / 'execution_prompt.md'}` first and follow it as the primary task prompt.\n"
        "Use the selected capability roles listed in that prompt as task guidance.\n"
        "Execute the objective end-to-end and finish with the final response requested in the brief.\n"
    )
    return AgentEntryPrompt(label="Claude Code SkillFabric execution", prompt=prompt)
