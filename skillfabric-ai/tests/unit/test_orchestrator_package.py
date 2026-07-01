from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from skillfabric.orchestrator.agent_run_spec import agent_run_spec_from_route
from skillfabric.orchestrator.package import (
    FORBIDDEN_EXECUTION_PROMPT_FRAGMENTS,
    PLANNER_PROMPT_ID,
    build_execution_package,
    deterministic_planner_output,
    finalize_execution_package,
    prepare_execution_package,
)
from skillfabric.orchestrator.renderers.claude_code import (
    render_claude_code_entry_prompt,
    render_execution_prompt,
)
from skillfabric.orchestrator.renderers.codex import render_codex_entry_prompt
from skillfabric.router.models import RouteEdge, RouteResult, RouteSelectedSkill
from skillfabric.storage import Workspace
from skillfabric.task_understanding import analyze_task
from skillfabric.wiki.pages import slug


def _route(workspace: Path) -> RouteResult:
    query = "Generate PNG figures and write report.docx from analyzed data."
    understanding = analyze_task(query)
    for requirement in understanding.coverage_requirements:
        if requirement.id == "deliverable:png":
            requirement.preferred_skill_ids = ["skill:data-visualization"]
            requirement.acceptable_skill_ids = ["skill:data-visualization"]
        if requirement.id == "deliverable:docx":
            requirement.preferred_skill_ids = ["skill:docx"]
            requirement.acceptable_skill_ids = ["skill:docx"]
    return RouteResult(
        query=query,
        trace_id="exec-test",
        trace_dir=workspace / "runs" / "exec-test",
        selected_skills=[
            RouteSelectedSkill(
                skill_id="skill:data-visualization",
                name="data-visualization",
                rank=1,
                reason="Create requested PNG figures.",
                evidence=["skills/core/data-visualization.md"],
            ),
            RouteSelectedSkill(
                skill_id="skill:docx",
                name="docx",
                rank=2,
                reason="Write the requested Word report.",
                evidence=["skills/core/docx.md"],
            ),
        ],
        required_edges=[
            RouteEdge(
                before_skill="skill:data-visualization",
                after_skill="skill:docx",
                edge_type="depend_on",
                reason="The report should include generated figures.",
                source="execution_index",
            )
        ],
        task_understanding=understanding,
        provenance="test",
    )


class OrchestratorPackageTests(unittest.TestCase):
    def test_agent_run_spec_from_route_contains_phases_and_acceptance_criteria(self) -> None:
        with TemporaryDirectory() as tmp:
            route = _route(Path(tmp) / ".skillfabric")

            spec = agent_run_spec_from_route(route)

            payload = spec.to_dict()
            self.assertEqual(payload["objective"], route.query)
            self.assertEqual(
                [item["skill_id"] for item in payload["selected_skills"]],
                ["skill:data-visualization", "skill:docx"],
            )
            self.assertEqual(payload["phases"][0]["skill_ids"], ["skill:data-visualization"])
            self.assertEqual(payload["phases"][1]["depends_on"], ["phase_1"])
            self.assertEqual(payload["required_order"][0]["before_skill"], "skill:data-visualization")
            self.assertIn("deliverable:png", {item["id"] for item in payload["acceptance_criteria"]})
            self.assertNotIn("completion_report", payload)
            operation_names = [item["operation"] for item in payload["execution_strategy"]["operations"]]
            self.assertEqual(operation_names[:2], ["orient", "inspect"])
            self.assertIn("apply_skill", operation_names)
            self.assertIn("verify", operation_names)
            self.assertEqual(operation_names[-1], "report")
            docx_operation = next(
                item
                for item in payload["execution_strategy"]["operations"]
                if item["operation"] == "apply_skill" and item["skill_ids"] == ["skill:docx"]
            )
            self.assertEqual(docx_operation["control"], "serial")
            self.assertIn("op_apply_skill_1", docx_operation["depends_on"])
            self.assertTrue(
                any(
                    item["operation"] == "aggregate"
                    for item in payload["execution_strategy"]["operations"]
                )
            )
            self.assertIn("delegate", payload["execution_strategy"]["delegation_policy"])

    def test_agent_run_spec_phases_are_topologically_ordered(self) -> None:
        with TemporaryDirectory() as tmp:
            route = _route(Path(tmp) / ".skillfabric")
            route.selected_skills = list(reversed(route.selected_skills))

            spec = agent_run_spec_from_route(route)

            payload = spec.to_dict()
            phase_skills = [phase["skill_ids"][0] for phase in payload["phases"]]
            self.assertEqual(phase_skills, ["skill:data-visualization", "skill:docx"])
            self.assertEqual(payload["phases"][1]["depends_on"], ["phase_1"])

    def test_agent_run_spec_adds_parallel_guidance_for_independent_skills(self) -> None:
        with TemporaryDirectory() as tmp:
            route = _route(Path(tmp) / ".skillfabric")
            route.required_edges = []

            spec = agent_run_spec_from_route(route)

            operations = spec.to_dict()["execution_strategy"]["operations"]
            parallelize = next(item for item in operations if item["operation"] == "parallelize")
            self.assertEqual(parallelize["control"], "parallel")
            self.assertCountEqual(parallelize["skill_ids"], ["skill:data-visualization", "skill:docx"])

    def test_renderer_prompts_use_target_agent_dialects(self) -> None:
        with TemporaryDirectory() as tmp:
            route = _route(Path(tmp) / ".skillfabric")
            spec = agent_run_spec_from_route(route)

            codex_prompt = render_execution_prompt(spec, renderer="codex")
            claude_prompt = render_execution_prompt(spec, renderer="claude-code")

            self.assertIn("update_plan", codex_prompt)
            self.assertIn("apply_patch", codex_prompt)
            self.assertIn("spawn_agent", codex_prompt)
            self.assertNotIn("TodoWrite", codex_prompt)
            self.assertNotIn("Task tool", codex_prompt)
            self.assertIn("TodoWrite", claude_prompt)
            self.assertIn("Read / Grep / Glob / LS", claude_prompt)
            self.assertIn("Task", claude_prompt)
            self.assertNotIn("apply_patch", claude_prompt)
            self.assertNotIn("spawn_agent", claude_prompt)
            self.assertIn("Skill Use Protocol", claude_prompt)
            self.assertIn("Verification Protocol", claude_prompt)
            self.assertIn("coverage gap", claude_prompt)
            self.assertIn("selected capability roles", claude_prompt)
            self.assertNotIn("Skill tool", claude_prompt)
            self.assertNotIn("Audit context", claude_prompt)
            self.assertNotIn("current Claude Code skill surface", claude_prompt)
            self.assertNotIn("not as a replacement for native skill instructions", claude_prompt)
            for heading in (
                "## TODO",
                "## Input",
                "## Output",
                "## Workflow",
                "## Rules",
                "## Constraints",
                "## Action Type Definitions",
            ):
                self.assertIn(heading, claude_prompt)

    def test_execution_package_contains_only_selected_skills_and_agent_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_path = Path(tmp) / ".skillfabric"
            workspace = Workspace(workspace_path)
            workspace.ensure()
            route = _route(workspace_path)
            for skill_id in ["skill:data-visualization", "skill:docx", "skill:outside"]:
                path = workspace.wiki_skills_dir / f"{slug(skill_id)}.md"
                path.write_text(f"# {skill_id}\n", encoding="utf-8")

            result = build_execution_package(workspace, route)

            root = result.root
            self.assertTrue((root / "execution_prompt.md").exists())
            self.assertTrue((root / "agent_run_spec.json").exists())
            self.assertFalse((root / "completion_report_schema.json").exists())
            self.assertTrue((root / "evidence" / "route_summary.json").exists())
            self.assertTrue((root / "evidence" / "selected_skill_evidence.json").exists())
            self.assertTrue((root / "evidence" / "required_edges.json").exists())
            copied = sorted(path.name for path in (root / "selected_skills").glob("*.md"))
            self.assertEqual(copied, ["data-visualization.md", "docx.md"])
            self.assertFalse((root / "selected_skills" / "outside.md").exists())
            self.assertEqual(result.prompt_path, root / "execution_prompt.md")
            self.assertEqual(result.renderer, "claude-code")
            spec_payload = json.loads((root / "agent_run_spec.json").read_text(encoding="utf-8"))
            self.assertEqual(spec_payload["selected_skills"][0]["skill_context_path"], "selected_skills/data-visualization.md")
            self.assertIn("execution_strategy", spec_payload)
            prompt = (root / "execution_prompt.md").read_text(encoding="utf-8")
            self.assertIn("SkillFabric Execution Prompt", prompt)
            self.assertIn("Selected Skills", prompt)
            self.assertIn("Final Report", prompt)
            for fragment in FORBIDDEN_EXECUTION_PROMPT_FRAGMENTS:
                self.assertNotIn(fragment.lower(), prompt.lower())

    def test_prepare_package_waits_for_planner_before_writing_final_prompt(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_path = Path(tmp) / ".skillfabric"
            workspace = Workspace(workspace_path)
            workspace.ensure()
            route = _route(workspace_path)
            for skill_id in ["skill:data-visualization", "skill:docx"]:
                path = workspace.wiki_skills_dir / f"{slug(skill_id)}.md"
                path.write_text(f"# {skill_id}\n", encoding="utf-8")

            prepared = prepare_execution_package(workspace, route)

            self.assertEqual(prepared.root, route.trace_dir / "execution_package")
            self.assertTrue((prepared.root / "route.json").exists())
            self.assertTrue((prepared.root / "agent_run_spec_draft.json").exists())
            self.assertTrue((prepared.root / "planner_request.json").exists())
            self.assertTrue((prepared.root / "PLANNER.md").exists())
            self.assertFalse((prepared.root / "execution_prompt.md").exists())
            self.assertFalse((prepared.root / "agent_run_spec.json").exists())
            planner_request = json.loads((prepared.root / "planner_request.json").read_text(encoding="utf-8"))
            self.assertEqual(planner_request["prompt_id"], PLANNER_PROMPT_ID)
            planner_prompt = (prepared.root / "PLANNER.md").read_text(encoding="utf-8")
            for heading in (
                "# Prompt Contract",
                "# Role",
                "# Success Criteria",
                "# Workflow",
                "# Output Contract",
                "# Final Execution Prompt Policy",
                "# Self-Check",
            ):
                self.assertIn(heading, planner_prompt)
            self.assertIn("skill:data-visualization -> skill:docx", planner_prompt)
            self.assertNotIn("benchmark", planner_prompt.lower())
            self.assertNotIn("audit", planner_prompt.lower())
            self.assertNotIn("completion_report", planner_prompt)

    def test_finalize_package_uses_planner_workflow_as_final_prompt_contract(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_path = Path(tmp) / ".skillfabric"
            workspace = Workspace(workspace_path)
            workspace.ensure()
            route = _route(workspace_path)
            for skill_id in ["skill:data-visualization", "skill:docx"]:
                path = workspace.wiki_skills_dir / f"{slug(skill_id)}.md"
                path.write_text(f"# {skill_id}\n", encoding="utf-8")
            prepared = prepare_execution_package(workspace, route)
            planner_output = deterministic_planner_output(route, prepared.draft_spec)
            planner_output["workflow_plan"]["phases"] = [
                {
                    "id": "phase_story",
                    "goal": "Create the PNG charts first.",
                    "skill_ids": ["skill:data-visualization"],
                    "depends_on": [],
                    "expected_outputs": ["PNG charts"],
                    "evidence_refs": ["evidence/selected_skill_evidence.json"],
                    "guidance": "Render figures before writing the report.",
                },
                {
                    "id": "phase_report",
                    "goal": "Write the Word report using generated charts.",
                    "skill_ids": ["skill:docx"],
                    "depends_on": ["phase_story"],
                    "expected_outputs": ["report.docx"],
                    "evidence_refs": ["evidence/required_edges.json"],
                    "guidance": "Integrate figures into the report.",
                },
            ]
            planner_output["execution_prompt"] = "# Planner Authored Prompt\n\nUse the planned workflow."

            result = finalize_execution_package(prepared.root, planner_output)

            self.assertTrue(result.prompt_path.exists())
            self.assertEqual(result.prompt_path.read_text(encoding="utf-8"), "# Planner Authored Prompt\n\nUse the planned workflow.\n")
            self.assertTrue((prepared.root / "workflow_plan.json").exists())
            self.assertTrue((prepared.root / "planner_output.json").exists())
            self.assertTrue((prepared.root / "planner_validation.json").exists())
            self.assertFalse((prepared.root / "handoff_prompt.md").exists())
            spec_payload = json.loads((prepared.root / "agent_run_spec.json").read_text(encoding="utf-8"))
            self.assertEqual([phase["id"] for phase in spec_payload["phases"]], ["phase_story", "phase_report"])
            self.assertEqual(spec_payload["phases"][1]["depends_on"], ["phase_story"])
            self.assertEqual(result.planner_validation_path, prepared.root / "planner_validation.json")

    def test_finalize_package_rejects_unselected_skills_and_order_violations(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_path = Path(tmp) / ".skillfabric"
            workspace = Workspace(workspace_path)
            workspace.ensure()
            route = _route(workspace_path)
            prepared = prepare_execution_package(workspace, route)
            planner_output = deterministic_planner_output(route, prepared.draft_spec)
            planner_output["workflow_plan"]["phases"] = [
                {
                    "id": "phase_report",
                    "goal": "Write report too early.",
                    "skill_ids": ["skill:docx"],
                    "depends_on": [],
                },
                {
                    "id": "phase_unknown",
                    "goal": "Use a skill that was not selected.",
                    "skill_ids": ["skill:outside"],
                    "depends_on": ["phase_report"],
                },
                {
                    "id": "phase_charts",
                    "goal": "Create charts after report.",
                    "skill_ids": ["skill:data-visualization"],
                    "depends_on": ["phase_report"],
                },
            ]

            with self.assertRaises(ValueError) as raised:
                finalize_execution_package(prepared.root, planner_output)

            message = str(raised.exception)
            self.assertIn("unselected skill", message)
            self.assertIn("violates required order", message)

    def test_finalize_package_rejects_runtime_mechanics_in_execution_prompt(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_path = Path(tmp) / ".skillfabric"
            workspace = Workspace(workspace_path)
            workspace.ensure()
            route = _route(workspace_path)
            prepared = prepare_execution_package(workspace, route)
            planner_output = deterministic_planner_output(route, prepared.draft_spec)
            planner_output["execution_prompt"] = (
                "# Bad Prompt\n\n"
                "Use the Skill tool and read selected_skills/data-visualization.md."
            )

            with self.assertRaises(ValueError) as raised:
                finalize_execution_package(prepared.root, planner_output)

            message = str(raised.exception)
            self.assertIn("forbidden runtime-mechanism wording", message)
            self.assertTrue((prepared.root / "planner_validation.json").exists())

    def test_renderers_return_entry_prompts_for_same_spec(self) -> None:
        with TemporaryDirectory() as tmp:
            route = _route(Path(tmp) / ".skillfabric")
            spec = agent_run_spec_from_route(route)

            claude = render_claude_code_entry_prompt(spec, execution_package_root=route.trace_dir / "execution_package")
            codex = render_codex_entry_prompt(spec, execution_package_root=route.trace_dir / "execution_package")

            self.assertIn("execution_prompt.md", claude.prompt)
            self.assertNotIn("agent_run_spec.json", claude.prompt)
            self.assertIn("Claude Code", claude.label)
            self.assertIn("execution_prompt.md", codex.prompt)
            self.assertNotIn("agent_run_spec.json", codex.prompt)
            self.assertIn("Codex", codex.label)
            self.assertNotEqual(claude.prompt, codex.prompt)


if __name__ == "__main__":
    unittest.main()
