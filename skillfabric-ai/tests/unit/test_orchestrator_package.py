from __future__ import annotations

import json
import unittest
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory

import skillfabric.orchestrator.package as package_module
from skillfabric.orchestrator.agent_run_spec import agent_run_spec_from_route
from skillfabric.orchestrator.package import (
    PLANNER_PROMPT_ID,
    finalize_execution_package,
    planner_output_json_schema,
    prepare_execution_package,
)
from skillfabric.router.models import (
    RouteEdge,
    RouteResult,
    RouteSelectedSkill,
)
from skillfabric.storage import Workspace
from skillfabric.wiki.pages import slug


def _route(workspace: Path) -> RouteResult:
    query = "Generate PNG figures and write report.docx from analyzed data."
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
                evidence=["skills/data-visualization.md"],
            ),
            RouteSelectedSkill(
                skill_id="skill:docx",
                name="docx",
                rank=2,
                reason="Write the requested Word report.",
                evidence=["skills/docx.md"],
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
        provenance="test",
    )


def _valid_planner_output(route: RouteResult) -> dict[str, str]:
    return {
        "execution_prompt": (
            "# Execution Prompt\n\n"
            "## Objective\n"
            f"{route.query}\n\n"
            "## Selected Skills\n"
            "- skill:data-visualization: Create requested PNG figures.\n"
            "- skill:docx: Write the requested Word report.\n\n"
            "## Final Report\n"
            "Briefly summarize deliverables, checks, and blockers."
        ),
    }


class OrchestratorPackageTests(unittest.TestCase):
    def test_agent_run_spec_from_route_contains_phases_and_execution_strategy(self) -> None:
        with TemporaryDirectory() as tmp:
            route = _route(Path(tmp) / ".skillfabric")

            spec = agent_run_spec_from_route(route)

            payload = spec.to_dict()
            self.assertEqual(payload["objective"], route.query)
            self.assertEqual(
                [item["skill_id"] for item in payload["selected_skills"]],
                ["skill:data-visualization", "skill:docx"],
            )
            self.assertEqual(
                [item["native_skill_name"] for item in payload["selected_skills"]],
                ["data-visualization", "docx"],
            )
            self.assertEqual(payload["phases"][0]["skill_ids"], ["skill:data-visualization"])
            self.assertEqual(payload["phases"][1]["depends_on"], ["phase_1"])
            self.assertEqual(payload["required_order"][0]["before_skill"], "skill:data-visualization")
            self.assertEqual(payload["acceptance_criteria"], [])
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
            self.assertIn("main Claude Code session", payload["execution_strategy"]["delegation_policy"])
            self.assertNotIn("subagent", payload["execution_strategy"]["delegation_policy"].lower())

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

    def test_prepare_execution_package_contains_selected_context_and_planner_artifacts_only(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_path = Path(tmp) / ".skillfabric"
            workspace = Workspace(workspace_path)
            workspace.ensure()
            route = _route(workspace_path)
            for skill_id in ["skill:data-visualization", "skill:docx", "skill:outside"]:
                path = workspace.wiki_skill_cards_dir / f"{slug(skill_id)}.md"
                path.write_text(f"# {skill_id}\n", encoding="utf-8")

            result = prepare_execution_package(workspace, route)

            root = result.root
            self.assertFalse((root / "execution_prompt.md").exists())
            self.assertFalse((root / "agent_run_spec.json").exists())
            self.assertFalse((root / "agent_run_spec_draft.json").exists())
            self.assertFalse((root / "workflow_plan.json").exists())
            self.assertTrue((root / "planner_request.json").exists())
            self.assertTrue((root / "PLANNER.md").exists())
            self.assertTrue((root / "evidence" / "route_summary.json").exists())
            self.assertTrue((root / "evidence" / "selected_skill_evidence.json").exists())
            self.assertTrue((root / "evidence" / "required_edges.json").exists())
            copied = sorted(path.name for path in (root / "selected_skills").glob("*.md"))
            self.assertEqual(copied, ["data-visualization.md", "docx.md"])
            self.assertFalse((root / "selected_skills" / "outside.md").exists())
            self.assertEqual(result.renderer, "claude-code")
            self.assertFalse(hasattr(result, "draft_spec"))

            planner_request = json.loads((root / "planner_request.json").read_text(encoding="utf-8"))
            self.assertEqual(planner_request["expected_output"], str(root / "planner_output.json"))
            self.assertEqual(planner_request["expected_schema"], planner_output_json_schema())
            self.assertEqual(planner_request["expected_schema"]["required"], ["execution_prompt"])
            self.assertNotIn("workflow_plan", planner_request["expected_schema"]["properties"])
            self.assertEqual(planner_request["final_artifacts"], {"execution_prompt": str(root / "execution_prompt.md")})
            self.assertEqual(planner_request["prompt_id"], PLANNER_PROMPT_ID)
            self.assertNotIn("draft_agent_run_spec", planner_request)
            self.assertNotIn("agent_run_spec", json.dumps(planner_request, ensure_ascii=False))

            planner_prompt = (root / "PLANNER.md").read_text(encoding="utf-8")
            for section in (
                "# Prompt Contract",
                "# Role",
                "# Authority",
                "# Inputs",
                "# Success Criteria",
                "# Reading Order",
                "# Planning Policy",
                "# Claude Code Execution Capabilities",
                "# Output Contract",
                "# Final Prompt Requirements",
                "# Self-Check",
            ):
                self.assertIn(section, planner_prompt)
            self.assertIn("Return one strict JSON object with exactly one top-level key", planner_prompt)
            self.assertIn("Read `planner_request.json` first", planner_prompt)
            self.assertIn("Do not execute the task", planner_prompt)
            self.assertIn("required_edges are hard ordering constraints", planner_prompt)
            self.assertIn("ordered_hints are soft ordering guidance", planner_prompt)
            self.assertIn("main Claude Code session should execute the task directly", planner_prompt)
            self.assertNotIn("subagent", planner_prompt.lower())
            self.assertNotIn("workflow_plan", planner_prompt)
            self.assertNotIn("Skill tool", planner_prompt)

    def test_finalize_execution_package_writes_prompt_only_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_path = Path(tmp) / ".skillfabric"
            workspace = Workspace(workspace_path)
            workspace.ensure()
            route = _route(workspace_path)
            for skill_id in ["skill:data-visualization", "skill:docx"]:
                path = workspace.wiki_skill_cards_dir / f"{slug(skill_id)}.md"
                path.write_text(f"# {skill_id}\n", encoding="utf-8")
            prepared = prepare_execution_package(workspace, route)
            planner_output = _valid_planner_output(route)

            result = finalize_execution_package(prepared.root, planner_output)

            root = result.root
            self.assertTrue((root / "planner_output.json").exists())
            self.assertTrue((root / "planner_validation.json").exists())
            self.assertFalse((root / "workflow_plan.json").exists())
            self.assertFalse((root / "agent_run_spec.json").exists())
            self.assertTrue((root / "execution_prompt.md").exists())
            self.assertEqual(result.prompt_path, root / "execution_prompt.md")
            self.assertFalse(hasattr(result, "workflow_plan_path"))
            self.assertFalse(hasattr(result, "spec"))
            prompt = (root / "execution_prompt.md").read_text(encoding="utf-8")
            self.assertIn("Execution Prompt", prompt)
            self.assertIn("Selected Skills", prompt)
            self.assertIn("Final Report", prompt)
            self.assertNotIn("Skill tool", prompt)
            self.assertNotIn("SkillFabric", prompt)
            self.assertNotIn("selected_skills/", prompt)
            validation = json.loads((root / "planner_validation.json").read_text(encoding="utf-8"))
            self.assertTrue(validation["valid"])

    def test_finalize_execution_package_normalizes_escaped_newlines(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_path = Path(tmp) / ".skillfabric"
            workspace = Workspace(workspace_path)
            workspace.ensure()
            route = _route(workspace_path)
            prepared = prepare_execution_package(workspace, route)
            planner_output = {
                "execution_prompt": "# Execution Prompt\\n\\n## Objective\\nExecute the selected task.\\n\\n## Final Report\\nReport checks.",
            }

            finalize_execution_package(prepared.root, planner_output)

            prompt = (prepared.root / "execution_prompt.md").read_text(encoding="utf-8")
            saved_output = json.loads((prepared.root / "planner_output.json").read_text(encoding="utf-8"))
            self.assertIn("# Execution Prompt\n\n## Objective", prompt)
            self.assertNotIn("\\n\\n", prompt)
            self.assertEqual(saved_output["execution_prompt"], prompt.rstrip())

    def test_finalize_execution_package_rejects_invalid_planner_outputs(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_path = Path(tmp) / ".skillfabric"
            workspace = Workspace(workspace_path)
            workspace.ensure()
            route = _route(workspace_path)
            prepared = prepare_execution_package(workspace, route)

            with self.assertRaisesRegex(ValueError, "execution_prompt must be a non-empty string"):
                finalize_execution_package(prepared.root, {})

            extra_key = {
                "workflow_plan": {"objective": route.query},
                "execution_prompt": "Do the task.",
            }
            with self.assertRaisesRegex(ValueError, "planner output keys must be exactly"):
                finalize_execution_package(prepared.root, extra_key)

            polluted_prompt = _valid_planner_output(route)
            polluted_prompt["execution_prompt"] = "Use the Skill tool and selected_skills/docx.md."
            with self.assertRaisesRegex(ValueError, "forbidden runtime-mechanism wording"):
                finalize_execution_package(prepared.root, polluted_prompt)
            self.assertTrue((prepared.root / "planner_validation.json").exists())

    def test_deterministic_planner_fallback_is_not_available(self) -> None:
        self.assertFalse(hasattr(package_module, "deterministic_planner_output"))
        self.assertFalse(hasattr(package_module, "build_execution_package"))

    def test_direct_execution_prompt_renderer_modules_are_removed(self) -> None:
        with self.assertRaises(ModuleNotFoundError):
            import_module("skillfabric.orchestrator.renderers.claude_code")
        with self.assertRaises(ModuleNotFoundError):
            import_module("skillfabric.orchestrator.renderers.codex")


if __name__ == "__main__":
    unittest.main()
