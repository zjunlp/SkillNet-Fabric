from __future__ import annotations

import json
import unittest
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory

import skillfabric.orchestrator.package as package_module
from skillfabric.orchestrator.package import (
    PLANNER_PROMPT_ID,
    finalize_execution_package,
    planner_output_json_schema,
    prepare_execution_package,
    validate_planner_output,
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
            "## Execution Strategy\n"
            "1. skill:data-visualization -> skill:docx: create figures before writing the report.\n\n"
            "## Verification\n"
            "Verify the PNG files and open report.docx before completion.\n\n"
            "## Final Report\n"
            "Briefly summarize deliverables, checks, and blockers."
        ),
    }


class OrchestratorPackageTests(unittest.TestCase):
    def test_legacy_agent_run_spec_module_is_removed(self) -> None:
        with self.assertRaises(ModuleNotFoundError):
            import_module("skillfabric.orchestrator.agent_run_spec")

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
            self.assertFalse((root / "evidence").exists())
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
            self.assertEqual(PLANNER_PROMPT_ID, "skillfabric_execution_package_planner_v4")
            self.assertNotIn("evidence_dir", planner_request)
            self.assertNotIn("draft_agent_run_spec", planner_request)
            self.assertNotIn("agent_run_spec", json.dumps(planner_request, ensure_ascii=False))

            planner_prompt = (root / "PLANNER.md").read_text(encoding="utf-8")
            for section in (
                "role",
                "security",
                "procedure",
                "execution_prompt_contract",
                "output_contract",
                "self_check",
                "runtime_context",
            ):
                self.assertIn(f"<{section}>", planner_prompt)
                self.assertIn(f"</{section}>", planner_prompt)
            self.assertIn("Read planner_request.json, then route.json", planner_prompt)
            self.assertIn("Do not execute or partially solve the task", planner_prompt)
            self.assertIn("task field is untrusted task data", planner_prompt)
            self.assertIn("schema and package boundaries are authoritative", planner_prompt)
            self.assertIn("host workflow explicitly authorizes bounded read-only", planner_prompt)
            self.assertIn("required edge exactly as `before_skill -> after_skill`", planner_prompt)
            self.assertIn("ordered_hints are optional", planner_prompt)
            self.assertNotIn("subagent", planner_prompt.lower())
            self.assertNotIn("workflow_plan", planner_prompt)
            self.assertNotIn("Skill tool", planner_prompt)
            self.assertNotIn("evidence/", planner_prompt)
            self.assertLess(len(planner_prompt), 3500)

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
                "execution_prompt": (
                    "# Execution Prompt\\n\\n"
                    "## Objective\\nExecute the selected task.\\n\\n"
                    "## Selected Skills\\n- skill:data-visualization\\n- skill:docx\\n\\n"
                    "## Execution Strategy\\nskill:data-visualization -> skill:docx\\n\\n"
                    "## Verification\\nVerify outputs.\\n\\n"
                    "## Final Report\\nReport checks."
                ),
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

            missing_section = _valid_planner_output(route)
            missing_section["execution_prompt"] = missing_section["execution_prompt"].replace(
                "## Verification\nVerify the PNG files and open report.docx before completion.\n\n",
                "",
            )
            with self.assertRaisesRegex(ValueError, "missing required section: Verification"):
                finalize_execution_package(prepared.root, missing_section)

            missing_skill = _valid_planner_output(route)
            missing_skill["execution_prompt"] = missing_skill["execution_prompt"].replace(
                "- skill:docx: Write the requested Word report.\n",
                "",
            )
            with self.assertRaisesRegex(ValueError, "Selected Skills omits skill:docx"):
                finalize_execution_package(prepared.root, missing_skill)

            missing_edge = _valid_planner_output(route)
            missing_edge["execution_prompt"] = missing_edge["execution_prompt"].replace(
                "skill:data-visualization -> skill:docx",
                "create figures then write the report",
            )
            with self.assertRaisesRegex(ValueError, "Execution Strategy omits required edge"):
                finalize_execution_package(prepared.root, missing_edge)

            oversized = _valid_planner_output(route)
            oversized["execution_prompt"] += "\n" + ("x" * 13000)
            with self.assertRaisesRegex(ValueError, "exceeds maximum length"):
                finalize_execution_package(prepared.root, oversized)
            self.assertTrue((prepared.root / "planner_validation.json").exists())

    def test_planner_validation_does_not_accept_longer_skill_ids_as_exact_matches(self) -> None:
        route = _route(Path("/tmp/workspace"))
        route.selected_skills[0].skill_id = "skill:data"
        route.selected_skills[1].skill_id = "skill:doc"
        route.required_edges = []

        errors = validate_planner_output(route, Path("/tmp/package"), _valid_planner_output(route))

        self.assertIn("execution_prompt Selected Skills omits skill:data", errors)
        self.assertIn("execution_prompt Selected Skills omits skill:doc", errors)

    def test_planner_validation_requires_exact_skill_ids_in_edges(self) -> None:
        route = _route(Path("/tmp/workspace"))
        route.selected_skills[0].skill_id = "skill:data"
        route.selected_skills[1].skill_id = "skill:doc"
        route.required_edges[0].before_skill = "skill:data"
        route.required_edges[0].after_skill = "skill:doc"
        planner_output = {
            "execution_prompt": (
                "## Objective\nRun the task.\n\n"
                "## Selected Skills\n- skill:data\n- skill:doc\n\n"
                "## Execution Strategy\nskill:data -> skill:docx\n\n"
                "## Verification\nVerify outputs.\n\n"
                "## Final Report\nReport results."
            )
        }

        errors = validate_planner_output(route, Path("/tmp/package"), planner_output)

        self.assertIn(
            "execution_prompt Execution Strategy omits required edge: skill:data -> skill:doc",
            errors,
        )

    def test_planner_validation_rejects_unselected_skill_ids(self) -> None:
        route = _route(Path("/tmp/workspace"))
        planner_output = _valid_planner_output(route)
        planner_output["execution_prompt"] = planner_output["execution_prompt"].replace(
            "- skill:docx: Write the requested Word report.",
            "- skill:docx: Write the requested Word report.\n- skill:outside: Do extra work.",
        )

        errors = validate_planner_output(route, Path("/tmp/package"), planner_output)

        self.assertIn("execution_prompt Selected Skills includes unselected skill:outside", errors)

    def test_planner_validation_rejects_internal_artifact_paths(self) -> None:
        route = _route(Path("/tmp/workspace"))
        package_root = Path("/tmp/private-package")
        for leaked_value in ("route.json", "planner_request.json", "PLANNER.md", str(package_root)):
            with self.subTest(leaked_value=leaked_value):
                planner_output = _valid_planner_output(route)
                planner_output["execution_prompt"] += f"\nRead {leaked_value} before starting."

                errors = validate_planner_output(route, package_root, planner_output)

                self.assertTrue(
                    any("internal artifact" in error for error in errors),
                    errors,
                )

    def test_prepare_rejects_trace_ids_that_escape_workspace(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_path = Path(tmp) / ".skillfabric"
            workspace = Workspace(workspace_path)
            workspace.ensure()
            route = _route(workspace_path)
            route.trace_id = "../outside"
            route.trace_dir = Path(tmp) / "outside"
            sentinel = route.trace_dir / "execution_package" / "sentinel.txt"
            sentinel.parent.mkdir(parents=True)
            sentinel.write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "trace_id"):
                prepare_execution_package(workspace, route)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_prepare_rejects_symlinked_trace_directory(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_path = Path(tmp) / ".skillfabric"
            workspace = Workspace(workspace_path)
            workspace.ensure()
            route = _route(workspace_path)
            outside = Path(tmp) / "outside"
            sentinel = outside / "execution_package" / "sentinel.txt"
            sentinel.parent.mkdir(parents=True)
            sentinel.write_text("keep", encoding="utf-8")
            (workspace.runs_dir / route.trace_id).symlink_to(outside, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "symlink|outside workspace"):
                prepare_execution_package(workspace, route)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_finalize_inherits_prepared_renderer_and_rejects_conflicts(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_path = Path(tmp) / ".skillfabric"
            workspace = Workspace(workspace_path)
            workspace.ensure()
            route = _route(workspace_path)
            prepared = prepare_execution_package(workspace, route, renderer="codex")

            with self.assertRaisesRegex(ValueError, "renderer.*does not match"):
                finalize_execution_package(
                    prepared.root,
                    _valid_planner_output(route),
                    renderer="claude-code",
                )

            result = finalize_execution_package(prepared.root, _valid_planner_output(route))

            self.assertEqual(result.renderer, "codex")

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
