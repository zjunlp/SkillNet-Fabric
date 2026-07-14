from __future__ import annotations

import json
import tarfile
import tomllib
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tests.unit.wiki_helpers import build_fixture_workspace

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_ROOT = PACKAGE_ROOT.parent
FIXTURE_SKILLS = Path(__file__).resolve().parents[1] / "fixtures" / "skills"


class PublicPackageTests(unittest.TestCase):
    def test_python_facade_exposes_the_documented_workflow(self) -> None:
        from skillfabric import SkillFabric

        client = SkillFabric(workspace=".skillfabric")

        self.assertEqual(str(client.workspace.root), ".skillfabric")
        self.assertTrue(callable(client.build))
        self.assertTrue(callable(client.route))
        self.assertTrue(callable(client.plan))

    def test_public_package_declares_required_build_runtime_dependencies(self) -> None:
        pyproject = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        dependencies = {
            item.split(";", 1)[0].split(">", 1)[0].split("=", 1)[0]
            for item in pyproject["project"]["dependencies"]
        }

        self.assertEqual(
            dependencies,
            {"faiss-cpu", "litellm", "numpy", "pyyaml"},
        )

    def test_claude_code_plugin_manifest_and_commands_exist(self) -> None:
        plugin_root = PUBLIC_ROOT / "plugins" / "claude-code" / "skillfabric"
        manifest_path = plugin_root / ".claude-plugin" / "plugin.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["name"], "skillfabric")
        self.assertEqual(manifest["version"], "0.1.0")
        self.assertEqual(manifest["license"], "MIT")
        self.assertEqual(manifest["skills"], ["./"])

        command_dir = plugin_root / "commands"
        expected_commands = {"doctor", "build", "prepare", "run"}
        self.assertEqual(
            {path.stem for path in command_dir.glob("*.md")},
            expected_commands,
        )
        for command in expected_commands:
            command_path = command_dir / f"{command}.md"
            command_text = command_path.read_text(encoding="utf-8")
            self.assertIn("description:", command_text)
            self.assertIn("argument-hint:", command_text)
            for section in (
                "# Command Contract",
                "## Inputs",
                "## Required Workflow",
                "## Boundaries",
                "## Completion Criteria",
            ):
                self.assertIn(section, command_text)
            self.assertIn(
                f"Use the `skillfabric-{command}` skill as the authoritative workflow.",
                command_text,
            )
            self.assertIn(
                "Never reveal env-file contents, API keys, tokens, or shell secret values.",
                command_text,
            )

            skill_path = plugin_root / "skills" / f"skillfabric-{command}" / "SKILL.md"
            skill_text = skill_path.read_text(encoding="utf-8")
            self.assertIn(f"name: skillfabric-{command}", skill_text)
            self.assertIn("disable-model-invocation: true", skill_text)
            self.assertIn("$ARGUMENTS", skill_text)

    def test_claude_code_plugin_prompts_have_quality_guardrails(self) -> None:
        plugin_root = PUBLIC_ROOT / "plugins" / "claude-code" / "skillfabric"
        command_skill_names = ("doctor", "build", "prepare", "run")
        for name in command_skill_names:
            command_skill = plugin_root / "skills" / f"skillfabric-{name}" / "SKILL.md"
            self.assertTrue(command_skill.exists(), f"missing command skill: {command_skill}")
            text = command_skill.read_text(encoding="utf-8")
            self.assertIn(f"name: skillfabric-{name}", text)
            self.assertIn("description:", text)
            self.assertIn("disable-model-invocation: true", text)
            self.assertIn("Use when", text)
            self.assertIn("Treat CLI JSON as canonical", text)
            self.assertIn("Do not reveal secret values or env-file contents.", text)

        required_command_sections = (
            "## Purpose",
            "## Input Contract",
            "## Safety Boundaries",
            "## Workflow",
            "## Failure Handling",
            "## Final Response",
        )
        for path in sorted((plugin_root / "skills").glob("*/SKILL.md")):
            text = path.read_text(encoding="utf-8")
            for section in required_command_sections:
                self.assertIn(section, text, f"{path.name} missing {section}")

        command_files = sorted((plugin_root / "commands").glob("*.md"))
        self.assertEqual([path.stem for path in command_files], sorted(command_skill_names))
        for path in command_files:
            text = path.read_text(encoding="utf-8")
            self.assertLessEqual(len(text.split()), 260)
            for section in (
                "# Command Contract",
                "## Inputs",
                "## Required Workflow",
                "## Boundaries",
                "## Completion Criteria",
            ):
                self.assertIn(section, text)
            self.assertIn(
                f"Use the `skillfabric-{path.stem}` skill as the authoritative workflow.", text
            )

        overview_text = (plugin_root / "SKILL.md").read_text(encoding="utf-8")
        self.assertIn("## Command Choice", overview_text)
        self.assertIn("Prefer the four slash commands", overview_text)
        self.assertIn("/skillfabric:doctor", overview_text)
        self.assertIn("/skillfabric:run", overview_text)

    def test_claude_code_prepare_and_run_prompts_match_single_plan_contract(self) -> None:
        plugin_root = PUBLIC_ROOT / "plugins" / "claude-code" / "skillfabric"
        reference_path = plugin_root / "references" / "route-plan.md"
        self.assertTrue(reference_path.exists())
        reference = reference_path.read_text(encoding="utf-8")
        reference_flat = " ".join(reference.split())

        for name in ("prepare", "run"):
            text = (plugin_root / "skills" / f"skillfabric-{name}" / "SKILL.md").read_text(
                encoding="utf-8"
            )
            with self.subTest(command=name):
                self.assertIn("The task is the user's natural-language request", text)
                self.assertIn("Treat only recognized flags as workflow configuration.", text)
                self.assertIn("references/route-plan.md", text)
                if name == "run":
                    self.assertIn('skillfabric run-state "$task"', text)
                    self.assertIn("set `$execution_prompt` to `prompt_path`", text)
                    self.assertIn(
                        "Before reading `execution_prompt.md`, do not use Web Search, Fetch", text
                    )
                    self.assertIn("If no reusable prompt exists and no task was provided", text)
                    self.assertIn("shell path discovery for `run-state`", text)
                self.assertIn("execution_prompt.md", text)
                self.assertIn("$ARGUMENTS", text)

        for snippet in (
            'skillfabric plan "$task"',
            '--workspace "$workspace"',
            '--env-file "$env_file"',
            "route.json",
            "execution_prompt.md",
            "planner_validation.json",
        ):
            self.assertIn(snippet, reference)
        self.assertNotIn("raw json", reference_flat.lower())

    def test_claude_code_plugin_prompts_handle_natural_language_commands(self) -> None:
        plugin_root = PUBLIC_ROOT / "plugins" / "claude-code" / "skillfabric"
        build_text = (plugin_root / "skills" / "skillfabric-build" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        doctor_text = (plugin_root / "skills" / "skillfabric-doctor" / "SKILL.md").read_text(
            encoding="utf-8"
        )

        self.assertIn("natural language that may contain a skill root", build_text)
        self.assertIn("first existing path-like directory as the skill root", build_text)
        self.assertIn("ignore surrounding", build_text)
        self.assertIn("must contain at least one `SKILL.md`.", build_text)
        self.assertNotIn("`skill.md`", build_text)
        self.assertIn("not built yet", doctor_text)
        self.assertIn("normal before\n  the first build", doctor_text)
        self.assertIn("SkillFabric ready.", doctor_text)
        self.assertIn("Workspace: ready, <skill_count> skills, build <build_id>", doctor_text)

    def test_claude_code_plugin_readme_is_product_grade(self) -> None:
        readme = (PUBLIC_ROOT / "plugins" / "claude-code" / "skillfabric" / "README.md").read_text(
            encoding="utf-8"
        )
        for section in (
            "## Requirements",
            "## Installation",
            "## Quickstart",
            "## Commands",
            "## Local Smoke Test",
            "## Security Model",
            "## Troubleshooting",
            "## Uninstall",
        ):
            self.assertIn(section, readme)
        for snippet in (
            "which skillfabric",
            "skillfabric --help",
            "claude plugin validate --strict",
            "claude plugin list --json",
            "/skillfabric:doctor",
            "/skillfabric:prepare",
            "/skillfabric:run",
            "reuses the latest prepared prompt when available",
            "--wiki-summary-mode off",
            "Do not paste API keys",
        ):
            self.assertIn(snippet, readme)

    def test_public_docs_do_not_leak_local_paths(self) -> None:
        doc_paths = [
            PUBLIC_ROOT / "README.md",
            PACKAGE_ROOT / "README.md",
            PACKAGE_ROOT / "tests" / "README.md",
            PUBLIC_ROOT / "plugins" / "claude-code" / "skillfabric" / "README.md",
        ]
        for path in doc_paths:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("/Users/", text, path)

    def test_python_facade_rejects_coerced_route_limits(self) -> None:
        from skillfabric import SkillFabric

        client = SkillFabric(workspace=".skillfabric")
        invalid_overrides = [
            {"max_selected_skills": True},
            {"seed_limit": "3"},
            {"explorer_timeout_seconds": "30"},
        ]

        with patch("skillfabric.api.route_task") as route_mock:
            for overrides in invalid_overrides:
                with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                    client.route("extract financial KPIs", **overrides)

        route_mock.assert_not_called()

    def test_python_facade_rejects_coerced_build_options(self) -> None:
        from skillfabric import SkillFabric

        client = SkillFabric(workspace=".skillfabric")
        invalid_overrides = [
            {"skip_wiki": "false"},
            {"llm_concurrency": True},
            {"embedding_model": 123},
        ]

        with patch("skillfabric.api.build_graph") as build_mock:
            for overrides in invalid_overrides:
                with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                    client.build(FIXTURE_SKILLS, **overrides)

        build_mock.assert_not_called()

    def test_python_facade_plans_once_and_preserves_original_task(self) -> None:
        from skillfabric import SkillFabric

        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            client = SkillFabric(workspace=workspace)
            with patch(
                "skillfabric.orchestrator.package.litellm_completion",
                return_value=json.dumps(
                    {
                        "execution_prompt": (
                            "Parse the PDF, extract the KPIs, and verify each value."
                        )
                    }
                ),
            ) as planner:
                result = client.plan(
                    "extract financial KPIs from a PDF report",
                    route=_facade_route(),
                )

            planner.assert_called_once()
            self.assertEqual(result.prompt_path, result.root / "execution_prompt.md")
            self.assertTrue(result.prompt_path.exists())
            prompt = result.prompt_path.read_text(encoding="utf-8")
            self.assertIn("extract financial KPIs from a PDF report", prompt)
            self.assertIn("Parse the PDF, extract the KPIs", prompt)

    def test_python_facade_rejects_route_file_query_mismatch(self) -> None:
        from skillfabric import SkillFabric

        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            trace = workspace / "runs" / "route-trace"
            trace.mkdir(parents=True)
            route_path = trace / "route.json"
            route_path.write_text(json.dumps(_facade_route().to_dict()), encoding="utf-8")
            (trace / "query.json").write_text(
                json.dumps({"query": "original task"}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "differs from the route query"):
                SkillFabric(workspace=workspace).plan(
                    "different task",
                    route_file=route_path,
                )

    def test_python_facade_resolves_relative_plan_paths_inside_workspace_runs(self) -> None:
        from skillfabric import SkillFabric

        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            trace = workspace / "runs" / "route-trace"
            trace.mkdir(parents=True)
            (trace / "route.json").write_text(
                json.dumps(_facade_route().to_dict()),
                encoding="utf-8",
            )
            (trace / "query.json").write_text(
                json.dumps({"query": "original task"}),
                encoding="utf-8",
            )

            with patch("skillfabric.api.plan_execution_package") as planner:
                SkillFabric(workspace=workspace).plan(
                    route_file="route-trace/route.json",
                    package_root="relative-package",
                )

            self.assertEqual(
                planner.call_args.kwargs["package_root"],
                (workspace / "runs" / "relative-package").resolve(),
            )

    def test_python_facade_rejects_non_string_route_query_artifact(self) -> None:
        from skillfabric import SkillFabric

        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            trace = workspace / "runs" / "route-trace"
            trace.mkdir(parents=True)
            route_path = trace / "route.json"
            route_path.write_text(json.dumps(_facade_route().to_dict()), encoding="utf-8")
            (trace / "query.json").write_text(json.dumps({"query": 123}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "query"):
                SkillFabric(workspace=workspace).plan(route_file=route_path)

    def test_python_facade_rejects_ambiguous_route_sources(self) -> None:
        from skillfabric import SkillFabric

        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            trace = workspace / "runs" / "route-trace"
            trace.mkdir(parents=True)
            route_path = trace / "route.json"
            route_path.write_text(json.dumps(_facade_route().to_dict()), encoding="utf-8")
            (trace / "query.json").write_text(
                json.dumps({"query": "original task"}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(TypeError, "route and route_file"):
                SkillFabric(workspace=workspace).plan(
                    "original task",
                    route=_facade_route(),
                    route_file=route_path,
                )

    def test_python_facade_rejects_route_file_outside_workspace_runs(self) -> None:
        from skillfabric import SkillFabric

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / ".skillfabric"
            build_fixture_workspace(workspace)
            external_trace = root / "external-trace"
            external_trace.mkdir()
            route_path = external_trace / "route.json"
            route_path.write_text(json.dumps(_facade_route().to_dict()), encoding="utf-8")
            (external_trace / "query.json").write_text(
                json.dumps({"query": "original task"}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "route_file must stay inside"):
                SkillFabric(workspace=workspace).plan(
                    "original task",
                    route_file=route_path,
                    package_root=workspace / "runs" / "package",
                )

    def test_python_facade_rejects_package_root_outside_workspace_runs(self) -> None:
        from skillfabric import SkillFabric

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / ".skillfabric"
            build_fixture_workspace(workspace)

            with self.assertRaisesRegex(ValueError, "package_root must stay inside"):
                SkillFabric(workspace=workspace).plan(
                    "original task",
                    route=_facade_route(),
                    package_root=root / "external-package",
                )

    def test_python_facade_rejects_unknown_embedding_provider(self) -> None:
        from skillfabric import SkillFabric

        client = SkillFabric(workspace=".skillfabric")

        with self.assertRaisesRegex(ValueError, "unsupported embedding provider"):
            client.build(FIXTURE_SKILLS, embedding_provider="custom-provider")

    def test_built_distributions_exclude_private_runtime_artifacts(self) -> None:
        dist_dir = PACKAGE_ROOT / "dist"
        if not dist_dir.exists():
            self.skipTest("dist/ not built")
        forbidden = (".env", "models/", "runs/", "experiments/", "__pycache__")
        archive_paths: list[str] = []
        for archive in dist_dir.iterdir():
            if archive.suffix == ".whl":
                with zipfile.ZipFile(archive) as handle:
                    archive_paths.extend(handle.namelist())
            elif archive.suffixes[-2:] == [".tar", ".gz"]:
                with tarfile.open(archive) as handle:
                    archive_paths.extend(handle.getnames())
        for path in archive_paths:
            self.assertFalse(any(marker in path for marker in forbidden), path)


def _facade_route():
    from skillfabric.router.models import RouteRelationEvidence, RouteResult, RouteSelectedSkill

    return RouteResult(
        selected_skills=(
            RouteSelectedSkill(
                skill_id="skill:pdf-table-parser",
                name="pdf-table-parser",
                reason="Parse the PDF into normalized tables.",
                evidence=("skills/cards/pdf-table-parser.md",),
            ),
            RouteSelectedSkill(
                skill_id="skill:financial-kpi-extractor",
                name="financial-kpi-extractor",
                reason="Extract financial KPI values.",
                evidence=("skills/cards/financial-kpi-extractor.md",),
            ),
        ),
        relation_evidence=(
            RouteRelationEvidence(
                relation_type="depend_on",
                source_skill="skill:financial-kpi-extractor",
                target_skill="skill:pdf-table-parser",
                confidence=0.94,
                reason="KPI extraction requires normalized tables.",
                evidence=("edges/semantic_edges.jsonl",),
            ),
        ),
        near_misses=(),
        coverage_gaps=(),
        wiki_pages_read=(
            "skills/cards/pdf-table-parser.md",
            "skills/cards/financial-kpi-extractor.md",
            "edges/semantic_edges.jsonl",
        ),
        rationale="Parse tables before extracting KPIs.",
    )


if __name__ == "__main__":
    unittest.main()
