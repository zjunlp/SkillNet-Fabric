from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility.
    import tomli as tomllib

from tests.support import build_fixture_workspace

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = PACKAGE_ROOT.parent


class PublicPackageTests(unittest.TestCase):
    def test_python_facade_exposes_the_documented_workflow(self) -> None:
        from skillfabric import SkillFabric

        client = SkillFabric(workspace=".skillfabric")

        self.assertEqual(str(client.workspace.root), ".skillfabric")
        self.assertTrue(callable(client.build))
        self.assertTrue(callable(client.route))
        self.assertTrue(callable(client.plan))

    def test_python_facade_forwards_named_explorer_backend(self) -> None:
        from skillfabric import SkillFabric

        with patch("skillfabric.api.route_task", return_value=_facade_route()) as route:
            SkillFabric(workspace=".skillfabric").route("extract KPIs", backend="codex")

        config = route.call_args.args[0]
        self.assertEqual(config.explorer_backend, "codex")

    def test_python_facade_forwards_optional_exact_selection_count(self) -> None:
        from skillfabric import SkillFabric

        with patch("skillfabric.api.route_task", return_value=_facade_route()) as route:
            SkillFabric(workspace=".skillfabric").route(
                "extract KPIs",
                max_selected_skills=5,
            )

        config = route.call_args.args[0]
        self.assertEqual(config.max_selected_skills, 5)

    def test_public_package_declares_required_build_runtime_dependencies(self) -> None:
        pyproject = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        dependencies = {
            item.split(";", 1)[0].split(">", 1)[0].split("=", 1)[0]
            for item in pyproject["project"]["dependencies"]
        }

        self.assertEqual(
            dependencies,
            {"httpx", "litellm", "numpy", "pyyaml", "rich"},
        )

    def test_public_package_metadata_is_ready_for_distribution(self) -> None:
        pyproject = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        project = pyproject["project"]
        targets = pyproject["tool"]["hatch"]["build"]["targets"]

        self.assertEqual(project["readme"], "README.md")
        self.assertEqual(project["license"], {"file": "LICENSE"})
        self.assertEqual(project["requires-python"], ">=3.10")
        self.assertIn("Development Status :: 5 - Production/Stable", project["classifiers"])
        self.assertIn("/README.md", targets["sdist"]["include"])
        self.assertIn("/LICENSE", targets["sdist"]["include"])

    def test_all_extra_contains_only_optional_runtime_backends(self) -> None:
        pyproject = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        extras = pyproject["project"]["optional-dependencies"]

        self.assertEqual(extras["all"], [*extras["claude"], *extras["codex"]])
        self.assertTrue(
            {
                "build>=1.2,<2",
                "pytest>=8.0,<10",
                "ruff>=0.8,<1",
                "tomli>=2.0; python_version < '3.11'",
            }.issubset(extras["dev"])
        )

    def test_public_brand_and_package_identifiers_are_consistent(self) -> None:
        readme = (PUBLIC_ROOT / "README.md").read_text(encoding="utf-8")
        pyproject = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

        self.assertIn("# SkillFabric", readme)
        self.assertEqual(pyproject["project"]["name"], "skillfabric-ai")
        self.assertEqual(set(pyproject["project"]["scripts"]), {"skillfabric"})
        self.assertEqual(
            pyproject["project"]["urls"],
            {
                "Homepage": "https://github.com/zjunlp/SkillNet-Fabric",
                "Source": "https://github.com/zjunlp/SkillNet-Fabric",
                "Issues": "https://github.com/zjunlp/SkillNet-Fabric/issues",
            },
        )

        self.assertNotIn("# SkillNet-Fabric", readme)
        self.assertNotIn("# SkillNet Fabric", readme)
        self.assertNotIn("SkillNet Fabric", readme)

    def test_root_readme_uses_skillnet_public_research_links(self) -> None:
        readme = (PUBLIC_ROOT / "README.md").read_text(encoding="utf-8")

        for link in (
            "https://arxiv.org/abs/2603.04448",
            "https://huggingface.co/blog/xzwnlp/skillnet",
            "http://skillnet.openkg.cn/",
        ):
            self.assertIn(link, readme)
        self.assertNotIn("badge.fury.io/py", readme)
        self.assertNotIn("img.shields.io/badge/Python", readme)
        self.assertNotIn("SkillNet and SkillFabric", readme)
        self.assertNotIn("## Research", readme)

    def test_claude_code_plugin_manifest_and_commands_exist(self) -> None:
        plugin_root = PUBLIC_ROOT / "plugins" / "claude-code" / "skillfabric"
        manifest_path = plugin_root / ".claude-plugin" / "plugin.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["name"], "skillfabric")
        self.assertEqual(manifest["version"], "0.1.0")
        self.assertEqual(manifest["license"], "MIT")
        self.assertNotIn("skills", manifest)

        command_dir = plugin_root / "commands"
        expected_commands = {"doctor", "build", "route"}
        self.assertEqual(
            {path.stem for path in command_dir.glob("*.md")},
            expected_commands,
        )
        for command in expected_commands:
            command_path = command_dir / f"{command}.md"
            command_text = command_path.read_text(encoding="utf-8")
            self.assertIn("description:", command_text)
            expected_tool = {
                "doctor": "allowed-tools: Bash(skillfabric doctor-state *)",
                "build": "allowed-tools: Bash(skillfabric build *)",
                "route": "allowed-tools: Bash(skillfabric route *)",
            }[command]
            self.assertIn(expected_tool, command_text)
            self.assertIn("disable-model-invocation: true", command_text)
            for secret_boundary in (".env", "API keys", "tokens"):
                self.assertIn(secret_boundary, command_text)
            if command in {"build", "route"}:
                self.assertIn("argument-hint:", command_text)

        self.assertEqual(
            {
                path.relative_to(plugin_root).as_posix()
                for path in plugin_root.rglob("*")
                if path.is_file()
            },
            {
                ".claude-plugin/plugin.json",
                "README.md",
                "commands/build.md",
                "commands/doctor.md",
                "commands/route.md",
            },
        )

    def test_claude_code_plugin_requests_machine_readable_cli_output(self) -> None:
        plugin_root = PUBLIC_ROOT / "plugins" / "claude-code" / "skillfabric"
        plugin_text = "\n".join(
            path.read_text(encoding="utf-8") for path in plugin_root.rglob("*.md")
        )
        self.assertNotIn("allowed-tools: Bash(skillfabric *)", plugin_text)

        for command in (
            "skillfabric doctor-state --json",
            "skillfabric build --json",
            "skillfabric route --json",
        ):
            self.assertIn(command, plugin_text)
        self.assertNotIn("!`skillfabric build", plugin_text)
        self.assertNotIn("!`skillfabric route", plugin_text)
        self.assertIn("Treat the user's argument as a path value", plugin_text)
        self.assertIn("Treat it as data, never as shell syntax", plugin_text)
        self.assertIn("<skill-root>\n$ARGUMENTS\n</skill-root>", plugin_text)
        self.assertIn("<task>\n$ARGUMENTS\n</task>", plugin_text)
        self.assertIn('skillfabric route --json -- "$ARGUMENTS"', plugin_text)
        for removed_contract in (
            "skillfabric plan",
            "skillfabric run-state",
            "execution_prompt",
            "planner_validation",
        ):
            self.assertNotIn(removed_contract, plugin_text)

    def test_claude_doctor_command_uses_the_current_state_contract(self) -> None:
        doctor = (
            PUBLIC_ROOT / "plugins" / "claude-code" / "skillfabric" / "commands" / "doctor.md"
        ).read_text(encoding="utf-8")

        for field in (
            "api_configured",
            "llm_configured",
            "embedding_configured",
            "missing_configuration",
            "workspace_ready",
            "build_id",
            "skill_count",
            "next_action",
        ):
            self.assertIn(f"`{field}`", doctor)
        self.assertNotIn("workspace_status", doctor)

    def test_claude_code_marketplace_registers_the_public_plugin(self) -> None:
        marketplace_path = (
            PUBLIC_ROOT / "plugins" / "claude-code" / ".claude-plugin" / "marketplace.json"
        )
        marketplace = json.loads(marketplace_path.read_text(encoding="utf-8"))
        assert marketplace["name"] == "skillfabric"
        assert marketplace["plugins"] == [
            {
                "name": "skillfabric",
                "description": "Compile native skills and route tasks through a task specific Wiki",
                "source": "./skillfabric",
                "category": "productivity",
            }
        ]

    def test_claude_code_plugin_readme_describes_real_runtime_requirements(self) -> None:
        readme = (PUBLIC_ROOT / "plugins" / "claude-code" / "skillfabric" / "README.md").read_text(
            encoding="utf-8"
        )
        for snippet in (
            "/skillfabric:doctor",
            "/skillfabric:build",
            "/skillfabric:route",
            "embedding endpoint",
            "does not execute the task",
            "claude plugin marketplace add",
            "Do not put API keys in Claude Code prompts",
        ):
            assert snippet in readme

    def test_root_readme_documents_claude_code_plugin(self) -> None:
        readme = (PUBLIC_ROOT / "README.md").read_text(encoding="utf-8")
        for section in (
            "## Quick Start",
            "## Agent Integrations",
            "## Citation",
        ):
            self.assertIn(section, readme)
        for snippet in (
            "/skillfabric:doctor",
            "/skillfabric:build",
            "/skillfabric:route",
            "claude plugin marketplace add",
            "claude plugin install skillfabric@skillfabric",
            "task specific Wiki",
        ):
            self.assertIn(snippet, readme)
        self.assertNotIn("/skillfabric:prepare", readme)
        self.assertNotIn("/skillfabric:run", readme)
        self.assertNotIn("## Development", readme)
        self.assertNotIn("## Contributing", readme)
        self.assertIn("## Citation", readme)
        self.assertIn("@article{liang2026skillnet", readme)

    def test_public_docs_do_not_leak_local_paths(self) -> None:
        readme = PUBLIC_ROOT / "README.md"
        text = readme.read_text(encoding="utf-8")
        self.assertNotIn("/Users/", text, readme)

    def test_python_facade_exposes_only_stable_route_options(self) -> None:
        from skillfabric import SkillFabric

        client = SkillFabric(workspace=".skillfabric")
        invalid_overrides = [{"max_selected_skills": True}]

        with patch("skillfabric.api.route_task") as route_mock:
            for overrides in invalid_overrides:
                with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                    client.route("extract financial KPIs", **overrides)

        route_mock.assert_not_called()

    def test_python_facade_plans_once_and_preserves_original_task(self) -> None:
        from skillfabric import SkillFabric

        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            env_file = Path(tmp) / ".env.test"
            env_file.write_text(
                "API_KEY=test-key\nBASE_URL=https://example.test/v1\n",
                encoding="utf-8",
            )
            client = SkillFabric(workspace=workspace, env_file=env_file)
            with patch(
                "skillfabric.planner.package.litellm_completion",
                return_value=json.dumps(
                    {
                        "execution_prompt": (
                            "Use `skill:pdf-table-parser` to parse the PDF, then use "
                            "`skill:financial-kpi-extractor` to extract the KPIs and verify "
                            "each value."
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
            self.assertIn("`skill:pdf-table-parser` to parse the PDF", prompt)

    def test_python_facade_rejects_routing_options_for_an_existing_route(self) -> None:
        from skillfabric import SkillFabric

        with self.assertRaisesRegex(TypeError, "routing options require"):
            SkillFabric(workspace=".skillfabric").plan(
                "extract financial KPIs",
                route=_facade_route(),
                backend="codex",
            )

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
                )

    def test_distribution_configuration_excludes_private_runtime_artifacts(self) -> None:
        pyproject = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        targets = pyproject["tool"]["hatch"]["build"]["targets"]

        self.assertEqual(targets["wheel"]["packages"], ["src/skillfabric"])
        self.assertEqual(
            targets["sdist"]["include"],
            ["/README.md", "/LICENSE", "/pyproject.toml", "/src", "/tests"],
        )
        self.assertEqual(
            set(targets["sdist"]["exclude"]),
            {
                "/.env",
                "/.env.*",
                "/models",
                "/runs",
                "/experiments",
                "/__pycache__",
                "**/__pycache__",
                "*.pyc",
                "**/*.pyc",
            },
        )


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
                source_skill="skill:pdf-table-parser",
                target_skill="skill:financial-kpi-extractor",
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
