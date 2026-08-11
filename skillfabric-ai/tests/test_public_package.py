from __future__ import annotations

import inspect
import json
import tomllib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tests.support import FIXTURE_SKILLS, FakeEmbeddingProvider, build_fixture_workspace

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

    def test_python_facade_exposes_and_forwards_explorer_backend(self) -> None:
        from skillfabric import SkillFabric

        backend = object()
        self.assertIn("explorer_backend", inspect.signature(SkillFabric.route).parameters)

        with patch("skillfabric.api.route_task", return_value=_facade_route()) as route:
            result = SkillFabric(workspace=".skillfabric").route(
                "extract KPIs",
                explorer_backend=backend,
            )

        self.assertEqual(result, _facade_route())
        self.assertIs(route.call_args.kwargs["explorer_backend"], backend)

    def test_python_facade_forwards_optional_exact_selection_count(self) -> None:
        from skillfabric import SkillFabric

        with patch("skillfabric.api.route_task", return_value=_facade_route()) as route:
            SkillFabric(workspace=".skillfabric").route(
                "extract KPIs",
                max_selected_skills=5,
                required_selected_skills=5,
            )

        config = route.call_args.args[0]
        self.assertEqual(config.max_selected_skills, 5)
        self.assertEqual(config.required_selected_skills, 5)

    def test_python_facade_forwards_explicit_planner_credentials(self) -> None:
        from skillfabric import SkillFabric

        expected = object()
        with patch("skillfabric.api.plan_execution_package", return_value=expected) as planner:
            result = SkillFabric(workspace=".skillfabric").plan(
                "extract KPIs",
                route=_facade_route(),
                llm_api_key="skillsbench-key",
                llm_api_base="https://skillsbench.example/v1",
                llm_timeout_seconds=0,
            )

        self.assertIs(result, expected)
        self.assertEqual(planner.call_args.kwargs["llm_api_key"], "skillsbench-key")
        self.assertEqual(
            planner.call_args.kwargs["llm_api_base"],
            "https://skillsbench.example/v1",
        )
        self.assertEqual(planner.call_args.kwargs["llm_timeout_seconds"], 0)

    def test_public_package_declares_required_build_runtime_dependencies(self) -> None:
        pyproject = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        dependencies = {
            item.split(";", 1)[0].split(">", 1)[0].split("=", 1)[0]
            for item in pyproject["project"]["dependencies"]
        }

        self.assertEqual(
            dependencies,
            {"httpx", "litellm", "numpy", "pyyaml"},
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

        public_brand_text = readme
        for transitional_url in (
            "https://github.com/zjunlp/SkillNet-Fabric",
            "https://img.shields.io/github/stars/zjunlp/SkillNet-Fabric?style=social",
        ):
            public_brand_text = public_brand_text.replace(transitional_url, "")
        self.assertNotIn("SkillNet-Fabric", public_brand_text)
        self.assertNotIn("SkillNet Fabric", public_brand_text)

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

    def test_root_readme_documents_claude_code_plugin(self) -> None:
        readme = (PUBLIC_ROOT / "README.md").read_text(encoding="utf-8")
        for section in (
            "## Quick Start",
            "## Claude Code Plugin",
            "## Development",
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
            "Do not paste API keys into Claude Code",
            "The CLI is the only writer",
            "The plugin installs no hooks",
            "To uninstall",
        ):
            self.assertIn(snippet, readme)

    def test_public_docs_do_not_leak_local_paths(self) -> None:
        readme = PUBLIC_ROOT / "README.md"
        text = readme.read_text(encoding="utf-8")
        self.assertNotIn("/Users/", text, readme)

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
            {"llm_concurrency": True},
            {"embedding_model": 123},
        ]

        with patch("skillfabric.api.build_graph") as build_mock:
            for overrides in invalid_overrides:
                with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                    client.build(FIXTURE_SKILLS, **overrides)

        build_mock.assert_not_called()

    def test_python_facade_forwards_build_only_llm_overrides(self) -> None:
        from skillfabric import SkillFabric

        client = SkillFabric(workspace=".skillfabric")
        with (
            patch("skillfabric.api.build_graph") as build_mock,
            patch("skillfabric.api.build_wiki"),
        ):
            client.build(
                FIXTURE_SKILLS,
                embedding_provider=FakeEmbeddingProvider(),
                llm_model="openai/responses/gpt-5.6-luna",
                llm_reasoning_effort="medium",
                llm_checkpoint_interval=25,
                llm_circuit_breaker_threshold=7,
            )

        config = build_mock.call_args.args[0]
        self.assertEqual(config.llm_model, "openai/responses/gpt-5.6-luna")
        self.assertEqual(config.llm_reasoning_effort, "medium")
        self.assertEqual(config.llm_options.checkpoint_interval, 25)
        self.assertEqual(config.llm_options.circuit_breaker_threshold, 7)

    def test_python_facade_rejects_removed_wiki_summary_mode(self) -> None:
        from skillfabric import SkillFabric

        client = SkillFabric(workspace=".skillfabric")
        with (
            patch("skillfabric.api.build_graph") as build_mock,
            patch("skillfabric.api.build_wiki"),
            self.assertRaisesRegex(TypeError, "wiki_summary_mode"),
        ):
            client.build(
                FIXTURE_SKILLS,
                embedding_provider=FakeEmbeddingProvider(),
                wiki_summary_mode="off",
            )

        build_mock.assert_not_called()

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
            client = SkillFabric(workspace=workspace)
            with patch(
                "skillfabric.orchestrator.package.litellm_completion",
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
                    env_file=env_file,
                )

            planner.assert_called_once()
            self.assertEqual(result.prompt_path, result.root / "execution_prompt.md")
            self.assertTrue(result.prompt_path.exists())
            prompt = result.prompt_path.read_text(encoding="utf-8")
            self.assertIn("extract financial KPIs from a PDF report", prompt)
            self.assertIn("`skill:pdf-table-parser` to parse the PDF", prompt)

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

    def test_python_facade_forwards_explicit_planner_usage_log(self) -> None:
        from skillfabric import SkillFabric

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / ".skillfabric"
            build_fixture_workspace(workspace)
            usage_log_path = root / "downstream-run" / "planner_usage.jsonl"

            with patch("skillfabric.api.plan_execution_package") as planner:
                SkillFabric(workspace=workspace).plan(
                    "original task",
                    route=_facade_route(),
                    usage_log_path=usage_log_path,
                )

            self.assertEqual(
                planner.call_args.kwargs["usage_log_path"],
                usage_log_path.resolve(),
            )

    def test_python_facade_forwards_explicit_planner_runtime_identity(self) -> None:
        from skillfabric import SkillFabric

        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)

            with patch("skillfabric.api.plan_execution_package") as planner:
                SkillFabric(workspace=workspace).plan(
                    "original task",
                    route=_facade_route(),
                    llm_model="gpt-5.5",
                    llm_reasoning_effort="xhigh",
                )

            self.assertEqual(planner.call_args.kwargs["llm_model"], "gpt-5.5")
            self.assertEqual(
                planner.call_args.kwargs["llm_reasoning_effort"],
                "xhigh",
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

    def test_distribution_configuration_excludes_private_runtime_artifacts(self) -> None:
        pyproject = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        targets = pyproject["tool"]["hatch"]["build"]["targets"]

        self.assertEqual(targets["wheel"]["packages"], ["src/skillfabric"])
        self.assertEqual(targets["sdist"]["include"], ["/pyproject.toml", "/src", "/tests"])
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
