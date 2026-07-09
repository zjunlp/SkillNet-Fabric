from __future__ import annotations

import dataclasses
import importlib.util
import json
import tarfile
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from skillfabric.wiki.materializer import build_wiki
from skillfabric.wiki.models import WikiBuildConfig
from tests.unit.wiki_helpers import build_fixture_workspace

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_ROOT = PACKAGE_ROOT.parent
FIXTURE_SKILLS = Path(__file__).resolve().parents[1] / "fixtures" / "skills"


class PublicPackageTests(unittest.TestCase):
    def test_build_config_exposes_only_standard_public_fields(self) -> None:
        from skillfabric.compiled_graph.builder import BuildConfig

        public_fields = {field.name for field in dataclasses.fields(BuildConfig)}

        self.assertEqual(
            public_fields,
            {
                "skill_root",
                "workspace",
                "llm_env_path",
                "skip_llm_validation",
                "llm_options",
            },
        )

    def test_python_facade_exports_only_high_level_methods(self) -> None:
        from skillfabric import SkillFabric

        client = SkillFabric(workspace=".skillfabric")

        self.assertEqual(str(client.workspace.root), ".skillfabric")
        self.assertTrue(callable(client.build))
        self.assertTrue(callable(client.route))
        self.assertTrue(callable(client.plan))
        self.assertFalse(hasattr(client, "package"))
        self.assertFalse(hasattr(client, "build_wiki"))
        self.assertFalse(hasattr(client, "build_execution_package"))
        self.assertFalse(hasattr(client, "status"))

    def test_public_orchestrator_does_not_export_isolated_native_skill_runtime(self) -> None:
        import skillfabric.orchestrator as orchestrator

        self.assertFalse(hasattr(orchestrator, "prepare_native_skill_runtime"))
        self.assertFalse(hasattr(orchestrator, "NativeSkillRuntimeError"))
        self.assertFalse(hasattr(orchestrator, "NativeSkillRuntimeResult"))
        self.assertFalse(
            (PACKAGE_ROOT / "src" / "skillfabric" / "orchestrator" / "native_skills.py").exists()
        )

    def test_public_package_does_not_ship_experiment_only_modules(self) -> None:
        import skillfabric.orchestrator as orchestrator

        removed_specs = (
            "skillfabric.evaluation",
            "skillfabric.evaluation.evaluation",
            "skillfabric.exporters",
            "skillfabric.exporters.neo4j",
            "skillfabric.orchestrator.outcome",
            "skillfabric.task_understanding",
        )
        for module_name in removed_specs:
            with self.subTest(module=module_name):
                self.assertIsNone(_find_spec(module_name))

        for path in (
            PACKAGE_ROOT / "src" / "skillfabric" / "evaluation",
            PACKAGE_ROOT / "src" / "skillfabric" / "exporters",
            PACKAGE_ROOT / "src" / "skillfabric" / "orchestrator" / "outcome.py",
            PACKAGE_ROOT / "src" / "skillfabric" / "task_understanding.py",
            PACKAGE_ROOT / "tests" / "unit" / "test_experimental_parity.py",
            PACKAGE_ROOT / "tests" / "unit" / "test_neo4j_export.py",
        ):
            with self.subTest(path=path):
                self.assertFalse(path.exists())

        self.assertFalse(hasattr(orchestrator, "ExecutionOutcome"))
        self.assertFalse(hasattr(orchestrator, "classify_execution_outcome"))

    def test_public_package_declares_required_build_runtime_dependencies(self) -> None:
        pyproject = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")

        for dependency in (
            "litellm",
            "networkx",
            "python-dotenv",
            "pyyaml",
            "rapidfuzz",
        ):
            with self.subTest(dependency=dependency):
                self.assertIn(f'"{dependency}', pyproject)

        self.assertNotIn("embedding-model-path", pyproject)

    def test_claude_code_plugin_manifest_and_commands_exist(self) -> None:
        plugin_root = PUBLIC_ROOT / "plugins" / "claude-code" / "skillfabric"
        manifest_path = plugin_root / ".claude-plugin" / "plugin.json"

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["name"], "skillfabric")
        self.assertEqual(manifest["version"], "0.1.0")
        self.assertEqual(manifest["license"], "MIT")
        self.assertLessEqual(
            set(manifest),
            {
                "name",
                "version",
                "description",
                "author",
                "license",
                "keywords",
                "homepage",
                "repository",
                "skills",
            },
        )
        self.assertEqual(manifest["skills"], ["./"])
        for unsupported_field in ("hooks", "mcpServers"):
            self.assertNotIn(unsupported_field, manifest)
        command_dir = plugin_root / "commands"
        expected_commands = {
            "doctor": (
                "skillfabric-doctor",
                "[--env-file path] [--workspace path]",
            ),
            "build": (
                "skillfabric-build",
                "[skill-root] [--workspace path] [--env-file path]",
            ),
            "prepare": (
                "skillfabric-prepare",
                "[task] [--workspace path] [--skill-root path]",
            ),
            "run": (
                "skillfabric-run",
                "[task] [--workspace path] [--skill-root path]",
            ),
        }
        self.assertTrue(command_dir.exists())
        self.assertEqual(
            sorted(path.stem for path in command_dir.glob("*.md")),
            sorted(expected_commands),
        )
        forbidden_tokens = (
            "<task>",
            "<workspace>",
            "<skill-root>",
            "<task-or-route-file>",
            "skillfabric scan",
            "skillfabric build-wiki",
            "skillfabric build-execution-package",
        )
        for command, required_snippets in expected_commands.items():
            command_path = command_dir / f"{command}.md"
            self.assertTrue(command_path.exists(), f"missing slash command: {command_path}")
            command_text = command_path.read_text(encoding="utf-8")
            self.assertIn("description:", command_text)
            self.assertIn("argument-hint:", command_text)
            if command == "run":
                self.assertIn("allowed-tools:", command_text)
                self.assertIn("!`skillfabric run-state $ARGUMENTS`", command_text)
                self.assertIn("Run State JSON", command_text)
                self.assertIn("Do not use `find`, `grep`, `rg`, or `ls`", command_text)
            if command == "doctor":
                self.assertIn("allowed-tools:", command_text)
                self.assertIn("!`skillfabric doctor-state $ARGUMENTS`", command_text)
                self.assertIn("Doctor State JSON", command_text)
                self.assertIn("Do not use `find`, `grep`, `rg`, `sed`, `cat`, or directory scans", command_text)
            for section in (
                "# Command Contract",
                "## Inputs",
                "## Required Workflow",
                "## Boundaries",
                "## Completion Criteria",
            ):
                self.assertIn(section, command_text)
            self.assertIn(f"Use the `skillfabric-{command}` skill as the authoritative workflow.", command_text)
            self.assertIn("Never reveal env-file contents, API keys, tokens, or shell secret values.", command_text)
            for snippet in required_snippets:
                self.assertIn(snippet, command_text)

            path = plugin_root / "skills" / f"skillfabric-{command}" / "SKILL.md"
            self.assertTrue(path.exists(), f"missing command skill: {path}")
            text = path.read_text(encoding="utf-8")
            self.assertIn(f"name: skillfabric-{command}", text)
            self.assertIn("disable-model-invocation: true", text)
            self.assertIn("$ARGUMENTS", text)
            if command == "run":
                self.assertIn("execute the user's task", text.lower())
                self.assertIn("skillfabric run-state", text)
                self.assertIn("Run State JSON", text)
            if command == "doctor":
                self.assertIn("Write a concise readiness summary", text)
                self.assertIn("not a raw JSON field dump", text)
                self.assertIn("Do not list every `present`", text)
            for snippet in required_snippets:
                if snippet.startswith("["):
                    continue
                self.assertIn(snippet, command_text + text)
            if command != "doctor":
                self.assertNotIn("--profile", text)
            if command in {"build", "prepare", "run"}:
                self.assertIn("--progress-json", text)
                self.assertNotIn("--budget-usd", text)
                self.assertNotIn("--estimate-only", text)
            if command in {"prepare", "run"}:
                self.assertNotIn("--task-atoms-file", text)
                self.assertNotIn("TaskAtoms Schema", text)
                self.assertNotIn("Do not output `skill_id`", text)
            for token in forbidden_tokens:
                self.assertNotIn(token, command_text + text)
            self.assertNotIn("print `.env`", text.lower())
            self.assertNotIn("print api key", text.lower())
            self.assertNotIn("claude-agent-sdk", text)
        for removed in ("init", "route", "plan", "assist", "execute"):
            self.assertFalse((command_dir / f"{removed}.md").exists())
            self.assertFalse((plugin_root / "skills" / removed).exists())
            self.assertFalse((plugin_root / "skills" / removed / "SKILL.md").exists())
            self.assertFalse((plugin_root / "skills" / f"skillfabric-{removed}" / "SKILL.md").exists())
        for public_name in expected_commands:
            self.assertFalse((plugin_root / "skills" / public_name).exists())
        self.assertFalse((plugin_root / "agents").exists())
        self.assertFalse((plugin_root / "hooks").exists())
        self.assertFalse((plugin_root / ".mcp.json").exists())
        self.assertTrue((plugin_root / "SKILL.md").exists())
        self.assertFalse((plugin_root / "skills" / "skillfabric" / "SKILL.md").exists())

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
            self.assertNotIn("--budget-usd", text)
            self.assertNotIn("--estimate-only", text)

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
            self.assertIn(f"Use the `skillfabric-{path.stem}` skill as the authoritative workflow.", text)
            if path.stem == "doctor":
                self.assertIn("Doctor State JSON", text)
                self.assertIn("Report readiness from that JSON only.", text)
            if path.stem in {"prepare", "run"}:
                self.assertIn("Treat all non-option text in `$ARGUMENTS` as the task query.", text)
                self.assertIn("Preserve the user's wording", text)
                self.assertIn("execution_prompt.md", text)
                self.assertIn("Do not stop after only loading the skill", text)
                if path.stem == "prepare":
                    self.assertIn("Run SkillFabric route prepare and route finalize.", text)
                    self.assertIn("Run SkillFabric plan prepare and plan finalize.", text)
                    self.assertIn("Do not answer or perform the user's task", text)
                    self.assertIn("route finalization", text)
                    self.assertIn("plan finalization", text)
                    self.assertIn("have run", text)
                else:
                    self.assertIn("Run State JSON", text)
                    self.assertIn("prepare/finalize and plan prepare/finalize", text)
                    self.assertIn("before task tools, search, or final answers", text)
                    self.assertIn("task tools, search, or final answers", text)
            if path.stem == "build":
                self.assertIn("the CLI build must", text)

        overview_text = (plugin_root / "SKILL.md").read_text(encoding="utf-8")
        self.assertIn("## Command Choice", overview_text)
        self.assertIn("Prefer the four slash commands", overview_text)
        self.assertIn("/skillfabric:doctor", overview_text)
        self.assertIn("/skillfabric:run", overview_text)
        self.assertNotIn("Run `skillfabric route", overview_text)
        self.assertFalse((plugin_root / "skills" / "skillfabric").exists())

    def test_claude_code_prepare_and_run_prompts_match_agent_cli_contract(self) -> None:
        plugin_root = PUBLIC_ROOT / "plugins" / "claude-code" / "skillfabric"
        for name in ("prepare", "run"):
            text = (plugin_root / "skills" / f"skillfabric-{name}" / "SKILL.md").read_text(
                encoding="utf-8"
            )
            with self.subTest(command=name):
                self.assertIn("The task is the user's natural-language request", text)
                self.assertIn("Treat only recognized flags as workflow configuration.", text)
                self.assertIn('skillfabric route "$task"', text)
                self.assertIn("--agent-mode prepare", text)
                self.assertIn("--agent-mode finalize", text)
                self.assertIn('--trace-id "$trace_id"', text)
                self.assertIn('--skill-package-file "$skill_package_file"', text)
                self.assertIn("single raw SkillPackage JSON object", text)
                self.assertIn("to the returned\n   `skill_package_file`", text)
                self.assertIn('skillfabric plan --workspace "$workspace"', text)
                self.assertIn('--route-file "$route_json"', text)
                self.assertIn('--package-root "$package_root"', text)
                self.assertIn('--planner-output-file "$planner_output_path"', text)
                self.assertIn("single raw planner JSON object", text)
                self.assertIn("to the returned\n    `planner_output_path`", text)
                self.assertIn("Use real newline characters inside `execution_prompt`", text)
                self.assertIn("Do not wrap it in Markdown fences", text)
                self.assertIn("comments, or", text)
                self.assertIn("explanatory text", text)
                if name == "run":
                    self.assertIn('skillfabric run-state "$task"', text)
                    self.assertIn('set `$execution_prompt` to `prompt_path`', text)
                    self.assertIn("Before reading `execution_prompt.md`, do not use Web Search, Fetch", text)
                    self.assertIn("If no reusable prompt exists and no task was provided", text)
                    self.assertIn("Do not substitute shell path discovery for `run-state`", text)
                self.assertIn("agent_route_request.json", text)
                self.assertIn("planner_request.json", text)
                self.assertIn("query_wiki_root", text)
                self.assertIn("In the main Claude Code session", text)
                self.assertIn("Do not inspect the active project workspace", text)
                self.assertIn("Route selection is about choosing skills", text)
                self.assertIn("bounded active-workspace inspection", text)
                self.assertIn("execution_prompt.md", text)
                self.assertNotIn("Launch `skillfabric-query-wiki-explorer`", text)
                self.assertNotIn("Launch `skillfabric-workflow-planner`", text)
                self.assertIn("$ARGUMENTS", text)

    def test_claude_code_plugin_prompts_handle_natural_language_commands(self) -> None:
        plugin_root = PUBLIC_ROOT / "plugins" / "claude-code" / "skillfabric"
        build_text = (plugin_root / "skills" / "skillfabric-build" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        doctor_text = (plugin_root / "skills" / "skillfabric-doctor" / "SKILL.md").read_text(
            encoding="utf-8"
        )

        self.assertIn("natural language that may contain a skill root", build_text)
        self.assertIn("first existing directory as the skill root", build_text)
        self.assertIn("Ignore surrounding natural language", build_text)
        self.assertIn("contains at least one `SKILL.md` or", build_text)
        self.assertIn("not built yet", doctor_text)
        self.assertIn("normal before\n  the first build", doctor_text)
        self.assertIn("SkillFabric ready.", doctor_text)
        self.assertIn("Workspace: ready, <skill_count> skills, build <build_id>", doctor_text)

    def test_claude_code_plugin_readme_is_product_grade(self) -> None:
        readme = (
            PUBLIC_ROOT / "plugins" / "claude-code" / "skillfabric" / "README.md"
        ).read_text(encoding="utf-8")
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
            "--embedding-provider disabled",
            "Do not paste API keys",
        ):
            self.assertIn(snippet, readme)
        for removed_command in (
            "/skillfabric:init",
            "/skillfabric:route",
            "/skillfabric:plan",
            "/skillfabric:assist",
            "/skillfabric:execute",
        ):
            self.assertNotIn(removed_command, readme)
        for internal_reference in (
            "/Users/chenjiang",
            "SkillNet/experiments",
            "AgentSkillOS",
            "benchmark",
        ):
            self.assertNotIn(internal_reference, readme)

    def test_public_docs_do_not_reference_internal_experiment_surfaces(self) -> None:
        doc_paths = [
            PUBLIC_ROOT / "README.md",
            PACKAGE_ROOT / "README.md",
            PACKAGE_ROOT / "tests" / "README.md",
            PUBLIC_ROOT / "plugins" / "claude-code" / "skillfabric" / "README.md",
        ]
        forbidden = (
            "/Users/chenjiang",
            "SkillNet/experiments",
            "AgentSkillOS",
            "SKILLFABRIC_EXPERIMENTAL_ROOT",
            "eval aggregation",
            "eval_report",
        )
        for path in doc_paths:
            text = path.read_text(encoding="utf-8")
            for marker in forbidden:
                with self.subTest(path=path, marker=marker):
                    self.assertNotIn(marker, text)

    def test_python_facade_plan_requires_agent_planner(self) -> None:
        from skillfabric import SkillFabric

        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))
            client = SkillFabric(workspace=workspace)

            route = client.route(
                "extract financial KPIs from a PDF report",
                use_llm_router=False,
                explorer_backend="fallback",
            )

            with self.assertRaisesRegex(ValueError, "no longer creates a finalized execution package"):
                client.plan(route=route, renderer="codex")

    def test_python_facade_prepare_and_finalize_plan_prompt_only(self) -> None:
        from skillfabric import SkillFabric

        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))
            client = SkillFabric(workspace=workspace)

            route = client.route(
                "extract financial KPIs from a PDF report",
                use_llm_router=False,
                explorer_backend="fallback",
            )
            prepared = client.prepare_plan(route=route, renderer="codex")
            self.assertTrue((prepared.root / "planner_request.json").exists())
            self.assertFalse((prepared.root / "agent_run_spec_draft.json").exists())
            self.assertFalse((prepared.root / "execution_prompt.md").exists())

            result = client.finalize_plan(
                prepared.root,
                {"execution_prompt": "# Prompt\n\nExtract the KPIs and verify the deliverable."},
                renderer="codex",
            )

            self.assertEqual(result.renderer, "codex")
            self.assertEqual(result.prompt_path, prepared.root / "execution_prompt.md")
            self.assertTrue(result.prompt_path.exists())
            self.assertFalse((prepared.root / "workflow_plan.json").exists())
            self.assertFalse((prepared.root / "agent_run_spec.json").exists())

    def test_python_facade_build_accepts_explicit_disabled_embeddings_without_api(self) -> None:
        from skillfabric import SkillFabric

        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            client = SkillFabric(workspace=workspace)

            result = client.build(
                FIXTURE_SKILLS,
                embedding_provider="disabled",
                embedding_model="openai/text-embedding-3-small",
                skip_llm_validation=True,
                wiki_summary_mode="off",
            )

            self.assertEqual(result.stats["embedding_model_id"], "disabled")
            self.assertTrue((workspace / "wiki" / "index.md").exists())
            metrics = json.loads((workspace / "reports" / "build_summary.json").read_text(encoding="utf-8"))
            self.assertIn("wiki_summary", metrics)
            self.assertEqual(
                metrics["wiki_summary"]["fallback_count"],
                result.stats["skill_count"],
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


def _find_spec(module_name: str) -> object | None:
    try:
        return importlib.util.find_spec(module_name)
    except ModuleNotFoundError:
        return None


if __name__ == "__main__":
    unittest.main()
