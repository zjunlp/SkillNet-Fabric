from __future__ import annotations

import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from skillfabric.cli import main as cli_main
from skillfabric.runtime.defaults import default_build_options, default_router_options
from skillfabric.runtime.progress import ProgressReporter

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_SKILLS = ROOT / "fixtures" / "skills"


class RuntimeControlsTests(unittest.TestCase):
    def test_public_defaults_use_llm_router(self) -> None:
        build = default_build_options()
        router = default_router_options()

        self.assertEqual(build.embedding_provider, "api")
        self.assertEqual(build.wiki_summary_mode, "off")
        self.assertTrue(router.use_llm_router)
        self.assertEqual(router.explorer_backend, "claude-code")

    def test_build_help_does_not_expose_cost_estimation_flags(self) -> None:
        stdout = io.StringIO()
        with self.assertRaises(SystemExit) as raised:
            with contextlib.redirect_stdout(stdout):
                cli_main(["build", "--help"])

        self.assertEqual(raised.exception.code, 0)
        help_text = stdout.getvalue()
        self.assertNotIn("--estimate-only", help_text)
        self.assertNotIn("--budget-usd", help_text)
        self.assertNotIn("--skip-llm-validation", help_text)

    def test_build_rejects_removed_skip_llm_validation_flag(self) -> None:
        with TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "API_KEY=sk-test\n"
                "BASE_URL=https://api.example.test/v1\n"
                "EMBEDDING_MODEL=openai/text-embedding-3-small\n",
                encoding="utf-8",
            )
            workspace = Path(tmp) / ".skillfabric"

            with patch("skillfabric.cli.build_graph") as build_graph_mock:
                with self.assertRaises(SystemExit):
                    with contextlib.redirect_stderr(io.StringIO()):
                        cli_main(
                            [
                                "build",
                                "--skill-root",
                                str(FIXTURE_SKILLS),
                                "--workspace",
                                str(workspace),
                                "--env-file",
                                str(env_file),
                                "--skip-llm-validation",
                                "--embedding-provider",
                                "api",
                                "--skip-wiki",
                            ]
                        )

            self.assertFalse(build_graph_mock.called)

    def test_build_rejects_unknown_embedding_provider_from_shell(self) -> None:
        with TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "API_KEY=sk-test\n"
                "BASE_URL=https://api.example.test/v1\n"
                "MODEL=openai/test-model\n"
                "EMBEDDING_MODEL=openai/text-embedding-3-small\n",
                encoding="utf-8",
            )
            workspace = Path(tmp) / ".skillfabric"

            with patch.dict("os.environ", {"EMBEDDING_PROVIDER": "custom-provider"}, clear=False):
                with patch("skillfabric.cli.build_graph") as build_graph_mock:
                    with self.assertRaisesRegex(SystemExit, "unsupported embedding provider"):
                        cli_main(
                            [
                                "build",
                                "--skill-root",
                                str(FIXTURE_SKILLS),
                                "--workspace",
                                str(workspace),
                                "--env-file",
                                str(env_file),
                            ]
                        )

            self.assertFalse(build_graph_mock.called)

    def test_build_failure_writes_status_for_plugin_diagnostics(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            checkpoint = workspace / "checkpoint.json"
            workspace.mkdir(parents=True)
            checkpoint.write_text(
                json.dumps({"stage": "interface", "build_id": "build-1", "config_digest": "digest-1"}) + "\n",
                encoding="utf-8",
            )

            with patch("skillfabric.cli.build_graph", side_effect=RuntimeError("provider returned html")):
                with self.assertRaisesRegex(RuntimeError, "provider returned html"):
                    cli_main(
                        [
                            "build",
                            "--skill-root",
                            str(FIXTURE_SKILLS),
                            "--workspace",
                            str(workspace),
                            "--embedding-provider",
                            "disabled",
                            "--skip-wiki",
                        ]
                    )

            status = json.loads((workspace / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["stage"], "interface")
            self.assertEqual(status["build_id"], "build-1")
            self.assertEqual(status["error_type"], "RuntimeError")
            self.assertIn("provider returned html", status["error"])
            self.assertNotIn("sk-", json.dumps(status))

    def test_progress_json_writes_jsonl_to_stderr_not_stdout(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                cli_main(
                        [
                            "build",
                            "--skill-root",
                            str(FIXTURE_SKILLS),
                            "--workspace",
                            str(workspace),
                            "--embedding-provider",
                            "disabled",
                            "--wiki-summary-mode",
                            "off",
                            "--progress-json",
                        ]
                )

            json.loads(stdout.getvalue())
            events = [json.loads(line) for line in stderr.getvalue().splitlines() if line.strip()]
            self.assertTrue(any(event["event"] == "start" and event["phase"] == "build" for event in events))
            self.assertTrue(any(event["event"] == "finish" and event["phase"] == "build" for event in events))

    def test_progress_reporter_quiet_suppresses_events(self) -> None:
        stderr = io.StringIO()
        reporter = ProgressReporter(enabled=True, json_mode=True, quiet=True, stream=stderr)

        with reporter.phase("test.phase"):
            pass

        self.assertEqual(stderr.getvalue(), "")

    def test_build_help_exposes_api_only_embedding_options(self) -> None:
        stdout = io.StringIO()
        with self.assertRaises(SystemExit) as raised:
            with contextlib.redirect_stdout(stdout):
                cli_main(["build", "--help"])

        self.assertEqual(raised.exception.code, 0)
        help_text = stdout.getvalue()
        self.assertIn("--embedding-provider", help_text)
        self.assertIn("--embedding-model", help_text)
        self.assertIn("{api,disabled}", help_text)
        self.assertIn("api", help_text)
        self.assertIn("disabled", help_text)


if __name__ == "__main__":
    unittest.main()
