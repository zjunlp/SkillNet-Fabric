from __future__ import annotations

import asyncio
import fnmatch
import json
import os
import sys
import tomllib
import unittest
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from skillfabric.wiki.explorer.backends.codex import (
    CODEX_EXECUTION_CONTRACT,
    CodexWikiExplorerBackend,
    _load_sdk_runtime,
)
from skillfabric.wiki.explorer.skill_package import skill_package_json_schema


def _package() -> dict[str, object]:
    return {
        "selected_skills": [],
        "near_misses": [],
        "coverage_gaps": ["No relevant skill was found."],
        "wiki_pages_read": [],
        "rationale": "The bounded query wiki does not cover the request.",
    }


def _event(method: str, **payload: object) -> SimpleNamespace:
    return SimpleNamespace(method=method, payload=SimpleNamespace(**payload))


def _item(item_type: str, **fields: object) -> SimpleNamespace:
    return SimpleNamespace(root=SimpleNamespace(type=item_type, **fields))


def _success_events(payload: dict[str, object] | None = None) -> list[SimpleNamespace]:
    response = json.dumps(payload or _package())
    usage = SimpleNamespace(
        total=SimpleNamespace(
            input_tokens=20,
            cached_input_tokens=4,
            output_tokens=6,
            reasoning_output_tokens=3,
            total_tokens=26,
        )
    )
    return [
        _event(
            "item/completed",
            item=_item("agentMessage", text=response, phase=SimpleNamespace(value="final_answer")),
        ),
        _event("thread/tokenUsage/updated", token_usage=usage),
        _event(
            "turn/completed",
            turn=SimpleNamespace(
                status=SimpleNamespace(value="completed"),
                error=None,
                duration_ms=25,
            ),
        ),
    ]


class _ReasoningEffort(StrEnum):
    medium = "medium"


class _CodexConfig:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class _Turn:
    def __init__(self, runtime: _Runtime, events: list[SimpleNamespace]) -> None:
        self.runtime = runtime
        self.id = "turn-1"
        self.events = events
        self.interrupts = 0

    async def stream(self):
        for event in self.events:
            if self.runtime.event_delay:
                await asyncio.sleep(self.runtime.event_delay)
            yield event

    async def interrupt(self) -> None:
        self.interrupts += 1
        self.runtime.lifecycle.append("interrupt")


class _Thread:
    def __init__(self, runtime: _Runtime) -> None:
        self.runtime = runtime
        self.id = "thread-1"

    async def turn(self, prompt: str, **kwargs: object) -> _Turn:
        self.runtime.turn_calls.append((prompt, kwargs))
        turn = _Turn(self.runtime, list(self.runtime.events))
        self.runtime.turns.append(turn)
        return turn


class _Codex:
    def __init__(self, runtime: _Runtime, config: _CodexConfig) -> None:
        self.runtime = runtime
        self.config = config
        self.metadata = SimpleNamespace(
            serverInfo=SimpleNamespace(name="codex-app-server", version="0.144.4"),
            userAgent="codex/0.144.4",
        )

    async def __aenter__(self) -> _Codex:
        self.runtime.configs.append(self.config)
        if self.runtime.enter_delay:
            await asyncio.sleep(self.runtime.enter_delay)
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.close()

    async def close(self) -> None:
        self.runtime.closes += 1
        self.runtime.lifecycle.append("close")
        if self.runtime.fail_close:
            raise RuntimeError("app-server close failed")

    async def thread_start(self, **kwargs: object) -> _Thread:
        self.runtime.thread_calls.append(kwargs)
        if self.runtime.fail_thread_start:
            home = self.config.kwargs["env"]["CODEX_HOME"]
            raise RuntimeError(
                f"app-server failed under {home}: OPENAI_API_KEY=sk-runtime-secret"
            )
        return _Thread(self.runtime)


class _Runtime:
    CodexConfig = _CodexConfig
    ApprovalMode = SimpleNamespace(deny_all="deny-all")
    ReasoningEffort = _ReasoningEffort
    __version__ = "0.0.0-test"

    def __init__(
        self,
        events: list[SimpleNamespace] | None = None,
        *,
        event_delay: float = 0,
        enter_delay: float = 0,
        fail_close: bool = False,
        fail_thread_start: bool = False,
    ) -> None:
        self.events = events or _success_events()
        self.event_delay = event_delay
        self.enter_delay = enter_delay
        self.fail_close = fail_close
        self.fail_thread_start = fail_thread_start
        self.configs: list[_CodexConfig] = []
        self.thread_calls: list[dict[str, object]] = []
        self.turn_calls: list[tuple[str, dict[str, object]]] = []
        self.turns: list[_Turn] = []
        self.closes = 0
        self.lifecycle: list[str] = []

    def AsyncCodex(self, *, config: _CodexConfig) -> _Codex:
        return _Codex(self, config)


class CodexWikiExplorerTests(unittest.TestCase):
    def test_runtime_loader_uses_the_sdk_types_reasoning_effort_export(self) -> None:
        package = ModuleType("openai_codex")
        package.__path__ = []  # type: ignore[attr-defined]
        package.__version__ = "0.144.4"  # type: ignore[attr-defined]
        package.ApprovalMode = object()  # type: ignore[attr-defined]
        package.AsyncCodex = object()  # type: ignore[attr-defined]
        package.CodexConfig = object()  # type: ignore[attr-defined]
        types_module = ModuleType("openai_codex.types")
        types_module.ReasoningEffort = _ReasoningEffort  # type: ignore[attr-defined]

        with patch.dict(
            sys.modules,
            {"openai_codex": package, "openai_codex.types": types_module},
        ):
            runtime = _load_sdk_runtime()

        self.assertIs(runtime.ReasoningEffort, _ReasoningEffort)

    def test_backend_is_public_and_codex_sdk_is_an_optional_dependency(self) -> None:
        from skillfabric.wiki.explorer.backends import CodexWikiExplorerBackend as PublicBackend

        package_root = Path(__file__).resolve().parents[2]
        pyproject = tomllib.loads(
            (package_root / "pyproject.toml").read_text(encoding="utf-8")
        )
        extras = pyproject["project"]["optional-dependencies"]

        self.assertIs(PublicBackend, CodexWikiExplorerBackend)
        self.assertEqual(extras["codex"], ["openai-codex"])
        self.assertIn("openai-codex", extras["all"])

    def test_success_uses_a_locked_down_ephemeral_thread_and_writes_artifacts(self) -> None:
        runtime = _Runtime()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            wiki = root / "query_wiki"
            trace = root / "trace"
            wiki.mkdir()
            env_file = root / ".env"
            env_file.write_text(
                "SKILLFABRIC_LLM_API_BASE=http://gateway.example/v1\n"
                "SKILLFABRIC_LLM_API_KEY=sk-experiment-secret\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"CODEX_API_KEY": "personal-token", "OPENAI_API_KEY": "personal-key"},
            ):
                package = CodexWikiExplorerBackend(
                    env_file=env_file,
                    max_selected_skills=5,
                    model="gpt-5.6-terror",
                    reasoning_effort="medium",
                    execution_timeout_seconds=60,
                    execution_contract=CODEX_EXECUTION_CONTRACT.to_dict(),
                    sdk_runtime=runtime,
                ).explore(query="find a skill", query_wiki_root=wiki, trace_dir=trace)

            self.assertEqual(package.to_dict(), _package())
            config = runtime.configs[0].kwargs
            self.assertEqual(config["env"]["OPENAI_API_KEY"], "sk-experiment-secret")
            self.assertEqual(config["env"]["CODEX_API_KEY"], "")
            codex_home = Path(config["env"]["CODEX_HOME"])
            self.assertFalse(codex_home.exists())
            self.assertEqual(config["cwd"], str(codex_home))
            self.assertEqual(
                config["config_overrides"],
                (
                    "project_root_markers=[]",
                    "check_for_update_on_startup=false",
                ),
            )

            thread = runtime.thread_calls[0]
            self.assertEqual(thread["approval_mode"], "deny-all")
            self.assertEqual(thread["cwd"], str(wiki.resolve()))
            self.assertEqual(thread["model"], "gpt-5.6-terror")
            self.assertTrue(thread["ephemeral"])
            self.assertNotIn("sandbox", thread)
            thread_config = thread["config"]
            self.assertEqual(thread_config["web_search"], "disabled")
            self.assertEqual(thread_config["openai_base_url"], "http://gateway.example/v1")
            self.assertEqual(thread_config["project_root_markers"], [])
            profile = thread_config["permissions"]["skillfabric-query-wiki"]
            self.assertEqual(
                profile["filesystem"],
                {":minimal": "read", str(wiki.resolve()): "read"},
            )
            self.assertFalse(profile["network"]["enabled"])
            disabled_features = {
                "apps",
                "browser_use",
                "browser_use_external",
                "browser_use_full_cdp_access",
                "code_mode",
                "code_mode_host",
                "code_mode_only",
                "collab",
                "collaboration_modes",
                "computer_use",
                "connectors",
                "enable_mcp_apps",
                "external_agent_memory_import",
                "image_generation",
                "imagegenext",
                "memory_tool",
                "multi_agent",
                "multi_agent_mode",
                "multi_agent_v2",
                "network_proxy",
                "plugin_hooks",
                "plugin_sharing",
                "plugins",
                "remote_plugin",
                "request_permissions",
                "request_permissions_tool",
                "request_rule",
                "search_tool",
                "skill_mcp_dependency_install",
                "skill_search",
                "standalone_web_search",
                "tool_search",
                "tool_suggest",
                "web_search",
                "web_search_cached",
                "web_search_request",
            }
            self.assertTrue(disabled_features <= set(thread_config["features"]))
            self.assertTrue(
                all(thread_config["features"][name] is False for name in disabled_features)
            )
            self.assertFalse(thread_config["skills"]["bundled"]["enabled"])
            self.assertFalse(thread_config["skills"]["include_instructions"])
            self.assertEqual(thread_config["mcp_servers"], {})
            self.assertEqual(
                thread_config["orchestrator"],
                {"skills": {"enabled": False}, "mcp": {"enabled": False}},
            )
            shell_policy = thread_config["shell_environment_policy"]
            self.assertEqual(shell_policy["inherit"], "core")
            self.assertFalse(shell_policy["ignore_default_excludes"])
            self.assertEqual(
                shell_policy["include_only"],
                ["PATH", "SHELL", "HOME", "LANG", "LC_*"],
            )
            self.assertEqual(shell_policy["set"], {"HOME": str(codex_home)})
            self.assertEqual(
                shell_policy["exclude"],
                ["*PASSWORD*", "*CREDENTIAL*"],
            )
            inherited = {
                "PATH": "/usr/bin",
                "HOME": "/personal/home",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "API_PASSWORD": "secret",
                "DB_CREDENTIAL": "secret",
                "UNRELATED": "discarded",
            }
            included = {
                name
                for name in inherited
                if any(
                    fnmatch.fnmatchcase(name.casefold(), pattern.casefold())
                    for pattern in shell_policy["include_only"]
                )
                and not any(
                    fnmatch.fnmatchcase(name.casefold(), pattern.casefold())
                    for pattern in shell_policy["exclude"]
                )
            }
            self.assertEqual(included, {"PATH", "HOME", "LANG", "LC_ALL"})
            self.assertEqual(thread_config["project_doc_max_bytes"], 0)
            self.assertEqual(thread_config["project_doc_fallback_filenames"], [])

            prompt, turn = runtime.turn_calls[0]
            self.assertIn("find a skill", prompt)
            self.assertEqual(turn["effort"], _ReasoningEffort.medium)
            self.assertEqual(turn["model"], "gpt-5.6-terror")
            self.assertEqual(turn["output_schema"], skill_package_json_schema())
            self.assertEqual(turn["cwd"], str(wiki.resolve()))

            prompt_context = json.loads(
                (trace / "cc_explorer" / "prompt_context.json").read_text(encoding="utf-8")
            )
            self.assertEqual(prompt_context["allowed_tools"], ["exec_command"])
            self.assertIn("exec_command<=21", (trace / "cc_explorer" / "prompt.system.md").read_text())

            artifacts = trace / "cc_explorer"
            backend = json.loads((artifacts / "backend.json").read_text(encoding="utf-8"))
            self.assertEqual(backend["backend"], "codex")
            self.assertEqual(backend["allowed_tools"], ["exec_command"])
            self.assertEqual(backend["tool_enforcement"], "event-audited-fail-closed")
            self.assertEqual(backend["command_budget"], 21)
            self.assertEqual(backend["execution_contract"], CODEX_EXECUTION_CONTRACT.to_dict())
            usage = json.loads((artifacts / "usage.json").read_text(encoding="utf-8"))
            self.assertEqual(
                usage,
                {
                    "duration_ms": 25,
                    "input_tokens": 20,
                    "cache_read_input_tokens": 4,
                    "output_tokens": 6,
                    "reasoning_output_tokens": 3,
                    "total_tokens": 26,
                    "total_calls": 0,
                    "num_turns": 1,
                },
            )
            combined = "\n".join(
                path.read_text(encoding="utf-8") for path in artifacts.iterdir() if path.is_file()
            )
            self.assertNotIn("sk-experiment-secret", combined)
            self.assertNotIn("personal-token", combined)
            self.assertEqual(runtime.closes, 1)

            events = [
                json.loads(line)
                for line in (artifacts / "agent_events.jsonl").read_text().splitlines()
            ]
            completed = next(event for event in events if event.get("method") == "turn/completed")
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["duration_ms"], 25)

    def test_budget_ten_uses_31_commands_and_each_call_is_isolated(self) -> None:
        runtime = _Runtime()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            wiki = root / "query_wiki"
            wiki.mkdir()
            env_file = root / ".env"
            env_file.write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")
            backend = CodexWikiExplorerBackend(
                env_file=env_file,
                max_selected_skills=10,
                execution_timeout_seconds=60,
                execution_contract=CODEX_EXECUTION_CONTRACT.to_dict(),
                sdk_runtime=runtime,
            )

            backend.explore(query="first", query_wiki_root=wiki, trace_dir=root / "trace-1")
            backend.explore(query="second", query_wiki_root=wiki, trace_dir=root / "trace-2")

            homes = [config.kwargs["env"]["CODEX_HOME"] for config in runtime.configs]
            self.assertEqual(len(set(homes)), 2)
            self.assertTrue(all(not Path(home).exists() for home in homes))
            self.assertEqual(len(runtime.thread_calls), 2)
            self.assertTrue(all(call["ephemeral"] for call in runtime.thread_calls))
            for trace_name in ("trace-1", "trace-2"):
                payload = json.loads(
                    (root / trace_name / "cc_explorer" / "backend.json").read_text()
                )
                self.assertEqual(payload["command_budget"], 31)

    def test_failure_artifacts_include_app_server_metadata_and_redact_runtime_home(self) -> None:
        runtime = _Runtime(fail_thread_start=True)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            wiki = root / "query_wiki"
            wiki.mkdir()
            env_file = root / ".env"
            env_file.write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")
            backend = CodexWikiExplorerBackend(
                env_file=env_file,
                execution_timeout_seconds=60,
                execution_contract=CODEX_EXECUTION_CONTRACT.to_dict(),
                sdk_runtime=runtime,
            )

            with self.assertRaisesRegex(RuntimeError, "app-server failed"):
                backend.explore(
                    query="find a skill",
                    query_wiki_root=wiki,
                    trace_dir=root / "trace",
                )

            artifacts = root / "trace" / "cc_explorer"
            backend_payload = json.loads((artifacts / "backend.json").read_text())
            self.assertEqual(backend_payload["app_server"]["version"], "0.144.4")
            error_text = (artifacts / "error.json").read_text()
            self.assertNotIn("sk-runtime-secret", error_text)
            self.assertNotIn(runtime.configs[0].kwargs["env"]["CODEX_HOME"], error_text)

    def test_command_budget_interrupts_the_turn_and_propagates_failure(self) -> None:
        commands = [
            _event("item/started", item=_item("commandExecution", command=f"read {index}"))
            for index in range(22)
        ]
        runtime = _Runtime([*commands, *_success_events()])
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            wiki = root / "query_wiki"
            wiki.mkdir()
            env_file = root / ".env"
            env_file.write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")
            backend = CodexWikiExplorerBackend(
                env_file=env_file,
                max_selected_skills=5,
                model="gpt-5.6-terror",
                reasoning_effort="medium",
                execution_timeout_seconds=60,
                execution_contract=CODEX_EXECUTION_CONTRACT.to_dict(),
                sdk_runtime=runtime,
            )

            with self.assertRaisesRegex(RuntimeError, "tool budget exceeded"):
                backend.explore(query="find a skill", query_wiki_root=wiki, trace_dir=root / "trace")

            self.assertEqual(runtime.turns[0].interrupts, 1)
            error = json.loads(
                (root / "trace" / "cc_explorer" / "error.json").read_text(encoding="utf-8")
            )
            self.assertEqual(error["error_type"], "RuntimeError")

    def test_observed_tools_outside_exec_command_fail_closed(self) -> None:
        completed = _success_events()[-1]
        cases = {
            "write-stdin": [
                _event(
                    "item/commandExecution/terminalInteraction",
                    item_id="command-1",
                    process_id="process-1",
                    stdin="",
                )
            ],
            "view-image": [_event("item/started", item=_item("imageView"))],
            "update-plan": [_event("item/completed", item=_item("plan"))],
            "background-session": [
                _event(
                    "item/completed",
                    item=_item("commandExecution", process_id="process-1"),
                )
            ],
        }
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            wiki = root / "query_wiki"
            wiki.mkdir()
            env_file = root / ".env"
            env_file.write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")

            for name, tool_events in cases.items():
                runtime = _Runtime([*tool_events, *_success_events()[:-1], completed])
                with self.subTest(name=name), self.assertRaisesRegex(
                    RuntimeError,
                    "disallowed Codex tool activity",
                ):
                    CodexWikiExplorerBackend(
                        env_file=env_file,
                        execution_timeout_seconds=60,
                        execution_contract=CODEX_EXECUTION_CONTRACT.to_dict(),
                        sdk_runtime=runtime,
                    ).explore(
                        query="find a skill",
                        query_wiki_root=wiki,
                        trace_dir=root / name,
                    )
                self.assertEqual(runtime.turns[0].interrupts, 1)

    def test_commentary_is_not_accepted_as_the_final_skill_package(self) -> None:
        events = [
            _event(
                "item/completed",
                item=_item(
                    "agentMessage",
                    text=json.dumps(_package()),
                    phase=SimpleNamespace(value="commentary"),
                ),
            ),
            _success_events()[-1],
        ]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            wiki = root / "query_wiki"
            wiki.mkdir()
            env_file = root / ".env"
            env_file.write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "structured SkillPackage"):
                CodexWikiExplorerBackend(
                    env_file=env_file,
                    execution_timeout_seconds=60,
                    execution_contract=CODEX_EXECUTION_CONTRACT.to_dict(),
                    sdk_runtime=_Runtime(events),
                ).explore(
                    query="find a skill",
                    query_wiki_root=wiki,
                    trace_dir=root / "trace",
                )

    def test_unknown_phase_remains_a_legacy_sdk_fallback(self) -> None:
        events = [
            _event(
                "item/completed",
                item=_item("agentMessage", text=json.dumps(_package()), phase=None),
            ),
            _success_events()[-1],
        ]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            wiki = root / "query_wiki"
            wiki.mkdir()
            env_file = root / ".env"
            env_file.write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")

            package = CodexWikiExplorerBackend(
                env_file=env_file,
                execution_timeout_seconds=60,
                execution_contract=CODEX_EXECUTION_CONTRACT.to_dict(),
                sdk_runtime=_Runtime(events),
            ).explore(
                query="find a skill",
                query_wiki_root=wiki,
                trace_dir=root / "trace",
            )

        self.assertEqual(package.to_dict(), _package())

    def test_rejects_contract_drift_and_invalid_output(self) -> None:
        contract = CODEX_EXECUTION_CONTRACT.to_dict()
        contract["web_search"] = True
        with self.assertRaisesRegex(ValueError, "execution_contract"):
            CodexWikiExplorerBackend(execution_contract=contract)

        runtime = _Runtime(_success_events({"not": "a package"}))
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            wiki = root / "query_wiki"
            wiki.mkdir()
            env_file = root / ".env"
            env_file.write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "skill package"):
                CodexWikiExplorerBackend(
                    env_file=env_file,
                    execution_contract=CODEX_EXECUTION_CONTRACT.to_dict(),
                    sdk_runtime=runtime,
                ).explore(query="find a skill", query_wiki_root=wiki, trace_dir=root / "trace")

    def test_rejects_malformed_missing_and_failed_final_responses(self) -> None:
        completed = _success_events()[-1]
        cases = {
            "malformed": (
                [
                    _event(
                        "item/completed",
                        item=_item(
                            "agentMessage",
                            text="{not-json",
                            phase=SimpleNamespace(value="final_answer"),
                        ),
                    ),
                    completed,
                ],
                "valid SkillPackage JSON",
            ),
            "missing": ([completed], "did not return a structured SkillPackage"),
            "failed": (
                [
                    _event(
                        "turn/completed",
                        turn=SimpleNamespace(
                            status=SimpleNamespace(value="failed"),
                            error=SimpleNamespace(message="provider unavailable"),
                            duration_ms=25,
                        ),
                    )
                ],
                "turn failed",
            ),
        }
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            wiki = root / "query_wiki"
            wiki.mkdir()
            env_file = root / ".env"
            env_file.write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")
            for name, (events, message) in cases.items():
                with self.subTest(name=name), self.assertRaisesRegex(Exception, message):
                    CodexWikiExplorerBackend(
                        env_file=env_file,
                        execution_timeout_seconds=60,
                        execution_contract=CODEX_EXECUTION_CONTRACT.to_dict(),
                        sdk_runtime=_Runtime(events),
                    ).explore(
                        query="find a skill",
                        query_wiki_root=wiki,
                        trace_dir=root / name,
                    )

    def test_timeout_interrupts_before_the_app_server_closes(self) -> None:
        runtime = _Runtime(event_delay=10)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            wiki = root / "query_wiki"
            wiki.mkdir()
            env_file = root / ".env"
            env_file.write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")

            with self.assertRaisesRegex(TimeoutError, "exceeded 1 seconds"):
                CodexWikiExplorerBackend(
                    env_file=env_file,
                    execution_timeout_seconds=1,
                    execution_contract=CODEX_EXECUTION_CONTRACT.to_dict(),
                    sdk_runtime=runtime,
                ).explore(query="find a skill", query_wiki_root=wiki, trace_dir=root / "trace")

        self.assertEqual(runtime.lifecycle, ["interrupt", "close"])

    def test_timeout_during_app_server_startup_still_closes_the_client(self) -> None:
        runtime = _Runtime(enter_delay=10)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            wiki = root / "query_wiki"
            wiki.mkdir()
            env_file = root / ".env"
            env_file.write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")

            with self.assertRaisesRegex(TimeoutError, "exceeded 1 seconds"):
                CodexWikiExplorerBackend(
                    env_file=env_file,
                    execution_timeout_seconds=1,
                    execution_contract=CODEX_EXECUTION_CONTRACT.to_dict(),
                    sdk_runtime=runtime,
                ).explore(query="find a skill", query_wiki_root=wiki, trace_dir=root / "trace")

        self.assertEqual(runtime.lifecycle, ["close"])

    def test_close_failure_does_not_replace_the_original_sdk_failure(self) -> None:
        runtime = _Runtime(fail_thread_start=True, fail_close=True)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            wiki = root / "query_wiki"
            wiki.mkdir()
            env_file = root / ".env"
            env_file.write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "app-server failed"):
                CodexWikiExplorerBackend(
                    env_file=env_file,
                    execution_timeout_seconds=60,
                    execution_contract=CODEX_EXECUTION_CONTRACT.to_dict(),
                    sdk_runtime=runtime,
                ).explore(query="find a skill", query_wiki_root=wiki, trace_dir=root / "trace")

            events = [
                json.loads(line)
                for line in (root / "trace" / "cc_explorer" / "agent_events.jsonl")
                .read_text()
                .splitlines()
            ]
            cleanup = next(event for event in events if event["event"] == "sdk:cleanup_error")
            self.assertEqual(cleanup["error_type"], "RuntimeError")

    def test_missing_sdk_is_recorded_as_a_backend_failure(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            wiki = root / "query_wiki"
            wiki.mkdir()
            trace = root / "trace"

            with (
                patch(
                    "skillfabric.wiki.explorer.backends.codex._load_sdk_runtime",
                    side_effect=RuntimeError("openai-codex is required"),
                ),
                self.assertRaisesRegex(RuntimeError, "openai-codex is required"),
            ):
                CodexWikiExplorerBackend().explore(
                    query="find a skill",
                    query_wiki_root=wiki,
                    trace_dir=trace,
                )

            error = json.loads((trace / "cc_explorer" / "error.json").read_text())
            self.assertEqual(error["error_type"], "RuntimeError")
            events = [
                json.loads(line)
                for line in (trace / "cc_explorer" / "agent_events.jsonl").read_text().splitlines()
            ]
            self.assertEqual(events[-1]["event"], "backend:error")


if __name__ == "__main__":
    unittest.main()
