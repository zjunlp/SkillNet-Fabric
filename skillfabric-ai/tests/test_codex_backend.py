from __future__ import annotations

import json
import os
import threading
import time
import tomllib
import unittest
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from skillfabric.wiki.explorer.backends.codex import (
    CODEX_EXECUTION_CONTRACT,
    CodexOperationalAccessError,
    CodexWikiExplorerBackend,
    build_codex_prompt_spec,
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
            "item/started",
            item=_item(
                "commandExecution",
                command="head -n 80 index.md",
                status=SimpleNamespace(value="inProgress"),
            ),
        ),
        _event(
            "item/completed",
            item=_item(
                "commandExecution",
                command="head -n 80 index.md",
                status=SimpleNamespace(value="completed"),
                exit_code=0,
            ),
        ),
        _event(
            "item/started",
            item=_item(
                "commandExecution",
                command="rg -n 'financial' cards",
                status=SimpleNamespace(value="inProgress"),
            ),
        ),
        _event(
            "item/completed",
            item=_item(
                "commandExecution",
                command="rg -n 'financial' cards",
                status=SimpleNamespace(value="completed"),
                exit_code=0,
            ),
        ),
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

    def stream(self):
        for event in self.events:
            if self.runtime.event_delay and self.runtime.stopped.wait(self.runtime.event_delay):
                return
            yield event

    def interrupt(self) -> None:
        self.interrupts += 1
        if self.runtime.interrupt_delay:
            time.sleep(self.runtime.interrupt_delay)
        self.runtime.stopped.set()
        self.runtime.lifecycle.append("interrupt")


class _Thread:
    def __init__(self, runtime: _Runtime) -> None:
        self.runtime = runtime
        self.id = "thread-1"

    def turn(self, prompt: str, **kwargs: object) -> _Turn:
        self.runtime.turn_calls.append((prompt, kwargs))
        turn = _Turn(self.runtime, list(self.runtime.events))
        self.runtime.turns.append(turn)
        return turn


class _Codex:
    def __init__(self, runtime: _Runtime, config: _CodexConfig) -> None:
        self.runtime = runtime
        self.config = config
        self.closed = False
        self.metadata = SimpleNamespace(
            serverInfo=SimpleNamespace(name="codex-app-server", version="0.144.4"),
            userAgent="codex/0.144.4",
        )

    def __enter__(self) -> _Codex:
        self.runtime.configs.append(self.config)
        if self.runtime.enter_delay and self.runtime.stopped.wait(self.runtime.enter_delay):
            raise RuntimeError("app-server startup interrupted")
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.runtime.closes += 1
        self.runtime.stopped.set()
        self.runtime.lifecycle.append("close")
        if self.runtime.fail_close:
            raise RuntimeError("app-server close failed")

    def login_api_key(self, api_key: str) -> None:
        self.runtime.login_calls.append(api_key)
        self.runtime.lifecycle.append("login")

    def thread_start(self, **kwargs: object) -> _Thread:
        self.runtime.lifecycle.append("thread_start")
        self.runtime.thread_calls.append(kwargs)
        if self.runtime.fail_thread_start:
            home = self.config.kwargs["env"]["CODEX_HOME"]
            raise RuntimeError(
                f"app-server failed under {home}: OPENAI_API_KEY=sk-runtime-secret "
                "DB_CREDENTIAL:credential-secret SESSION_TOKEN token-secret"
            )
        return _Thread(self.runtime)


class _Runtime:
    CodexConfig = _CodexConfig
    ApprovalMode = SimpleNamespace(deny_all="deny-all")
    Sandbox = SimpleNamespace(read_only="read-only")
    ReasoningEffort = _ReasoningEffort
    __version__ = "0.0.0-test"

    def __init__(
        self,
        events: list[SimpleNamespace] | None = None,
        *,
        event_delay: float = 0,
        enter_delay: float = 0,
        interrupt_delay: float = 0,
        fail_close: bool = False,
        fail_thread_start: bool = False,
    ) -> None:
        self.events = events or _success_events()
        self.event_delay = event_delay
        self.enter_delay = enter_delay
        self.interrupt_delay = interrupt_delay
        self.fail_close = fail_close
        self.fail_thread_start = fail_thread_start
        self.configs: list[_CodexConfig] = []
        self.login_calls: list[str] = []
        self.thread_calls: list[dict[str, object]] = []
        self.turn_calls: list[tuple[str, dict[str, object]]] = []
        self.turns: list[_Turn] = []
        self.closes = 0
        self.lifecycle: list[str] = []
        self.stopped = threading.Event()

    def Codex(self, *, config: _CodexConfig) -> _Codex:
        return _Codex(self, config)


class CodexWikiExplorerTests(unittest.TestCase):
    def test_codex_prompt_spec_contains_the_real_exec_contract(self) -> None:
        spec = build_codex_prompt_spec(
            query="extract financial KPIs",
            query_wiki_root=Path("query_wiki"),
            max_selected_skills=5,
        )

        self.assertEqual(spec["allowed_tools"], ["exec_command"])
        self.assertEqual(spec["tool_budget"]["exec_command"], 21)
        self.assertEqual(spec["schema"], skill_package_json_schema())
        self.assertIn("index.md", spec["system_prompt"])
        self.assertIn("non-interactive", spec["system_prompt"])
        self.assertIn("extract financial KPIs", spec["user_prompt"])
        self.assertNotIn("query", spec)
        self.assertNotIn("required_selected_skills", spec)

    def test_completed_empty_package_without_successful_wiki_access_fails_closed(self) -> None:
        success = _success_events()
        runtime = _Runtime([*success[4:]])
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            wiki = root / "query_wiki"
            wiki.mkdir()
            env_file = root / ".env"
            env_file.write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")

            with self.assertRaises(CodexOperationalAccessError):
                CodexWikiExplorerBackend(
                    env_file=env_file,
                    execution_contract=CODEX_EXECUTION_CONTRACT.to_dict(),
                    sdk_runtime=runtime,
                ).explore(query="find a skill", query_wiki_root=wiki, trace_dir=root / "trace")

            access = json.loads(
                (root / "trace" / "cc_explorer" / "operational_access.json").read_text()
            )
            self.assertFalse(access["evidence_access"])

    def test_invalid_skill_package_payload_is_retryable(self) -> None:
        invalid_payload = _package()
        del invalid_payload["rationale"]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            wiki = root / "query_wiki"
            wiki.mkdir()
            env_file = root / ".env"
            env_file.write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")

            with self.assertRaises(ValueError) as raised:
                CodexWikiExplorerBackend(
                    env_file=env_file,
                    execution_contract=CODEX_EXECUTION_CONTRACT.to_dict(),
                    sdk_runtime=_Runtime(_success_events(invalid_payload)),
                ).explore(query="find a skill", query_wiki_root=wiki, trace_dir=root / "trace")

            self.assertTrue(
                getattr(raised.exception, "__skillfabric_recoverable_route_failure__", False)
            )

    def test_command_audit_fails_closed_on_write_network_and_outside_access(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            wiki = root / "query_wiki"
            wiki.mkdir()
            env_file = root / ".env"
            env_file.write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")
            cases = {
                "write": ("touch cards/new.md", None),
                "network": ("curl https://example.invalid", None),
                "wrapped-write": ("/bin/bash -lc 'touch cards/new.md'", None),
                "wrapped-network": ("/bin/bash -lc 'curl https://example.invalid'", None),
                "wrapped-runtime": ("/bin/bash -lc 'python3 -c \"print(1)\"'", None),
                "outside": (
                    "sed -n '1,20p' ../outside.md",
                    [
                        SimpleNamespace(
                            root=SimpleNamespace(type="read", path=str(root / "outside.md"))
                        )
                    ],
                ),
                "outside-without-actions": ("cat ../outside.md", None),
                "absolute-without-actions": ("cat /etc/passwd", None),
                "relative-executable": ("./cat index.md", None),
                "outside-executable": ("/tmp/cat index.md", None),
                "environment-expansion": ("cat $HOME/.codex/auth.json", None),
                "command-substitution": ("cat $(pwd)/index.md", None),
                "sed-in-place": ("sed -i 's/a/b/' index.md", None),
                "sed-execute": ("sed -n 'e cat /etc/passwd' index.md", None),
                "sed-read": ("sed -n 'r /etc/passwd' index.md", None),
                "find-delete": ("find cards -delete", None),
                "find-exec": ("find cards -exec cat {} ;", None),
                "find-fprint0": ("find cards -fprint0 paths.txt", None),
                "grep-pattern-file": ("grep -f/etc/passwd financial index.md", None),
                "ripgrep-preprocessor": ("rg --pre cat financial cards", None),
                "ripgrep-pattern-file": ("rg -f/etc/passwd financial index.md", None),
                "ripgrep-search-zip": ("rg --search-zip financial .", None),
                "ripgrep-hostname-bin": ("rg --hostname-bin cat financial .", None),
                "ripgrep-hostname-bin-equals": ("rg --hostname-bin=cat financial .", None),
                "sort-output": ("sort -o sorted.md index.md", None),
                "sort-short-output": ("sort -osorted.md index.md", None),
                "uniq-output": ("uniq index.md unique.md", None),
                "wc-files0-from": ("wc --files0-from paths.list", None),
                "wc-files0-from-equals": ("wc --files0-from=paths.list", None),
            }
            for name, (command, actions) in cases.items():
                command_fields = {
                    "command": command,
                    "command_actions": actions or [],
                    "status": SimpleNamespace(value="inProgress"),
                }
                runtime = _Runtime(
                    [
                        _event(
                            "item/started",
                            item=_item("commandExecution", **command_fields),
                        ),
                        *_success_events(),
                    ]
                )
                with (
                    self.subTest(name=name),
                    self.assertRaisesRegex(
                        RuntimeError, "disallowed Codex tool activity"
                    ) as raised,
                ):
                    CodexWikiExplorerBackend(
                        env_file=env_file,
                        execution_contract=CODEX_EXECUTION_CONTRACT.to_dict(),
                        sdk_runtime=runtime,
                    ).explore(
                        query="find a skill",
                        query_wiki_root=wiki,
                        trace_dir=root / name,
                    )
                self.assertTrue(getattr(raised.exception, "__skillfabric_non_retryable__", False))

    def test_command_audit_allows_read_only_app_server_shell_wrapper(self) -> None:
        events = _success_events()
        for event in events[:2]:
            event.payload.item.root.command = "/bin/bash -lc 'head -n 80 index.md'"
            event.payload.item.root.command_actions = [
                SimpleNamespace(root=SimpleNamespace(type="listFiles", path="index.md"))
            ]

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            wiki = root / "query_wiki"
            wiki.mkdir()
            env_file = root / ".env"
            env_file.write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")

            package = CodexWikiExplorerBackend(
                env_file=env_file,
                execution_contract=CODEX_EXECUTION_CONTRACT.to_dict(),
                sdk_runtime=_Runtime(events),
            ).explore(
                query="find a skill",
                query_wiki_root=wiki,
                trace_dir=root / "trace",
            )

            self.assertEqual(package.to_dict(), _package())
            access = json.loads(
                (root / "trace" / "cc_explorer" / "operational_access.json").read_text()
            )
            self.assertIsNone(access["policy_violation"])
            self.assertTrue(access["index_read"])

    def test_backend_is_public_and_codex_sdk_is_an_optional_dependency(self) -> None:
        from skillfabric.wiki.explorer.backends import CodexWikiExplorerBackend as PublicBackend

        package_root = Path(__file__).resolve().parents[1]
        pyproject = tomllib.loads((package_root / "pyproject.toml").read_text(encoding="utf-8"))
        extras = pyproject["project"]["optional-dependencies"]
        codex_requirement = "openai-codex>=0.144.4"

        self.assertIs(PublicBackend, CodexWikiExplorerBackend)
        self.assertEqual(extras["codex"], [codex_requirement])
        self.assertIn(codex_requirement, extras["all"])

    def test_backend_rejects_a_symlinked_query_wiki_root_before_startup(self) -> None:
        runtime = _Runtime()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_wiki = root / "real-query-wiki"
            real_wiki.mkdir()
            wiki_link = root / "query_wiki"
            wiki_link.symlink_to(real_wiki, target_is_directory=True)
            env_file = root / ".env"
            env_file.write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "symlink"):
                CodexWikiExplorerBackend(
                    env_file=env_file,
                    execution_contract=CODEX_EXECUTION_CONTRACT.to_dict(),
                    sdk_runtime=runtime,
                ).explore(
                    query="find a skill",
                    query_wiki_root=wiki_link,
                    trace_dir=root / "trace",
                )

            self.assertEqual(runtime.configs, [])

    def test_backend_rejects_symlinks_inside_the_query_wiki(self) -> None:
        runtime = _Runtime()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            wiki = root / "query_wiki"
            wiki.mkdir()
            outside = root / "outside.md"
            outside.write_text("private", encoding="utf-8")
            (wiki / "outside.md").symlink_to(outside)
            env_file = root / ".env"
            env_file.write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "symlink"):
                CodexWikiExplorerBackend(
                    env_file=env_file,
                    execution_contract=CODEX_EXECUTION_CONTRACT.to_dict(),
                    sdk_runtime=runtime,
                ).explore(
                    query="find a skill",
                    query_wiki_root=wiki,
                    trace_dir=root / "trace",
                )

            self.assertEqual(runtime.configs, [])

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
                "SKILLFABRIC_LLM_API_KEY=sk-project-test\n",
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
                    codex_bin=root / "codex",
                    sdk_runtime=runtime,
                ).explore(query="find a skill", query_wiki_root=wiki, trace_dir=trace)

            self.assertEqual(package.to_dict(), _package())
            config = runtime.configs[0].kwargs
            self.assertEqual(config["codex_bin"], str(root / "codex"))
            self.assertEqual(config["env"]["OPENAI_API_KEY"], "sk-project-test")
            self.assertEqual(config["env"]["CODEX_API_KEY"], "")
            self.assertNotIn("CODEX_APP_SERVER_DISABLE_MANAGED_CONFIG", config["env"])
            self.assertEqual(runtime.login_calls, ["sk-project-test"])
            self.assertLess(
                runtime.lifecycle.index("login"),
                runtime.lifecycle.index("thread_start"),
            )
            codex_home = Path(config["env"]["CODEX_HOME"])
            self.assertFalse(codex_home.exists())
            self.assertEqual(config["cwd"], str(codex_home))
            self.assertEqual(
                config["config_overrides"],
                ("check_for_update_on_startup=false",),
            )

            thread = runtime.thread_calls[0]
            self.assertEqual(thread["approval_mode"], "deny-all")
            self.assertEqual(thread["cwd"], str(wiki.resolve()))
            self.assertEqual(thread["model"], "gpt-5.6-terror")
            self.assertTrue(thread["ephemeral"])
            self.assertEqual(thread["sandbox"], "read-only")
            self.assertIn("index.md", thread["developer_instructions"])
            self.assertNotIn("base_instructions", thread)
            thread_config = thread["config"]
            self.assertEqual(
                thread_config,
                {
                    "openai_base_url": "http://gateway.example/v1",
                    "web_search": "disabled",
                    "project_root_markers": [],
                    "project_doc_max_bytes": 0,
                    "project_doc_fallback_filenames": [],
                    "allow_login_shell": False,
                    "features": {
                        "apps": False,
                        "multi_agent": False,
                        "network_proxy": False,
                        "plugins": False,
                        "remote_plugin": False,
                        "skill_mcp_dependency_install": False,
                    },
                    "mcp_servers": {},
                    "shell_environment_policy": {
                        "inherit": "core",
                        "ignore_default_excludes": False,
                        "set": {"HOME": str(codex_home)},
                    },
                },
            )
            shell_policy = thread_config["shell_environment_policy"]
            self.assertEqual(shell_policy["inherit"], "core")
            self.assertFalse(shell_policy["ignore_default_excludes"])
            self.assertEqual(shell_policy["set"], {"HOME": str(codex_home)})
            self.assertEqual(thread_config["project_doc_max_bytes"], 0)
            self.assertEqual(thread_config["project_doc_fallback_filenames"], [])

            prompt, turn = runtime.turn_calls[0]
            self.assertIn("find a skill", prompt)
            self.assertEqual(turn["effort"], _ReasoningEffort.medium)
            self.assertEqual(turn["model"], "gpt-5.6-terror")
            self.assertEqual(turn["output_schema"], skill_package_json_schema())
            self.assertEqual(turn["cwd"], str(wiki.resolve()))
            self.assertEqual(turn["sandbox"], "read-only")

            prompt_context = json.loads(
                (trace / "cc_explorer" / "prompt_context.json").read_text(encoding="utf-8")
            )
            self.assertEqual(prompt_context["allowed_tools"], ["exec_command"])
            self.assertIn(
                "exec_command<=21", (trace / "cc_explorer" / "prompt.system.md").read_text()
            )

            artifacts = trace / "cc_explorer"
            backend = json.loads((artifacts / "backend.json").read_text(encoding="utf-8"))
            self.assertEqual(backend["backend"], "codex")
            self.assertEqual(backend["allowed_tools"], ["exec_command"])
            self.assertEqual(backend["tool_enforcement"], "event-audited-fail-closed")
            self.assertEqual(backend["command_budget"], 21)
            self.assertEqual(backend["execution_contract"], CODEX_EXECUTION_CONTRACT.to_dict())
            self.assertFalse((artifacts / "usage.json").exists())
            combined = "\n".join(
                path.read_text(encoding="utf-8") for path in artifacts.iterdir() if path.is_file()
            )
            self.assertNotIn("sk-project-test", combined)
            self.assertNotIn("personal-token", combined)
            self.assertEqual(runtime.closes, 1)

            events = [
                json.loads(line)
                for line in (artifacts / "agent_events.jsonl").read_text().splitlines()
            ]
            completed = next(event for event in events if event.get("method") == "turn/completed")
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["duration_ms"], 25)

    def test_success_does_not_require_a_token_usage_event(self) -> None:
        events = [
            event for event in _success_events() if event.method != "thread/tokenUsage/updated"
        ]
        runtime = _Runtime(events)
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
                sdk_runtime=runtime,
            ).explore(
                query="find a skill",
                query_wiki_root=wiki,
                trace_dir=root / "trace",
            )

            self.assertEqual(package.to_dict(), _package())
            self.assertFalse((root / "trace" / "cc_explorer" / "usage.json").exists())

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
            self.assertNotIn("credential-secret", error_text)
            self.assertNotIn("token-secret", error_text)
            self.assertNotIn(runtime.configs[0].kwargs["env"]["CODEX_HOME"], error_text)

    def test_command_budget_interrupts_the_turn_and_propagates_failure(self) -> None:
        commands = [
            _event(
                "item/started",
                item=_item("commandExecution", command=f"head -n {index + 1} index.md"),
            )
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
                backend.explore(
                    query="find a skill", query_wiki_root=wiki, trace_dir=root / "trace"
                )

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
            "unknown-item": [_event("item/started", item=_item(""))],
        }
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            wiki = root / "query_wiki"
            wiki.mkdir()
            env_file = root / ".env"
            env_file.write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")

            for name, tool_events in cases.items():
                runtime = _Runtime([*tool_events, *_success_events()[:-1], completed])
                with (
                    self.subTest(name=name),
                    self.assertRaisesRegex(
                        RuntimeError,
                        "disallowed Codex tool activity",
                    ),
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

        self.assertEqual(
            runtime.lifecycle,
            ["login", "thread_start", "interrupt", "close"],
        )

    def test_timeout_closes_transport_without_waiting_for_interrupt_rpc(self) -> None:
        runtime = _Runtime(event_delay=10, interrupt_delay=1)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            wiki = root / "query_wiki"
            wiki.mkdir()
            env_file = root / ".env"
            env_file.write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")

            started = time.monotonic()
            with self.assertRaisesRegex(TimeoutError, "exceeded 0.05 seconds"):
                CodexWikiExplorerBackend(
                    env_file=env_file,
                    execution_timeout_seconds=0.05,
                    execution_contract=CODEX_EXECUTION_CONTRACT.to_dict(),
                    sdk_runtime=runtime,
                ).explore(query="find a skill", query_wiki_root=wiki, trace_dir=root / "trace")
            elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.5)
        self.assertEqual(runtime.closes, 1)
        self.assertEqual(runtime.turns[0].interrupts, 1)

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
