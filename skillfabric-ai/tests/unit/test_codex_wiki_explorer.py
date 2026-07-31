from __future__ import annotations

import asyncio
import fnmatch
import inspect
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

from skillfabric.wiki.explorer.agent import WikiExplorerConfig, explore_query_wiki
from skillfabric.wiki.explorer.backends.codex import (
    CODEX_EXECUTION_CONTRACT,
    CodexOperationalAccessError,
    CodexWikiExplorerBackend,
    _load_sdk_runtime,
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
                command="sed -n '1,80p' index.md",
                status=SimpleNamespace(value="inProgress"),
            ),
        ),
        _event(
            "item/completed",
            item=_item(
                "commandExecution",
                command="sed -n '1,80p' index.md",
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

    async def login_api_key(self, api_key: str) -> None:
        self.runtime.login_calls.append(api_key)
        self.runtime.lifecycle.append("login")

    async def thread_start(self, **kwargs: object) -> _Thread:
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
        fail_close: bool = False,
        fail_thread_start: bool = False,
    ) -> None:
        self.events = events or _success_events()
        self.event_delay = event_delay
        self.enter_delay = enter_delay
        self.fail_close = fail_close
        self.fail_thread_start = fail_thread_start
        self.configs: list[_CodexConfig] = []
        self.login_calls: list[str] = []
        self.thread_calls: list[dict[str, object]] = []
        self.turn_calls: list[tuple[str, dict[str, object]]] = []
        self.turns: list[_Turn] = []
        self.closes = 0
        self.lifecycle: list[str] = []

    def AsyncCodex(self, *, config: _CodexConfig) -> _Codex:
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

    def test_codex_prompt_spec_carries_the_exact_selection_count(self) -> None:
        spec = build_codex_prompt_spec(
            query="extract financial KPIs",
            query_wiki_root=Path("query_wiki"),
            max_selected_skills=5,
            required_selected_skills=5,
        )

        self.assertEqual(spec["required_selected_skills"], 5)
        self.assertIn("Return exactly 5 selected skills", spec["system_prompt"])
        self.assertIn(
            "<required_selected_skills>5</required_selected_skills>",
            spec["user_prompt"],
        )

    def test_codex_prompt_contract_has_no_custom_builder_surface(self) -> None:
        self.assertNotIn(
            "prompt_spec_builder", inspect.signature(build_codex_prompt_spec).parameters
        )
        self.assertNotIn(
            "prompt_spec_builder",
            CodexWikiExplorerBackend.__dataclass_fields__,
        )

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

    def test_completed_empty_requires_index_and_candidate_lookup(self) -> None:
        success = _success_events()
        events = [*success[:2], *success[4:]]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            wiki = root / "query_wiki"
            wiki.mkdir()
            env_file = root / ".env"
            env_file.write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")

            with self.assertRaisesRegex(CodexOperationalAccessError, "candidate lookup"):
                CodexWikiExplorerBackend(
                    env_file=env_file,
                    execution_contract=CODEX_EXECUTION_CONTRACT.to_dict(),
                    sdk_runtime=_Runtime(events),
                ).explore(query="find a skill", query_wiki_root=wiki, trace_dir=root / "trace")

            access = json.loads(
                (root / "trace" / "cc_explorer" / "operational_access.json").read_text()
            )
            self.assertTrue(access["index_read"])
            self.assertFalse(access["candidate_lookup"])
            self.assertFalse(access["semantic_empty_valid"])

    def test_completed_empty_rejects_a_failed_candidate_lookup(self) -> None:
        success = _success_events()
        failed_lookup = _event(
            "item/completed",
            item=_item(
                "commandExecution",
                command="rg -n 'financial' cards",
                status=SimpleNamespace(value="failed"),
                exit_code=1,
            ),
        )
        events = [*success[:3], failed_lookup, *success[4:]]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            wiki = root / "query_wiki"
            wiki.mkdir()
            env_file = root / ".env"
            env_file.write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")

            with self.assertRaisesRegex(CodexOperationalAccessError, "candidate lookup"):
                CodexWikiExplorerBackend(
                    env_file=env_file,
                    execution_contract=CODEX_EXECUTION_CONTRACT.to_dict(),
                    sdk_runtime=_Runtime(events),
                ).explore(query="find a skill", query_wiki_root=wiki, trace_dir=root / "trace")

            access = json.loads(
                (root / "trace" / "cc_explorer" / "operational_access.json").read_text()
            )
            self.assertFalse(access["candidate_lookup"])
            self.assertFalse(access["semantic_empty_valid"])

    def test_completed_turn_without_usage_event_is_unmetered(self) -> None:
        events = [
            event for event in _success_events() if event.method != "thread/tokenUsage/updated"
        ]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            wiki = root / "query_wiki"
            wiki.mkdir()
            env_file = root / ".env"
            env_file.write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "usage closure"):
                CodexWikiExplorerBackend(
                    env_file=env_file,
                    execution_contract=CODEX_EXECUTION_CONTRACT.to_dict(),
                    sdk_runtime=_Runtime(events),
                ).explore(query="find a skill", query_wiki_root=wiki, trace_dir=root / "trace")

            artifacts = root / "trace" / "cc_explorer"
            self.assertTrue((artifacts / "turn_state.json").is_file())
            self.assertFalse((artifacts / "usage.json").exists())

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

    def test_pwd_only_and_failed_wiki_commands_do_not_prove_access(self) -> None:
        success = _success_events()
        cases = {
            "pwd-only": [
                _event(
                    "item/started",
                    item=_item(
                        "commandExecution",
                        command="pwd",
                        status=SimpleNamespace(value="inProgress"),
                    ),
                ),
                _event(
                    "item/completed",
                    item=_item(
                        "commandExecution",
                        command="pwd",
                        status=SimpleNamespace(value="completed"),
                        exit_code=0,
                    ),
                ),
                *success[4:],
            ],
            "all-failed": [
                _event(
                    "item/started",
                    item=_item(
                        "commandExecution",
                        command="sed -n '1,80p' index.md",
                        status=SimpleNamespace(value="inProgress"),
                    ),
                ),
                _event(
                    "item/completed",
                    item=_item(
                        "commandExecution",
                        command="sed -n '1,80p' index.md",
                        status=SimpleNamespace(value="failed"),
                        exit_code=1,
                    ),
                ),
                *success[2:],
            ],
        }
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            wiki = root / "query_wiki"
            wiki.mkdir()
            env_file = root / ".env"
            env_file.write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")
            for name, events in cases.items():
                with self.subTest(name=name), self.assertRaises(CodexOperationalAccessError):
                    CodexWikiExplorerBackend(
                        env_file=env_file,
                        execution_contract=CODEX_EXECUTION_CONTRACT.to_dict(),
                        sdk_runtime=_Runtime(events),
                    ).explore(
                        query="find a skill",
                        query_wiki_root=wiki,
                        trace_dir=root / name,
                    )

    def test_successful_non_evidence_command_does_not_prove_wiki_access(self) -> None:
        payload = {
            "selected_skills": [
                {
                    "skill_id": "skill:test",
                    "role": "candidate",
                    "evidence": [{"path": "notes.md"}],
                }
            ],
            "near_misses": [],
            "coverage_gaps": [],
            "wiki_pages_read": ["notes.md"],
            "rationale": "Selected from an unclassified file.",
        }
        success = _success_events(payload)
        events = [
            _event(
                "item/started",
                item=_item(
                    "commandExecution",
                    command="ls",
                    status=SimpleNamespace(value="inProgress"),
                ),
            ),
            _event(
                "item/completed",
                item=_item(
                    "commandExecution",
                    command="ls",
                    status=SimpleNamespace(value="completed"),
                    exit_code=0,
                ),
            ),
            *success[4:],
        ]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            wiki = root / "query_wiki"
            wiki.mkdir()
            env_file = root / ".env"
            env_file.write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")

            with self.assertRaisesRegex(CodexOperationalAccessError, "evidence access"):
                CodexWikiExplorerBackend(
                    env_file=env_file,
                    execution_contract=CODEX_EXECUTION_CONTRACT.to_dict(),
                    sdk_runtime=_Runtime(events),
                ).explore(query="find a skill", query_wiki_root=wiki, trace_dir=root / "trace")

            access = json.loads(
                (root / "trace" / "cc_explorer" / "operational_access.json").read_text()
            )
            self.assertFalse(access["evidence_access"])
            self.assertEqual(access["evidence_categories"], [])

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

    def test_command_audit_does_not_treat_search_terms_as_shell_activity(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            wiki = root / "query_wiki"
            wiki.mkdir()
            env_file = root / ".env"
            env_file.write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")
            for name, command in {
                "ampersand": "rg -n 'AT&T' cards",
                "comparison": "rg -n 'revenue > cost' cards",
                "network-word": "rg -n curl cards",
                "runtime-word": "rg -n bash cards",
            }.items():
                success = _success_events()
                lookup = [
                    _event(
                        "item/started",
                        item=_item(
                            "commandExecution",
                            command=command,
                            status=SimpleNamespace(value="inProgress"),
                        ),
                    ),
                    _event(
                        "item/completed",
                        item=_item(
                            "commandExecution",
                            command=command,
                            status=SimpleNamespace(value="completed"),
                            exit_code=0,
                        ),
                    ),
                ]
                events = [*success[:2], *lookup, *success[4:]]

                package = CodexWikiExplorerBackend(
                    env_file=env_file,
                    execution_contract=CODEX_EXECUTION_CONTRACT.to_dict(),
                    sdk_runtime=_Runtime(events),
                ).explore(
                    query="find a skill",
                    query_wiki_root=wiki,
                    trace_dir=root / name,
                )

                self.assertEqual(package.to_dict(), _package())
                access = json.loads(
                    (root / name / "cc_explorer" / "operational_access.json").read_text()
                )
                self.assertIsNone(access["policy_violation"])

    def test_command_audit_allows_read_only_app_server_shell_wrapper(self) -> None:
        events = _success_events()
        for event in events[:2]:
            event.payload.item.root.command = "/bin/bash -lc 'sed -n 1,80p index.md'"
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
        self.assertFalse(hasattr(runtime, "Sandbox"))

    def test_backend_is_public_and_codex_sdk_is_an_optional_dependency(self) -> None:
        from skillfabric.wiki.explorer.backends import CodexWikiExplorerBackend as PublicBackend

        package_root = Path(__file__).resolve().parents[2]
        pyproject = tomllib.loads((package_root / "pyproject.toml").read_text(encoding="utf-8"))
        extras = pyproject["project"]["optional-dependencies"]
        codex_requirement = "openai-codex>=0.144.4,<0.145"

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
                    codex_bin=root / "codex",
                    sdk_runtime=runtime,
                ).explore(query="find a skill", query_wiki_root=wiki, trace_dir=trace)

            self.assertEqual(package.to_dict(), _package())
            config = runtime.configs[0].kwargs
            self.assertEqual(config["codex_bin"], str(root / "codex"))
            self.assertEqual(config["env"]["OPENAI_API_KEY"], "sk-experiment-secret")
            self.assertEqual(config["env"]["CODEX_API_KEY"], "")
            self.assertEqual(config["env"]["CODEX_APP_SERVER_DISABLE_MANAGED_CONFIG"], "1")
            self.assertEqual(runtime.login_calls, ["sk-experiment-secret"])
            self.assertLess(
                runtime.lifecycle.index("login"),
                runtime.lifecycle.index("thread_start"),
            )
            codex_home = Path(config["env"]["CODEX_HOME"])
            self.assertFalse(codex_home.exists())
            self.assertEqual(config["cwd"], str(codex_home))
            self.assertEqual(
                config["config_overrides"],
                (
                    "project_root_markers=[]",
                    "check_for_update_on_startup=false",
                    "features.plugins=false",
                    "features.remote_plugin=false",
                    "features.plugin_sharing=false",
                    "features.plugin_hooks=false",
                    "skills.bundled.enabled=false",
                    "orchestrator.skills.enabled=false",
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
                {
                    ":minimal": "read",
                    str(wiki.resolve()): "read",
                    str((root / "codex").resolve()): "read",
                },
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
            self.assertNotIn("sandbox", turn)

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
                    "total_calls": 2,
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

    def test_zero_timeout_waits_for_completion(self) -> None:
        runtime = _Runtime()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            wiki = root / "query_wiki"
            wiki.mkdir()
            env_file = root / ".env"
            env_file.write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")

            package = CodexWikiExplorerBackend(
                env_file=env_file,
                execution_timeout_seconds=0,
                execution_contract=CODEX_EXECUTION_CONTRACT.to_dict(),
                sdk_runtime=runtime,
            ).explore(
                query="find a skill",
                query_wiki_root=wiki,
                trace_dir=root / "trace",
            )

            self.assertEqual(package.to_dict(), _package())
            self.assertEqual(runtime.lifecycle[-1], "close")

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
            self.assertNotIn("credential-secret", error_text)
            self.assertNotIn("token-secret", error_text)
            self.assertNotIn(runtime.configs[0].kwargs["env"]["CODEX_HOME"], error_text)

    def test_outer_attempt_closure_reuses_the_backend_sanitized_error(self) -> None:
        runtime = _Runtime(fail_thread_start=True)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            wiki = root / "query_wiki"
            wiki.mkdir()
            env_file = root / ".env"
            env_file.write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")
            trace = root / "trace"
            backend = CodexWikiExplorerBackend(
                env_file=env_file,
                execution_contract=CODEX_EXECUTION_CONTRACT.to_dict(),
                sdk_runtime=runtime,
            )

            with self.assertRaisesRegex(RuntimeError, "app-server failed"):
                explore_query_wiki(
                    WikiExplorerConfig(
                        env_file=env_file,
                        max_attempts=1,
                        retry_delay_seconds=0,
                    ),
                    query="find a skill",
                    query_wiki_root=wiki,
                    trace_dir=trace,
                    backend=backend,
                )

            codex_home = runtime.configs[0].kwargs["env"]["CODEX_HOME"]
            closure_text = (trace / "cc_explorer" / "closure.json").read_text(encoding="utf-8")
            self.assertNotIn(codex_home, closure_text)
            self.assertNotIn("sk-runtime-secret", closure_text)
            self.assertNotIn("credential-secret", closure_text)
            self.assertNotIn("token-secret", closure_text)

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

    def test_completed_exec_command_with_pty_process_id_is_allowed(self) -> None:
        runtime = _Runtime(
            [
                _event(
                    "item/started",
                    item=_item(
                        "commandExecution",
                        process_id="process-1",
                        status=SimpleNamespace(value="inProgress"),
                    ),
                ),
                _event(
                    "item/completed",
                    item=_item(
                        "commandExecution",
                        process_id="process-1",
                        status=SimpleNamespace(value="completed"),
                    ),
                ),
                *_success_events(),
            ]
        )
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
            ).explore(query="find a skill", query_wiki_root=wiki, trace_dir=root / "trace")

        self.assertEqual(package.to_dict(), _package())
        self.assertEqual(runtime.turns[0].interrupts, 0)

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
            *_success_events()[:4],
            *_success_events()[5:],
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
            *_success_events()[:4],
            *_success_events()[5:],
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
        success = _success_events()
        access_events = [*success[:4], *success[5:-1]]
        cases = {
            "malformed": (
                [
                    *access_events,
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
            "missing": ([*access_events, completed], "did not return a structured SkillPackage"),
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

        self.assertEqual(
            runtime.lifecycle,
            ["login", "thread_start", "interrupt", "close"],
        )

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
