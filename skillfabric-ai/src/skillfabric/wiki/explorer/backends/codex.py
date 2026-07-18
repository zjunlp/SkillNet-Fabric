"""OpenAI Codex SDK backend for strict query-wiki exploration."""

from __future__ import annotations

import asyncio
import json
import math
import re
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from skillfabric.runtime.sdk_env import CodexSdkEnvironment, build_codex_sdk_env
from skillfabric.storage import atomic_write_text
from skillfabric.wiki.explorer.prompting import (
    EXPLORER_PROMPT_ID,
    ExplorerPromptContext,
    default_tool_budget,
    render_system_prompt,
    render_user_prompt,
)
from skillfabric.wiki.explorer.skill_package import SkillPackage, skill_package_json_schema

CodexSdkRuntime = Any
CODEX_ALLOWED_TOOLS = ("exec_command",)
CODEX_ALLOWED_ITEM_TYPES = frozenset(
    {
        "agentMessage",
        "commandExecution",
        "contextCompaction",
        "reasoning",
        "userMessage",
    }
)
CODEX_TERMINAL_INTERACTION_EVENT = "item/commandExecution/terminalInteraction"
PERMISSION_PROFILE = "skillfabric-query-wiki"


@dataclass(frozen=True, slots=True)
class CodexExecutionContract:
    """Capabilities enforced by the Codex query-wiki backend."""

    schema_version: int = 1
    approval_policy: str = "never"
    filesystem_scope: str = "query-wiki-root"
    sandbox: str = "read-only"
    network_access: bool = False
    web_search: bool = False
    external_tools: bool = False
    system_skills: bool = False
    user_skills: bool = False
    plugins: bool = False
    fresh_session_per_attempt: bool = True
    model_configurable: bool = True
    reasoning_effort_configurable: bool = True
    timeout_configurable: bool = True
    max_attempts_configurable: bool = True
    recovery_owner: str = "skillfabric-explorer"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CODEX_EXECUTION_CONTRACT = CodexExecutionContract()


@dataclass(slots=True)
class CodexWikiExplorerBackend:
    env_file: str | Path = ".env"
    max_selected_skills: int = 8
    model: str | None = None
    reasoning_effort: str = "medium"
    execution_timeout_seconds: float = 300.0
    execution_contract: dict[str, Any] | None = None
    sdk_runtime: CodexSdkRuntime | None = None
    tool_budget: dict[str, int] | None = None

    CODEX_EXECUTION_CONTRACT = CODEX_EXECUTION_CONTRACT

    def __post_init__(self) -> None:
        _require_int_at_least(
            self.max_selected_skills,
            name="max_selected_skills",
            minimum=0,
        )
        if self.model is not None and (not isinstance(self.model, str) or not self.model.strip()):
            raise ValueError("model must be a non-empty string when provided")
        if (
            not isinstance(self.reasoning_effort, str)
            or not self.reasoning_effort
            or self.reasoning_effort != self.reasoning_effort.strip()
        ):
            raise ValueError("reasoning_effort must be a non-empty trimmed string")
        timeout = self.execution_timeout_seconds
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout < 1.0
        ):
            raise ValueError("execution_timeout_seconds must be finite and at least 1")
        supplied_contract = (
            CODEX_EXECUTION_CONTRACT.to_dict()
            if self.execution_contract is None
            else self.execution_contract
        )
        if not _same_json_shape(supplied_contract, CODEX_EXECUTION_CONTRACT.to_dict()):
            raise ValueError("execution_contract does not match the Codex backend contract")
        self.execution_contract = CODEX_EXECUTION_CONTRACT.to_dict()
        self.tool_budget = _normalize_tool_budget(
            self.tool_budget,
            max_selected_skills=self.max_selected_skills,
        )

    def explore(
        self,
        *,
        query: str,
        query_wiki_root: Path,
        trace_dir: Path,
    ) -> SkillPackage:
        """Return one schema-valid SkillPackage or propagate the failure."""

        query_wiki_root = query_wiki_root.resolve()
        if not query_wiki_root.is_dir():
            raise FileNotFoundError(f"query_wiki root does not exist: {query_wiki_root}")
        cc_dir = trace_dir / "cc_explorer"
        cc_dir.mkdir(parents=True, exist_ok=True)
        tool_budget = dict(self.tool_budget or {})
        context = ExplorerPromptContext(
            query=query,
            query_wiki_root=query_wiki_root,
            max_selected_skills=self.max_selected_skills,
            allowed_tools=CODEX_ALLOWED_TOOLS,
            tool_budget=tool_budget,
        )
        system_prompt = render_system_prompt(context)
        user_prompt = render_user_prompt(context)
        atomic_write_text(cc_dir / "prompt.system.md", system_prompt)
        atomic_write_text(cc_dir / "prompt.user.md", user_prompt)
        atomic_write_text(
            cc_dir / "prompt_contract.json",
            json.dumps({"prompt_id": EXPLORER_PROMPT_ID}, indent=2) + "\n",
        )
        atomic_write_text(
            cc_dir / "prompt_context.json",
            json.dumps(context.to_trace_context(), ensure_ascii=False, indent=2) + "\n",
        )
        _write_json(cc_dir / "backend.json", self._backend_payload(runtime=None, metadata=None))
        _write_event(
            cc_dir,
            {
                "event": "backend:start",
                "backend": "codex",
                "allowed_tools": list(CODEX_ALLOWED_TOOLS),
                "command_budget": _command_budget(tool_budget),
                "prompt_id": EXPLORER_PROMPT_ID,
            },
        )
        codex_home: Path | None = None
        try:
            runtime = self.sdk_runtime or _load_sdk_runtime()
            _write_json(
                cc_dir / "backend.json",
                self._backend_payload(runtime=runtime, metadata=None),
            )
            with TemporaryDirectory(prefix="skillfabric-codex-") as home:
                codex_home = Path(home)
                settings = build_codex_sdk_env(self.env_file, codex_home=home)
                payload, usage, metadata = _run_codex_sync(
                    lambda: self._explore_async(
                        runtime=runtime,
                        settings=settings,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        query_wiki_root=query_wiki_root,
                        codex_home=Path(home),
                        cc_dir=cc_dir,
                        command_budget=_command_budget(tool_budget),
                    ),
                    timeout_seconds=self.execution_timeout_seconds,
                )
            _write_json(cc_dir / "backend.json", self._backend_payload(runtime, metadata))
            _write_json(cc_dir / "usage.json", usage)
            package = SkillPackage.from_dict(payload)
            _write_json(cc_dir / "skill_package.json", package.to_dict())
            _write_event(
                cc_dir,
                {"event": "backend:finish", "selected_count": len(package.selected_skills)},
            )
            return package
        except Exception as exc:
            error = {
                "error_type": type(exc).__name__,
                "error": _safe_error_text(
                    str(exc),
                    paths=(() if codex_home is None else (codex_home,)),
                ),
            }
            _write_json(cc_dir / "error.json", error)
            _write_event(cc_dir, {"event": "backend:error", **error})
            raise

    async def _explore_async(
        self,
        *,
        runtime: CodexSdkRuntime,
        settings: CodexSdkEnvironment,
        system_prompt: str,
        user_prompt: str,
        query_wiki_root: Path,
        codex_home: Path,
        cc_dir: Path,
        command_budget: int,
    ) -> tuple[dict[str, Any], dict[str, int], dict[str, str]]:
        config = runtime.CodexConfig(
            cwd=str(codex_home),
            env=dict(settings.env),
            config_overrides=(
                "project_root_markers=[]",
                "check_for_update_on_startup=false",
            ),
        )
        codex = runtime.AsyncCodex(config=config)
        turn = None
        operation_succeeded = False
        try:
            async with asyncio.timeout(self.execution_timeout_seconds):
                await codex.__aenter__()
                metadata = _sdk_metadata(codex.metadata)
                _write_json(
                    cc_dir / "backend.json",
                    self._backend_payload(runtime=runtime, metadata=metadata),
                )
                thread = await codex.thread_start(
                    approval_mode=runtime.ApprovalMode.deny_all,
                    base_instructions=system_prompt,
                    config=_thread_config(
                        query_wiki_root=query_wiki_root,
                        codex_home=codex_home,
                        api_base=settings.api_base,
                    ),
                    cwd=str(query_wiki_root),
                    ephemeral=True,
                    model=self.model,
                )
                turn = await thread.turn(
                    user_prompt,
                    cwd=str(query_wiki_root),
                    effort=runtime.ReasoningEffort(self.reasoning_effort),
                    model=self.model,
                    output_schema=skill_package_json_schema(),
                )
                final_response, usage = await _collect_turn(
                    turn,
                    cc_dir=cc_dir,
                    command_budget=command_budget,
                )
                result = _strict_json_object(final_response), usage, metadata
                operation_succeeded = True
                return result
        except TimeoutError as exc:
            if turn is not None:
                await _best_effort_interrupt(turn)
            _write_event(
                cc_dir,
                {"event": "sdk:timeout", "timeout_seconds": self.execution_timeout_seconds},
            )
            raise TimeoutError(
                f"Codex query-wiki explorer exceeded {self.execution_timeout_seconds:g} seconds"
            ) from exc
        finally:
            try:
                await codex.close()
            except Exception as exc:
                _write_event(
                    cc_dir,
                    {
                        "event": "sdk:cleanup_error",
                        "error_type": type(exc).__name__,
                        "error": _safe_error_text(str(exc), paths=(codex_home,)),
                    },
                )
                if operation_succeeded:
                    raise

    def _backend_payload(
        self,
        runtime: CodexSdkRuntime | None,
        metadata: dict[str, str] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "backend": "codex",
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "sdk_version": str(getattr(runtime, "__version__", "unavailable")),
            "execution_contract": CODEX_EXECUTION_CONTRACT.to_dict(),
            "permission_profile": PERMISSION_PROFILE,
            "allowed_tools": list(CODEX_ALLOWED_TOOLS),
            "tool_enforcement": "event-audited-fail-closed",
            "command_budget": _command_budget(dict(self.tool_budget or {})),
        }
        if metadata:
            payload["app_server"] = metadata
        return payload


async def _collect_turn(
    turn: Any,
    *,
    cc_dir: Path,
    command_budget: int,
) -> tuple[str, dict[str, int]]:
    command_count = 0
    budget_exceeded = False
    policy_violation = ""
    final_response = ""
    fallback_response = ""
    usage = None
    completed_turn = None
    async for event in turn.stream():
        method = str(getattr(event, "method", ""))
        payload = getattr(event, "payload", None)
        item = _root_item(getattr(payload, "item", None))
        item_type = str(getattr(item, "type", "") or "")
        observed_violation = ""
        if method == CODEX_TERMINAL_INTERACTION_EVENT:
            observed_violation = "write_stdin"
        elif (
            method in {"item/started", "item/completed"}
            and item_type
            and item_type not in CODEX_ALLOWED_ITEM_TYPES
        ):
            observed_violation = item_type
        elif (
            method == "item/completed"
            and item_type == "commandExecution"
            and getattr(item, "process_id", None) is not None
        ):
            observed_violation = "background exec_command session"
        if observed_violation and not policy_violation:
            policy_violation = observed_violation
            _write_event(
                cc_dir,
                {"event": "sdk:policy_violation", "activity": policy_violation},
            )
            await _best_effort_interrupt(turn)
        if method == "item/started" and item_type == "commandExecution":
            command_count += 1
            if command_count > command_budget and not budget_exceeded:
                budget_exceeded = True
                await _best_effort_interrupt(turn)
        elif method == "item/completed" and getattr(item, "type", "") == "agentMessage":
            text = str(getattr(item, "text", "") or "")
            phase = _enum_value(getattr(item, "phase", None))
            if phase == "final_answer":
                final_response = text
            elif not phase and text:
                fallback_response = text
        elif method == "thread/tokenUsage/updated":
            usage = getattr(payload, "token_usage", None)
        elif method == "turn/completed":
            completed_turn = getattr(payload, "turn", None)
        _write_event(cc_dir, _event_summary(method, payload, item, command_count))

    if budget_exceeded:
        raise RuntimeError(f"Codex query-wiki tool budget exceeded: exec_command<={command_budget}")
    if policy_violation:
        raise RuntimeError(
            f"Codex query-wiki observed disallowed Codex tool activity: {policy_violation}"
        )
    if completed_turn is None:
        raise RuntimeError("Codex agent finished without a turn/completed event")
    status = _enum_value(getattr(completed_turn, "status", None))
    if status != "completed":
        error = getattr(completed_turn, "error", None)
        detail = str(getattr(error, "message", "") or status or "unknown status")
        raise RuntimeError(f"Codex agent turn failed: {_safe_error_text(detail)}")
    response = final_response or fallback_response
    if not response.strip():
        raise RuntimeError("Codex agent did not return a structured SkillPackage")
    return response, _sdk_usage(
        usage,
        completed_turn=completed_turn,
        command_count=command_count,
    )


def _thread_config(
    *,
    query_wiki_root: Path,
    codex_home: Path,
    api_base: str,
) -> dict[str, Any]:
    return {
        "openai_base_url": api_base,
        "default_permissions": PERMISSION_PROFILE,
        "permissions": {
            PERMISSION_PROFILE: {
                "filesystem": {
                    ":minimal": "read",
                    str(query_wiki_root): "read",
                },
                "network": {"enabled": False},
            }
        },
        "web_search": "disabled",
        "include_permissions_instructions": False,
        "include_apps_instructions": False,
        "include_collaboration_mode_instructions": False,
        "include_environment_context": False,
        "project_root_markers": [],
        "project_doc_max_bytes": 0,
        "project_doc_fallback_filenames": [],
        "allow_login_shell": False,
        "check_for_update_on_startup": False,
        "features": {
            "apps": False,
            "browser_use": False,
            "browser_use_external": False,
            "browser_use_full_cdp_access": False,
            "code_mode": False,
            "code_mode_host": False,
            "code_mode_only": False,
            "collab": False,
            "collaboration_modes": False,
            "computer_use": False,
            "connectors": False,
            "enable_mcp_apps": False,
            "external_agent_memory_import": False,
            "image_generation": False,
            "imagegenext": False,
            "memory_tool": False,
            "multi_agent": False,
            "multi_agent_mode": False,
            "multi_agent_v2": False,
            "network_proxy": False,
            "plugin_hooks": False,
            "plugin_sharing": False,
            "plugins": False,
            "remote_plugin": False,
            "request_permissions": False,
            "request_permissions_tool": False,
            "request_rule": False,
            "search_tool": False,
            "skill_mcp_dependency_install": False,
            "skill_search": False,
            "standalone_web_search": False,
            "tool_search": False,
            "tool_suggest": False,
            "web_search": False,
            "web_search_cached": False,
            "web_search_request": False,
        },
        "skills": {
            "bundled": {"enabled": False},
            "include_instructions": False,
            "config": [],
        },
        "orchestrator": {
            "skills": {"enabled": False},
            "mcp": {"enabled": False},
        },
        "mcp_servers": {},
        "shell_environment_policy": {
            "inherit": "core",
            "ignore_default_excludes": False,
            "exclude": ["*PASSWORD*", "*CREDENTIAL*"],
            "set": {"HOME": str(codex_home)},
            "include_only": ["PATH", "SHELL", "HOME", "LANG", "LC_*"],
        },
    }


def _normalize_tool_budget(
    tool_budget: dict[str, int] | None,
    *,
    max_selected_skills: int,
) -> dict[str, int]:
    default_total = default_tool_budget(max_selected_skills)["total"]
    merged = {"exec_command": default_total, "total": default_total}
    if tool_budget is None:
        return merged
    unexpected = set(tool_budget) - {"exec_command", "total"}
    if unexpected:
        raise ValueError(f"tool_budget has unsupported keys: {', '.join(sorted(unexpected))}")
    for key, value in tool_budget.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"tool_budget.{key} must be a non-negative integer")
        merged[key] = value
    return merged


def _command_budget(tool_budget: dict[str, int]) -> int:
    return min(tool_budget.get("exec_command", 0), tool_budget.get("total", 0))


def _sdk_usage(
    usage: Any,
    *,
    completed_turn: Any,
    command_count: int,
) -> dict[str, int]:
    total = getattr(usage, "total", None)
    return {
        "duration_ms": _nonnegative_int(getattr(completed_turn, "duration_ms", 0)),
        "input_tokens": _nonnegative_int(getattr(total, "input_tokens", 0)),
        "cache_read_input_tokens": _nonnegative_int(
            getattr(total, "cached_input_tokens", 0)
        ),
        "output_tokens": _nonnegative_int(getattr(total, "output_tokens", 0)),
        "reasoning_output_tokens": _nonnegative_int(
            getattr(total, "reasoning_output_tokens", 0)
        ),
        "total_tokens": _nonnegative_int(getattr(total, "total_tokens", 0)),
        "total_calls": command_count,
        "num_turns": 1,
    }


def _strict_json_object(value: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"Codex response contains duplicate JSON key: {key}")
            result[key] = item
        return result

    try:
        payload = json.loads(value, object_pairs_hook=object_pairs)
    except json.JSONDecodeError as exc:
        raise ValueError("Codex agent did not return valid SkillPackage JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Codex agent SkillPackage response must be a JSON object")
    return payload


def _sdk_metadata(metadata: Any) -> dict[str, str]:
    server = getattr(metadata, "serverInfo", None)
    result = {
        "name": str(getattr(server, "name", "") or "unknown"),
        "version": str(getattr(server, "version", "") or "unknown"),
    }
    user_agent = str(getattr(metadata, "userAgent", "") or "")
    if user_agent:
        result["user_agent"] = user_agent
    return result


def _event_summary(
    method: str,
    payload: Any,
    item: Any,
    command_count: int,
) -> dict[str, Any]:
    event: dict[str, Any] = {"event": "sdk:event", "method": method}
    item_type = str(getattr(item, "type", "") or "")
    if item_type:
        event["item_type"] = item_type
    if item_type == "commandExecution":
        event["command_count"] = command_count
        status = _enum_value(getattr(item, "status", None))
        if status:
            event["status"] = status
    if method == "turn/completed":
        turn = getattr(payload, "turn", None)
        status = _enum_value(getattr(turn, "status", None))
        if status:
            event["status"] = status
        duration_ms = getattr(turn, "duration_ms", None)
        if isinstance(duration_ms, int) and not isinstance(duration_ms, bool) and duration_ms >= 0:
            event["duration_ms"] = duration_ms
    return event


def _root_item(item: Any) -> Any:
    return getattr(item, "root", item)


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


async def _best_effort_interrupt(turn: Any) -> None:
    try:
        await turn.interrupt()
    except Exception:  # noqa: BLE001 - cleanup must preserve the original SDK failure.
        return


def _run_codex_sync(
    coroutine_factory: Any,
    *,
    timeout_seconds: float,
) -> tuple[dict[str, Any], dict[str, int], dict[str, str]]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine_factory())

    result: dict[str, Any] = {}
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            result["value"] = asyncio.run(coroutine_factory())
        except BaseException as exc:  # noqa: BLE001 - preserve SDK failures across threads.
            errors.append(exc)

    thread = threading.Thread(target=worker, name="skillfabric-codex-sdk", daemon=True)
    thread.start()
    thread.join(timeout_seconds + 2.0)
    if thread.is_alive():
        raise TimeoutError(f"Codex query-wiki explorer exceeded {timeout_seconds:g} seconds")
    if errors:
        raise errors[0]
    value = result.get("value")
    if not isinstance(value, tuple) or len(value) != 3:
        raise RuntimeError("Codex SDK worker did not return an explorer result")
    return value


def _same_json_shape(value: Any, expected: Any) -> bool:
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(value) == set(expected) and all(
            _same_json_shape(value[key], expected[key]) for key in expected
        )
    return value == expected


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _require_int_at_least(value: Any, *, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")


def _safe_error_text(value: str, *, paths: tuple[Path, ...] = ()) -> str:
    text = re.sub(r"(?i)\bsk-[a-z0-9._-]+", "[redacted]", value)
    text = re.sub(
        r"(?i)\b([A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD)[A-Z0-9_]*)=\S+",
        r"\1=[redacted]",
        text,
    )
    for path in sorted((str(path) for path in paths), key=len, reverse=True):
        if path:
            text = text.replace(path, "[isolated-codex-home]")
    return text[:2_000]


def _write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_event(directory: Path, event: dict[str, Any]) -> None:
    path = directory / "agent_events.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def _load_sdk_runtime() -> Any:
    try:
        import openai_codex
        from openai_codex import ApprovalMode, AsyncCodex, CodexConfig
        from openai_codex.types import ReasoningEffort
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "openai-codex is required for the Codex query-wiki backend; "
            "install skillfabric-ai[codex]"
        ) from exc

    class Runtime:
        pass

    Runtime.__version__ = openai_codex.__version__
    Runtime.ApprovalMode = ApprovalMode
    Runtime.AsyncCodex = AsyncCodex
    Runtime.CodexConfig = CodexConfig
    Runtime.ReasoningEffort = ReasoningEffort
    return Runtime


__all__ = [
    "CODEX_EXECUTION_CONTRACT",
    "CodexExecutionContract",
    "CodexSdkRuntime",
    "CodexWikiExplorerBackend",
]
