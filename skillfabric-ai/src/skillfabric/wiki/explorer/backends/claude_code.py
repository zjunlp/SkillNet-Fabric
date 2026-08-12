"""Claude Agent SDK backend for strict query-wiki exploration."""

from __future__ import annotations

import asyncio
import json
import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skillfabric.runtime.sdk_env import build_claude_code_sdk_env
from skillfabric.storage import atomic_write_text
from skillfabric.wiki.explorer.prompting import (
    DEFAULT_ALLOWED_TOOLS,
    EXPLORER_PROMPT_ID,
    ExplorerPromptContext,
    default_tool_budget,
    render_system_prompt,
    render_user_prompt,
    validate_required_selected_skills,
)
from skillfabric.wiki.explorer.skill_package import SkillPackage, skill_package_json_schema

ClaudeCodeSdkRuntime = Any

ALLOWED_TOOLS = list(DEFAULT_ALLOWED_TOOLS)
PATH_KEYS = {
    "Glob": "path",
    "Grep": "path",
    "LS": "path",
    "Read": "file_path",
}
WRITE_TOOLS = {"Edit", "NotebookEdit", "Write"}
DISALLOWED_TOOLS = sorted(
    WRITE_TOOLS
    | {
        "Agent",
        "AskUserQuestion",
        "Bash",
        "EnterPlanMode",
        "ExitPlanMode",
        "Task",
        "TodoWrite",
        "WebFetch",
        "WebSearch",
    }
)


@dataclass(slots=True)
class ClaudeCodeWikiExplorerBackend:
    env_file: str | Path = ".env"
    max_selected_skills: int = 8
    model: str | None = None
    sdk_runtime: ClaudeCodeSdkRuntime | None = None
    max_turns: int = 24
    load_timeout_ms: int = 30_000
    execution_timeout_seconds: float = 300.0
    tool_budget: dict[str, int] | None = None
    reasoning_effort: str | None = None
    required_selected_skills: int | None = None

    def __post_init__(self) -> None:
        _require_int_at_least(
            self.max_selected_skills,
            name="max_selected_skills",
            minimum=0,
        )
        validate_required_selected_skills(
            self.required_selected_skills,
            max_selected_skills=self.max_selected_skills,
        )
        _require_int_at_least(self.max_turns, name="max_turns", minimum=1)
        _require_int_at_least(
            self.load_timeout_ms,
            name="load_timeout_ms",
            minimum=1_000,
        )
        timeout = self.execution_timeout_seconds
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout < 0
        ):
            raise ValueError("execution_timeout_seconds must be finite and non-negative")
        if self.model is not None and not self.model.strip():
            raise ValueError("model must be a non-empty string when provided")
        if self.reasoning_effort is not None and (
            not isinstance(self.reasoning_effort, str) or not self.reasoning_effort.strip()
        ):
            raise ValueError("reasoning_effort must be a non-empty string when provided")
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
            required_selected_skills=self.required_selected_skills,
            allowed_tools=ALLOWED_TOOLS,
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
        _write_event(
            cc_dir,
            {
                "event": "backend:start",
                "allowed_tools": ALLOWED_TOOLS,
                "read_root": str(query_wiki_root),
                "prompt_id": EXPLORER_PROMPT_ID,
                "tool_budget": tool_budget,
            },
        )
        try:
            payload = self._explore_with_sdk(
                system_prompt,
                user_prompt,
                query_wiki_root,
                cc_dir,
                tool_budget,
            )
            package = SkillPackage.from_dict(payload)
            atomic_write_text(
                cc_dir / "skill_package.json",
                json.dumps(package.to_dict(), ensure_ascii=False, indent=2) + "\n",
            )
            _write_event(
                cc_dir,
                {"event": "backend:finish", "selected_count": len(package.selected_skills)},
            )
            return package
        except Exception as exc:
            error = {
                "error_type": type(exc).__name__,
                "error": _safe_error_text(str(exc)),
            }
            atomic_write_text(
                cc_dir / "error.json",
                json.dumps(error, ensure_ascii=False, indent=2) + "\n",
            )
            _write_event(cc_dir, {"event": "backend:error", **error})
            raise

    def _explore_with_sdk(
        self,
        system_prompt: str,
        user_prompt: str,
        query_wiki_root: Path,
        cc_dir: Path,
        tool_budget: dict[str, int],
    ) -> dict[str, Any]:
        runtime = self.sdk_runtime or _load_sdk_runtime()
        options = _build_claude_agent_options(
            runtime,
            system_prompt=system_prompt,
            cwd=query_wiki_root,
            model=self.model,
            read_roots=[query_wiki_root],
            env=_sdk_env(
                self.env_file,
                model=self.model,
                reasoning_effort=self.reasoning_effort,
            ),
            event_dir=cc_dir,
            max_turns=self.max_turns,
            load_timeout_ms=self.load_timeout_ms,
            tool_budget=tool_budget,
        )
        result_message = _run_sdk_query_sync(
            runtime,
            prompt=user_prompt,
            options=options,
            event_dir=cc_dir,
            timeout_seconds=self.execution_timeout_seconds,
        )
        return _payload_from_result_message(result_message)


def _build_claude_agent_options(
    runtime: ClaudeCodeSdkRuntime,
    *,
    system_prompt: str,
    cwd: Path,
    model: str | None,
    read_roots: list[Path],
    env: dict[str, str],
    event_dir: Path,
    max_turns: int,
    load_timeout_ms: int,
    tool_budget: dict[str, int],
) -> Any:
    cwd_path = cwd.resolve()
    resolved_roots = tuple(dict.fromkeys(root.resolve() for root in read_roots))
    tool_counts = dict.fromkeys(ALLOWED_TOOLS, 0)
    tool_counts["total"] = 0

    async def pre_tool_use(
        hook_input: dict[str, Any],
        _tool_use_id: str | None,
        _context: Any,
    ) -> dict[str, Any]:
        tool_name = str(hook_input.get("tool_name", ""))
        raw_input = hook_input.get("tool_input", {})
        tool_input = raw_input if isinstance(raw_input, dict) else {}
        path_key = PATH_KEYS.get(tool_name)
        if path_key is None:
            return _hook_permission(
                "deny",
                f"{tool_name} is not allowed for query_wiki exploration.",
            )
        raw_path = tool_input.get(path_key)
        candidate = cwd_path if raw_path is None else Path(str(raw_path))
        candidate = (cwd_path / candidate if not candidate.is_absolute() else candidate).resolve()
        if not any(candidate.is_relative_to(root) for root in resolved_roots):
            return _hook_permission(
                "deny",
                f"{tool_name} path is outside the query_wiki read root.",
            )
        budget_error = _consume_tool_budget(tool_name, tool_counts, tool_budget)
        if budget_error:
            _write_event(
                event_dir,
                {
                    "event": "sdk:tool_denied_budget",
                    "tool": tool_name,
                    "tool_counts": dict(tool_counts),
                    "tool_budget": dict(tool_budget),
                },
            )
            return _hook_permission("deny", budget_error)
        _write_event(
            event_dir,
            {
                "event": "sdk:tool_allowed",
                "tool": tool_name,
                "tool_counts": dict(tool_counts),
            },
        )
        return _hook_permission("allow")

    def stderr(line: str) -> None:
        if line:
            _write_event(event_dir, {"event": "sdk:stderr", "line": _safe_error_text(line)})

    kwargs: dict[str, Any] = {
        "tools": list(ALLOWED_TOOLS),
        "allowed_tools": list(ALLOWED_TOOLS),
        "disallowed_tools": list(DISALLOWED_TOOLS),
        "permission_mode": "default",
        "system_prompt": system_prompt,
        "cwd": cwd_path,
        "add_dirs": [str(root) for root in resolved_roots if root != cwd_path],
        "env": env,
        "effort": env.get("ANTHROPIC_REASONING_EFFORT") or None,
        "setting_sources": [],
        "extra_args": {"disable-slash-commands": None},
        "max_turns": max_turns,
        "load_timeout_ms": load_timeout_ms,
        "stderr": stderr,
        "hooks": {
            "PreToolUse": [
                runtime.HookMatcher(
                    matcher="|".join(ALLOWED_TOOLS),
                    hooks=[pre_tool_use],
                )
            ]
        },
        "output_format": {
            "type": "json_schema",
            "schema": skill_package_json_schema(),
        },
    }
    if model:
        kwargs["model"] = model
    return runtime.ClaudeAgentOptions(**kwargs)


def _consume_tool_budget(
    tool_name: str,
    tool_counts: dict[str, int],
    tool_budget: dict[str, int],
) -> str | None:
    total_limit = tool_budget.get("total", 0)
    tool_limit = tool_budget.get(tool_name, 0)
    if tool_counts["total"] >= total_limit:
        return f"query_wiki tool budget exceeded: total<={total_limit}"
    if tool_counts[tool_name] >= tool_limit:
        return f"query_wiki tool budget exceeded: {tool_name}<={tool_limit}"
    tool_counts[tool_name] += 1
    tool_counts["total"] += 1
    return None


def _normalize_tool_budget(
    tool_budget: dict[str, int] | None,
    *,
    max_selected_skills: int,
) -> dict[str, int]:
    merged = default_tool_budget(max_selected_skills)
    if tool_budget is None:
        return merged
    allowed_keys = {*ALLOWED_TOOLS, "total"}
    unexpected = set(tool_budget) - allowed_keys
    if unexpected:
        raise ValueError(f"tool_budget has unsupported keys: {', '.join(sorted(unexpected))}")
    for key, value in tool_budget.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"tool_budget.{key} must be a non-negative integer")
        merged[key] = value
    return merged


def _hook_permission(decision: str, reason: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {
        "hookEventName": "PreToolUse",
        "permissionDecision": decision,
    }
    if reason:
        output["permissionDecisionReason"] = reason
    return {"hookSpecificOutput": output}


def _require_int_at_least(value: Any, *, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")


def _sdk_env(
    env_file: str | Path,
    *,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, str]:
    return build_claude_code_sdk_env(
        env_file,
        model=model,
        reasoning_effort=reasoning_effort,
    )


def _run_sdk_query_sync(
    runtime: ClaudeCodeSdkRuntime,
    *,
    prompt: str,
    options: Any,
    event_dir: Path,
    timeout_seconds: float,
) -> Any:
    timeout = timeout_seconds
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            _run_sdk_query_with_timeout(
                runtime,
                prompt=prompt,
                options=options,
                event_dir=event_dir,
                timeout_seconds=timeout,
            )
        )

    result: dict[str, Any] = {}
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            result["message"] = asyncio.run(
                _run_sdk_query_with_timeout(
                    runtime,
                    prompt=prompt,
                    options=options,
                    event_dir=event_dir,
                    timeout_seconds=timeout,
                )
            )
        except BaseException as exc:  # noqa: BLE001 - preserve cross-thread SDK failures.
            errors.append(exc)

    thread = threading.Thread(target=worker, name="skillfabric-claude-agent-sdk", daemon=True)
    thread.start()
    thread.join(None if timeout == 0 else timeout + 1.0)
    if thread.is_alive():
        _write_event(event_dir, {"event": "sdk:timeout", "timeout_seconds": timeout})
        raise TimeoutError(f"Claude query-wiki explorer exceeded {timeout:g} seconds")
    if errors:
        raise errors[0]
    return result["message"]


async def _run_sdk_query_with_timeout(
    runtime: ClaudeCodeSdkRuntime,
    *,
    prompt: str,
    options: Any,
    event_dir: Path,
    timeout_seconds: float,
) -> Any:
    if timeout_seconds == 0:
        return await _run_sdk_query(
            runtime,
            prompt=prompt,
            options=options,
            event_dir=event_dir,
        )
    try:
        return await asyncio.wait_for(
            _run_sdk_query(runtime, prompt=prompt, options=options, event_dir=event_dir),
            timeout=timeout_seconds,
        )
    except TimeoutError as exc:
        _write_event(event_dir, {"event": "sdk:timeout", "timeout_seconds": timeout_seconds})
        raise TimeoutError(
            f"Claude query-wiki explorer exceeded {timeout_seconds:g} seconds"
        ) from exc


async def _run_sdk_query(
    runtime: ClaudeCodeSdkRuntime,
    *,
    prompt: str,
    options: Any,
    event_dir: Path,
) -> Any:
    result_message = None
    try:
        async for message in runtime.query(prompt=_prompt_stream(prompt), options=options):
            event = _message_event(message)
            if event is not None:
                _write_event(event_dir, event)
            if isinstance(message, runtime.ResultMessage):
                result_message = message
    except Exception as exc:
        if result_message is None:
            raise
        _write_event(
            event_dir,
            {
                "event": "sdk:post_result_error",
                "error_type": type(exc).__name__,
                "error": _safe_error_text(str(exc)),
            },
        )
    if result_message is None:
        raise RuntimeError("Claude agent finished without a ResultMessage")
    return result_message


async def _prompt_stream(prompt: str) -> Any:
    yield {
        "type": "user",
        "session_id": "",
        "message": {"role": "user", "content": prompt},
        "parent_tool_use_id": None,
    }


def _payload_from_result_message(result_message: Any) -> dict[str, Any]:
    if getattr(result_message, "is_error", False):
        raise RuntimeError(_result_message_error_detail(result_message))
    structured_output = getattr(result_message, "structured_output", None)
    if not isinstance(structured_output, dict):
        raise RuntimeError("Claude agent did not return a structured SkillPackage object")
    return structured_output


def _result_message_error_detail(result_message: Any) -> str:
    details: list[str] = []
    api_error_status = getattr(result_message, "api_error_status", None)
    if isinstance(api_error_status, int) and not isinstance(api_error_status, bool):
        details.append(f"HTTP {api_error_status}")
    errors = getattr(result_message, "errors", None)
    if isinstance(errors, list):
        details.extend(
            _safe_error_text(value) for item in errors if (value := str(item or "").strip())
        )
    if details:
        return "Claude agent API request failed: " + "; ".join(details)
    for value in (
        getattr(result_message, "result", ""),
        getattr(result_message, "subtype", ""),
    ):
        text = str(value or "").strip()
        if text and text.lower() != "success":
            return _safe_error_text(text)
    return "Claude agent query failed"


def _message_event(message: Any) -> dict[str, Any] | None:
    message_type = type(message).__name__
    subtype = getattr(message, "subtype", "")
    if message_type == "SystemMessage" and subtype == "thinking_tokens":
        return None
    event: dict[str, Any] = {"event": "sdk:message", "type": message_type}
    if subtype:
        event["subtype"] = str(subtype)
    content = getattr(message, "content", None)
    if isinstance(content, list):
        tools = [str(getattr(block, "name", "")) for block in content if getattr(block, "name", "")]
        if tools:
            event["tools"] = tools
    if message_type == "AssistantMessage":
        error = str(getattr(message, "error", "") or "").strip()
        if error:
            event["error"] = _safe_error_text(error)
    if message_type == "ResultMessage":
        event["is_error"] = bool(getattr(message, "is_error", False))
        event["structured_output_present"] = isinstance(
            getattr(message, "structured_output", None), dict
        )
        api_error_status = getattr(message, "api_error_status", None)
        if isinstance(api_error_status, int) and not isinstance(api_error_status, bool):
            event["api_error_status"] = api_error_status
        errors = getattr(message, "errors", None)
        if isinstance(errors, list):
            event["errors"] = [
                _safe_error_text(value) for item in errors if (value := str(item or "").strip())
            ]
    elif subtype == "api_retry":
        diagnostics = _safe_system_diagnostics(getattr(message, "data", None))
        if diagnostics:
            event["diagnostics"] = diagnostics
    return event


def _safe_system_diagnostics(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    allowed = {
        "attempt",
        "error",
        "max_retries",
        "retry_delay_ms",
        "status",
        "status_code",
    }
    diagnostics: dict[str, Any] = {}
    for key in sorted(allowed & data.keys()):
        value = data[key]
        if isinstance(value, str):
            diagnostics[key] = _safe_error_text(value)
        elif isinstance(value, (int, float, bool)) or value is None:
            diagnostics[key] = value
    return diagnostics


def _load_sdk_runtime() -> Any:
    try:
        from claude_agent_sdk import ClaudeAgentOptions, HookMatcher, query
        from claude_agent_sdk.types import ResultMessage
    except ModuleNotFoundError as exc:
        raise RuntimeError("claude-agent-sdk is required for query-wiki routing") from exc

    class Runtime:
        pass

    Runtime.ClaudeAgentOptions = ClaudeAgentOptions
    Runtime.HookMatcher = HookMatcher
    Runtime.ResultMessage = ResultMessage
    Runtime.query = staticmethod(query)
    return Runtime


def _safe_error_text(value: str) -> str:
    text = re.sub(r"(?i)\bsk-[a-z0-9._-]+", "[redacted]", value)
    return re.sub(
        r"(?i)\b(API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY|ANTHROPIC_AUTH_TOKEN)=\S+",
        r"\1=[redacted]",
        text,
    )[:2_000]


def _write_event(directory: Path, event: dict[str, Any]) -> None:
    path = directory / "agent_events.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


__all__ = ["ClaudeCodeSdkRuntime", "ClaudeCodeWikiExplorerBackend"]
