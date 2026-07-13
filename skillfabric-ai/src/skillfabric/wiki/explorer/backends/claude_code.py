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
    max_attempts: int = 2
    tool_budget: dict[str, int] | None = None

    def __post_init__(self) -> None:
        _require_int_at_least(
            self.max_selected_skills,
            name="max_selected_skills",
            minimum=0,
        )
        _require_int_at_least(self.max_turns, name="max_turns", minimum=1)
        _require_int_at_least(
            self.load_timeout_ms,
            name="load_timeout_ms",
            minimum=1_000,
        )
        _require_int_at_least(self.max_attempts, name="max_attempts", minimum=1)
        timeout = self.execution_timeout_seconds
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout < 1.0
        ):
            raise ValueError("execution_timeout_seconds must be finite and at least 1")
        if self.model is not None and not self.model.strip():
            raise ValueError("model must be a non-empty string when provided")
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
        attempts = self.max_attempts
        for attempt in range(1, attempts + 1):
            _write_event(
                cc_dir,
                {"event": "backend:attempt", "attempt": attempt, "max_attempts": attempts},
            )
            try:
                payload, sdk_metrics = self._explore_with_sdk(
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
                atomic_write_text(
                    cc_dir / "usage.json",
                    json.dumps(
                        {
                            "backend": "claude-code",
                            "runtime": "claude-agent-sdk",
                            "sdk_metrics": sdk_metrics,
                        },
                        indent=2,
                    )
                    + "\n",
                )
                _write_event(
                    cc_dir,
                    {
                        "event": "backend:finish",
                        "attempt": attempt,
                        "selected_count": len(package.selected_skills),
                    },
                )
                return package
            except Exception as exc:
                error = {
                    "error_type": type(exc).__name__,
                    "error": _safe_error_text(str(exc)),
                    "attempt": attempt,
                }
                if attempt < attempts and _is_retryable_explorer_error(exc):
                    _write_event(cc_dir, {"event": "backend:retry", **error})
                    continue
                atomic_write_text(
                    cc_dir / "error.json",
                    json.dumps(error, ensure_ascii=False, indent=2) + "\n",
                )
                _write_event(cc_dir, {"event": "backend:error", **error})
                raise
        raise RuntimeError("Claude explorer stopped without a result or error")

    def _explore_with_sdk(
        self,
        system_prompt: str,
        user_prompt: str,
        query_wiki_root: Path,
        cc_dir: Path,
        tool_budget: dict[str, int],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        runtime = self.sdk_runtime or _load_sdk_runtime()
        options = _build_claude_agent_options(
            runtime,
            system_prompt=system_prompt,
            cwd=query_wiki_root,
            model=self.model,
            read_roots=[query_wiki_root],
            env=_sdk_env(self.env_file),
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
        return _payload_from_result_message(result_message), _sdk_metrics(result_message)


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


def _sdk_env(env_file: str | Path) -> dict[str, str]:
    return build_claude_code_sdk_env(env_file)


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
    thread.join(timeout + 1.0)
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
    async for message in runtime.query(prompt=_prompt_stream(prompt), options=options):
        event = _message_event(message)
        if event is not None:
            _write_event(event_dir, event)
        if isinstance(message, runtime.ResultMessage):
            result_message = message
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
    for value in (
        getattr(result_message, "result", ""),
        getattr(result_message, "subtype", ""),
    ):
        text = str(value or "").strip()
        if text and text.lower() != "success":
            return _safe_error_text(text)
    return "Claude agent query failed"


def _sdk_metrics(message: Any) -> dict[str, Any]:
    usage = getattr(message, "usage", None)
    if not isinstance(usage, dict):
        usage = {}
    return {
        "duration_ms": getattr(message, "duration_ms", 0) or 0,
        "total_cost_usd": getattr(message, "total_cost_usd", 0.0) or 0.0,
        "input_tokens": usage.get("input_tokens", 0) or 0,
        "output_tokens": usage.get("output_tokens", 0) or 0,
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0) or 0,
        "num_turns": getattr(message, "num_turns", 0) or 0,
        "is_error": bool(getattr(message, "is_error", False)),
        "subtype": getattr(message, "subtype", "") or "",
    }


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
    if message_type == "ResultMessage":
        event["is_error"] = bool(getattr(message, "is_error", False))
        event["structured_output_present"] = isinstance(
            getattr(message, "structured_output", None), dict
        )
    return event


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


def _is_retryable_explorer_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "socket connection was closed",
            "service temporarily unavailable",
            "bad gateway",
            "gateway timeout",
            "connection reset",
            "connection aborted",
            "econnreset",
            "etimedout",
            " 502",
            " 503",
            " 504",
        )
    )


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
