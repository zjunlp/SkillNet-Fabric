"""Claude Code adapter for query-wiki skill-package exploration."""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skillfabric.router.models import RouterBundle
from skillfabric.sdk_env import build_claude_code_sdk_env
from skillfabric.storage import atomic_write_text
from skillfabric.wiki.explorer.prompting import (
    DEFAULT_TOOL_BUDGET,
    EXPLORER_PROMPT_ID,
    ExplorerPromptContext,
    render_system_prompt,
    render_user_prompt,
)
from skillfabric.wiki.explorer.skill_package import SkillPackage

ClaudeCodeSdkRuntime = Any

ALLOWED_TOOLS = ["Read", "LS", "Glob", "Grep"]
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
    """Run a controlled Claude Code-style explorer over query_wiki."""

    env_file: str | Path = ".env"
    max_selected_skills: int = 8
    model: str | None = None
    sdk_runtime: ClaudeCodeSdkRuntime | None = None
    max_turns: int = 24
    load_timeout_ms: int = 30_000
    execution_timeout_seconds: float = 300.0
    max_attempts: int = 3
    tool_budget: dict[str, int] | None = None

    def explore(
        self,
        *,
        query: str,
        query_wiki_root: Path,
        bundle: RouterBundle,
        trace_dir: Path,
    ) -> SkillPackage:
        """Return a SkillPackage from query_wiki-only context."""

        del bundle
        cc_dir = trace_dir / "cc_explorer"
        cc_dir.mkdir(parents=True, exist_ok=True)
        prompt_context = ExplorerPromptContext(
            query=query,
            query_wiki_root=query_wiki_root,
            max_selected_skills=self.max_selected_skills,
            allowed_tools=ALLOWED_TOOLS,
            tool_budget=_normalize_tool_budget(self.tool_budget),
        )
        system_prompt = render_system_prompt(prompt_context)
        user_prompt = render_user_prompt(prompt_context)
        atomic_write_text(cc_dir / "prompt.system.md", system_prompt)
        atomic_write_text(cc_dir / "prompt.user.md", user_prompt)
        atomic_write_text(
            cc_dir / "prompt_contract.json",
            json.dumps({"prompt_id": EXPLORER_PROMPT_ID}, ensure_ascii=False, indent=2) + "\n",
        )
        atomic_write_text(
            cc_dir / "prompt_context.json",
            json.dumps(prompt_context.to_trace_context(), ensure_ascii=False, indent=2) + "\n",
        )
        _write_event(
            cc_dir,
            {
                "event": "backend:start",
                "allowed_tools": ALLOWED_TOOLS,
                "read_root": str(query_wiki_root),
                "prompt_id": EXPLORER_PROMPT_ID,
                "tool_budget": _normalize_tool_budget(self.tool_budget),
            },
        )
        attempts = max(1, int(self.max_attempts))
        for attempt in range(1, attempts + 1):
            _write_event(cc_dir, {"event": "backend:attempt", "attempt": attempt, "max_attempts": attempts})
            try:
                payload, sdk_metrics = self._explore_with_sdk(system_prompt, user_prompt, query_wiki_root, cc_dir)
                package = _package_from_payload(payload, query_wiki_root)
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
                    {"event": "backend:finish", "attempt": attempt, "selected_count": len(package.selected_skills)},
                )
                return package
            except Exception as exc:
                error = {"error_type": type(exc).__name__, "error": str(exc), "attempt": attempt}
                if attempt < attempts and _is_retryable_explorer_error(exc):
                    _write_event(cc_dir, {"event": "backend:retry", **error})
                    continue
                atomic_write_text(cc_dir / "error.json", json.dumps(error, ensure_ascii=False, indent=2) + "\n")
                _write_event(cc_dir, {"event": "backend:error", **error})
                raise
        raise RuntimeError("Claude Code query-wiki explorer failed without raising an error.")

    def _explore_with_sdk(
        self,
        system_prompt: str,
        user_prompt: str,
        query_wiki_root: Path,
        cc_dir: Path,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        runtime = self.sdk_runtime or _load_sdk_runtime()
        options = _build_claude_agent_options(
            runtime,
            system_prompt=system_prompt,
            cwd=query_wiki_root,
            model=self.model,
            read_roots=[query_wiki_root],
            write_roots=[],
            env=_sdk_env(self.env_file),
            event_dir=cc_dir,
            max_turns=self.max_turns,
            load_timeout_ms=self.load_timeout_ms,
            tool_budget=_normalize_tool_budget(self.tool_budget),
        )
        result_message = _run_sdk_query_sync(
            runtime,
            prompt=user_prompt,
            options=options,
            event_dir=cc_dir,
            timeout_seconds=self.execution_timeout_seconds,
        )
        structured_output = _payload_from_result_message(result_message)
        return _unwrap_finish_payload(structured_output), _result_message_to_sdk_metrics(result_message)


def _build_claude_agent_options(
    runtime: ClaudeCodeSdkRuntime,
    *,
    system_prompt: str,
    cwd: Path,
    model: str | None,
    read_roots: list[Path],
    write_roots: list[Path],
    env: dict[str, str],
    event_dir: Path,
    max_turns: int,
    load_timeout_ms: int,
    tool_budget: dict[str, int],
) -> Any:
    cwd_path = cwd.resolve()
    resolved_read_roots = tuple(root.resolve() for root in read_roots)
    resolved_write_roots = tuple(root.resolve() for root in write_roots)
    all_roots = tuple(dict.fromkeys([cwd_path, *resolved_read_roots, *resolved_write_roots]))
    add_dirs = [str(root) for root in all_roots if root != cwd_path]
    permission_updates = _directory_permission_updates(runtime, all_roots)
    permission_updates_sent = False
    tool_counts: dict[str, int] = {tool: 0 for tool in ALLOWED_TOOLS}
    tool_counts["total"] = 0

    def allow_tool() -> Any:
        nonlocal permission_updates_sent
        if permission_updates and not permission_updates_sent:
            permission_updates_sent = True
            return runtime.PermissionResultAllow(updated_permissions=permission_updates)
        return runtime.PermissionResultAllow()

    async def can_use_tool(tool_name: str, tool_input: dict[str, Any], _context: Any) -> Any:
        if tool_name in WRITE_TOOLS:
            return runtime.PermissionResultDeny(message=f"{tool_name} is not allowed for query_wiki exploration.")
        path_key = PATH_KEYS.get(tool_name)
        if path_key is None:
            return runtime.PermissionResultDeny(message=f"{tool_name} is not allowed for query_wiki exploration.")
        raw_path = tool_input.get(path_key)
        if raw_path is None:
            candidate_path = cwd_path
        else:
            path = Path(str(raw_path))
            candidate_path = (cwd_path / path if not path.is_absolute() else path).resolve()
        if not any(candidate_path.is_relative_to(root) for root in resolved_read_roots):
            return runtime.PermissionResultDeny(message=f"{tool_name} path outside allowed read roots: {candidate_path}")
        budget_result = _consume_tool_budget(tool_name, tool_counts, tool_budget)
        if budget_result is not None:
            _write_event(
                event_dir,
                {
                    "event": "sdk:tool_denied_budget",
                    "tool": tool_name,
                    "tool_counts": dict(tool_counts),
                    "tool_budget": dict(tool_budget),
                    "message": budget_result,
                },
            )
            return runtime.PermissionResultDeny(message=budget_result)
        _write_event(
            event_dir,
            {
                "event": "sdk:tool_allowed",
                "tool": tool_name,
                "tool_counts": dict(tool_counts),
                "tool_budget": dict(tool_budget),
            },
        )
        return allow_tool()

    def stderr(line: str) -> None:
        if line:
            _write_event(event_dir, {"event": "sdk:stderr", "line": line})

    kwargs: dict[str, Any] = {
        "tools": list(ALLOWED_TOOLS),
        "allowed_tools": list(ALLOWED_TOOLS),
        "disallowed_tools": list(DISALLOWED_TOOLS),
        "permission_mode": "default",
        "system_prompt": system_prompt,
        "cwd": cwd_path,
        "add_dirs": add_dirs,
        "env": env,
        "effort": env.get("ANTHROPIC_REASONING_EFFORT") or None,
        "setting_sources": [],
        "extra_args": {"disable-slash-commands": None},
        "max_turns": max(1, int(max_turns)),
        "load_timeout_ms": max(1_000, int(load_timeout_ms)),
        "stderr": stderr,
        "can_use_tool": can_use_tool,
        "output_format": {
            "type": "json_schema",
            "schema": _skill_package_schema(),
        },
    }
    if model:
        kwargs["model"] = model
    return runtime.ClaudeAgentOptions(**kwargs)


def _consume_tool_budget(tool_name: str, tool_counts: dict[str, int], tool_budget: dict[str, int]) -> str | None:
    total_limit = tool_budget.get("total", 0)
    tool_limit = tool_budget.get(tool_name, 0)
    if total_limit > 0 and tool_counts.get("total", 0) >= total_limit:
        return f"query_wiki exploration tool budget exceeded: total<={total_limit}"
    if tool_limit > 0 and tool_counts.get(tool_name, 0) >= tool_limit:
        return f"query_wiki exploration tool budget exceeded: {tool_name}<={tool_limit}"
    tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
    tool_counts["total"] = tool_counts.get("total", 0) + 1
    return None


def _normalize_tool_budget(tool_budget: dict[str, int] | None) -> dict[str, int]:
    merged = dict(DEFAULT_TOOL_BUDGET)
    if tool_budget:
        for key, value in tool_budget.items():
            if key in ALLOWED_TOOLS or key == "total":
                try:
                    merged[key] = max(0, int(value))
                except (TypeError, ValueError):
                    continue
    return merged


def _directory_permission_updates(runtime: ClaudeCodeSdkRuntime, roots: tuple[Path, ...]) -> list[Any]:
    if not roots or not hasattr(runtime, "PermissionUpdate"):
        return []
    return [
        runtime.PermissionUpdate(
            type="addDirectories",
            directories=[str(root) for root in roots],
            destination="session",
        )
    ]


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
    timeout = max(1.0, float(timeout_seconds))
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_run_sdk_query_with_timeout(runtime, prompt=prompt, options=options, event_dir=event_dir, timeout_seconds=timeout))

    result: dict[str, Any] = {}
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            result["message"] = asyncio.run(
                _run_sdk_query_with_timeout(runtime, prompt=prompt, options=options, event_dir=event_dir, timeout_seconds=timeout)
            )
        except BaseException as exc:  # noqa: BLE001 - cross-thread boundary preserves SDK failures.
            errors.append(exc)

    thread = threading.Thread(target=worker, name="skillfabric-claude-agent-sdk", daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        _write_event(event_dir, {"event": "sdk:timeout", "timeout_seconds": timeout})
        raise TimeoutError(f"Claude Code query-wiki explorer exceeded {timeout:g} seconds")
    if errors:
        raise errors[0]
    return result["message"]


def _result_message_to_sdk_metrics(message: Any) -> dict[str, Any]:
    return {
        "duration_ms": getattr(message, "duration_ms", 0) or 0,
        "total_cost_usd": getattr(message, "total_cost_usd", 0.0) or 0.0,
        "input_tokens": getattr(message, "input_tokens", 0) or 0,
        "output_tokens": getattr(message, "output_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(message, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(message, "cache_read_input_tokens", 0) or 0,
        "num_turns": getattr(message, "num_turns", 0) or 0,
        "is_error": getattr(message, "is_error", False) or False,
        "subtype": getattr(message, "subtype", "") or "",
    }


async def _run_sdk_query(runtime: ClaudeCodeSdkRuntime, *, prompt: str, options: Any, event_dir: Path) -> Any:
    result_message = None
    assistant_text_parts: list[str] = []
    query_exception: BaseException | None = None
    try:
        async for message in runtime.query(prompt=_prompt_stream(prompt), options=options):
            _write_event(event_dir, _message_event(message))
            assistant_text_parts.extend(_assistant_text_parts(message))
            if isinstance(message, runtime.ResultMessage):
                result_message = message
    except Exception as exc:  # noqa: BLE001 - SDK may raise after yielding an error ResultMessage.
        if result_message is None:
            raise
        query_exception = exc
        _write_event(
            event_dir,
            {
                "event": "sdk:query_exception_after_result",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
    if result_message is None:
        raise RuntimeError("Claude agent finished without result message.")
    if assistant_text_parts and not getattr(result_message, "structured_output", None):
        result_message._skillfabric_assistant_text = "\n".join(assistant_text_parts)
    if query_exception is not None:
        result_message._skillfabric_query_exception = f"{type(query_exception).__name__}: {query_exception}"
    return result_message


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
        raise TimeoutError(f"Claude Code query-wiki explorer exceeded {timeout_seconds:g} seconds") from exc


async def _prompt_stream(prompt: str) -> Any:
    yield {
        "type": "user",
        "session_id": "",
        "message": {"role": "user", "content": prompt},
        "parent_tool_use_id": None,
    }


def _message_event(message: Any) -> dict[str, Any]:
    event: dict[str, Any] = {"event": "sdk:message", "type": type(message).__name__}
    subtype = getattr(message, "subtype", "")
    if subtype:
        event["subtype"] = subtype
    content = getattr(message, "content", None)
    if isinstance(content, list):
        tool_names = [
            str(getattr(block, "name", ""))
            for block in content
            if getattr(block, "name", "")
        ]
        text_chars = sum(len(str(getattr(block, "text", ""))) for block in content if getattr(block, "text", ""))
        if tool_names:
            event["tools"] = tool_names
        if text_chars:
            event["text_chars"] = text_chars
            text_preview = "\n".join(_assistant_text_parts(message)).strip()
            if text_preview.startswith("API Error:"):
                event["text_preview"] = text_preview[:500]
    if type(message).__name__ == "ResultMessage":
        event["is_error"] = bool(getattr(message, "is_error", False))
        event["structured_output_present"] = getattr(message, "structured_output", None) is not None
        result = str(getattr(message, "result", "") or "")
        if result:
            event["result_preview"] = result[:500]
    return event


def _assistant_text_parts(message: Any) -> list[str]:
    if type(message).__name__ != "AssistantMessage":
        return []
    parts: list[str] = []
    for block in getattr(message, "content", []) or []:
        text = getattr(block, "text", "")
        if text:
            parts.append(str(text))
    return parts


def _load_sdk_runtime() -> Any:
    try:
        from claude_agent_sdk import ClaudeAgentOptions, query
        from claude_agent_sdk.types import (
            PermissionResultAllow,
            PermissionResultDeny,
            PermissionUpdate,
            ResultMessage,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError("claude-agent-sdk is required for the claude-code explorer backend") from exc

    class Runtime:
        pass

    Runtime.ClaudeAgentOptions = ClaudeAgentOptions
    Runtime.PermissionResultAllow = PermissionResultAllow
    Runtime.PermissionResultDeny = PermissionResultDeny
    Runtime.PermissionUpdate = PermissionUpdate
    Runtime.ResultMessage = ResultMessage
    Runtime.query = staticmethod(query)
    return Runtime


def _skill_package_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "selected_skills": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "skill_id": {"type": "string", "description": "Manifest skill_id selected from query_wiki."},
                        "scope": {
                            "type": "string",
                            "enum": ["core", "workflow_bridge", "graph_frontier"],
                            "description": "Scope from manifest.json.",
                        },
                        "role": {"type": "string", "description": "Short evidence-grounded reason for selecting this skill."},
                        "evidence": {
                            "type": "array",
                            "description": "Files under query_wiki that justify selecting this skill.",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "path": {"type": "string", "description": "Relative query_wiki evidence path."},
                                    "reason": {"type": "string", "description": "Why this file supports the selection."},
                                },
                                "required": ["path", "reason"],
                            },
                        },
                    },
                    "required": ["skill_id", "scope", "role", "evidence"],
                },
            },
            "required_edges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "before": {"type": "string", "description": "Skill id that must run before the after skill."},
                        "after": {"type": "string", "description": "Skill id that consumes context, artifacts, or state."},
                        "relation_type": {
                            "type": "string",
                            "enum": ["depend_on", "compose_with", "artifact_compatibility", "state_compatibility"],
                            "description": "Dependency type for the before -> after edge.",
                        },
                        "evidence_path": {"type": "string", "description": "Relative query_wiki evidence path for the edge."},
                        "reason": {"type": "string", "description": "Why the edge direction is before -> after."},
                    },
                    "required": ["before", "after", "relation_type", "evidence_path", "reason"],
                },
            },
            "ordered_hints": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"skill_id": {"type": "string"}, "hint": {"type": "string"}},
                    "required": ["skill_id", "hint"],
                },
            },
            "near_misses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"skill_id": {"type": "string"}, "reason": {"type": "string"}},
                    "required": ["skill_id", "reason"],
                },
            },
            "coverage_notes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "requirement_id": {"type": "string"},
                        "status": {"type": "string"},
                        "reason": {"type": "string"},
                        "skill_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["requirement_id", "status", "reason", "skill_ids"],
                },
            },
            "rationale": {"type": "string"},
        },
        "required": ["selected_skills", "required_edges", "ordered_hints", "near_misses", "coverage_notes", "rationale"],
    }


def _unwrap_finish_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("result"), str):
        try:
            nested = json.loads(str(payload["result"]))
        except json.JSONDecodeError:
            nested = None
        if isinstance(nested, dict):
            return _unwrap_finish_payload(nested)
    if payload.get("action") == "finish" and isinstance(payload.get("args"), dict):
        return dict(payload["args"])
    return payload


def _payload_from_result_message(result_message: Any) -> dict[str, Any]:
    if getattr(result_message, "is_error", False):
        raise RuntimeError(_result_message_error_detail(result_message))
    structured_output = getattr(result_message, "structured_output", None)
    if isinstance(structured_output, dict):
        return structured_output
    if structured_output is not None:
        raise RuntimeError("Claude agent structured output was not a JSON object.")
    for text in (
        getattr(result_message, "_skillfabric_assistant_text", ""),
        getattr(result_message, "result", ""),
    ):
        payload = _json_object_from_text(str(text or ""))
        if payload is not None:
            return payload
    raise RuntimeError("Claude agent finished without structured output or JSON assistant text.")


def _result_message_error_detail(result_message: Any) -> str:
    candidates = [
        str(getattr(result_message, "result", "") or "").strip(),
        str(getattr(result_message, "_skillfabric_assistant_text", "") or "").strip(),
        str(getattr(result_message, "_skillfabric_query_exception", "") or "").strip(),
    ]
    for candidate in candidates:
        if candidate.startswith("API Error:"):
            return candidate
    for candidate in candidates:
        if candidate and candidate.lower() != "success":
            return candidate
    return "Claude agent query failed."


def _is_retryable_explorer_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    retryable_markers = (
        "api error",
        "socket connection was closed",
        "service temporarily unavailable",
        "503",
        "502",
        "504",
        "bad gateway",
        "gateway timeout",
        "connection reset",
        "connection aborted",
        "econnreset",
        "etimedout",
    )
    return any(marker in text for marker in retryable_markers)


def _json_object_from_text(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    candidates = [stripped]
    if "```" in stripped:
        parts = stripped.split("```")
        candidates.extend(part.strip() for part in parts if part.strip())
        candidates.extend(
            part.removeprefix("json").strip()
            for part in parts
            if part.strip().lower().startswith("json")
        )
    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        candidates.append(stripped[first_brace : last_brace + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _package_from_payload(payload: dict[str, Any], query_wiki_root: Path) -> SkillPackage:
    normalized = dict(payload)
    manifest = _load_manifest(query_wiki_root)
    scope_by_skill = {
        str(item.get("skill_id", "")): str(item.get("scope", ""))
        for item in manifest.get("skills", [])
        if isinstance(item, dict)
    }
    selected = []
    for raw in normalized.get("selected_skills", []):
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        skill_id = str(row.get("skill_id", ""))
        row.setdefault("scope", scope_by_skill.get(skill_id, ""))
        if not row.get("role"):
            row["role"] = _first_text(row, "selection_reason", "why_selected", "why", "reason")
        evidence = row.get("evidence", row.get("evidence_paths", []))
        if isinstance(evidence, str):
            evidence = [evidence]
        if isinstance(evidence, list):
            row["evidence"] = [
                _normalize_evidence_item(item, query_wiki_root)
                for item in evidence
                if str(item)
            ]
        selected.append(row)
    normalized["selected_skills"] = selected
    normalized["required_edges"] = [
        edge
        for edge in (
            _normalize_edge(item, query_wiki_root)
            for item in normalized.get("required_edges", [])
            if isinstance(item, dict)
        )
        if edge["before"] and edge["after"]
    ]
    normalized["ordered_hints"] = _normalize_ordered_hints(normalized.get("ordered_hints", []))
    normalized["near_misses"] = _normalize_near_misses(normalized.get("near_misses", []))
    normalized["coverage_notes"] = _normalize_coverage_notes(normalized.get("coverage_notes", []))
    return SkillPackage.from_dict(normalized)


def _normalize_edge(raw: dict[str, Any], query_wiki_root: Path) -> dict[str, Any]:
    evidence_paths = raw.get("evidence_path", raw.get("evidence_paths", ""))
    if isinstance(evidence_paths, list):
        evidence_path = next((str(item) for item in evidence_paths if str(item).startswith(("edges/", "workflows/"))), "")
        if not evidence_path and evidence_paths:
            evidence_path = str(evidence_paths[0])
    else:
        evidence_path = str(evidence_paths)
    return {
        "before": str(raw.get("before", raw.get("before_skill", raw.get("from", "")))),
        "after": str(raw.get("after", raw.get("after_skill", raw.get("to", "")))),
        "relation_type": str(raw.get("relation_type", raw.get("edge_type", raw.get("type", raw.get("relation", "depend_on"))))),
        "evidence_path": _normalize_query_wiki_path(evidence_path, query_wiki_root),
        "reason": _first_text(raw, "reason", "selection_reason", "why"),
    }


def _normalize_evidence_item(raw: Any, query_wiki_root: Path) -> dict[str, str]:
    if isinstance(raw, dict):
        path = str(raw.get("path", ""))
        reason = _first_text(raw, "reason", "selection_reason", "why")
    else:
        path = str(raw)
        reason = ""
    return {"path": _normalize_query_wiki_path(path, query_wiki_root), "reason": reason}


def _normalize_query_wiki_path(path: str, query_wiki_root: Path) -> str:
    if not path:
        return ""
    candidate = Path(path)
    if not candidate.is_absolute():
        return path
    try:
        return candidate.resolve().relative_to(query_wiki_root.resolve()).as_posix()
    except (OSError, ValueError):
        return path


def _normalize_ordered_hints(raw_hints: Any) -> list[dict[str, str]]:
    if not isinstance(raw_hints, list):
        return []
    hints: list[dict[str, str]] = []
    for item in raw_hints:
        if isinstance(item, dict):
            hints.append({"skill_id": str(item.get("skill_id", "")), "hint": str(item.get("hint", ""))})
        elif isinstance(item, str) and item.startswith("skill:"):
            hints.append({"skill_id": item, "hint": "ordered sequence"})
    return [item for item in hints if item["skill_id"]]


def _normalize_near_misses(raw_near_misses: Any) -> list[dict[str, str]]:
    if not isinstance(raw_near_misses, list):
        return []
    near_misses: list[dict[str, str]] = []
    for item in raw_near_misses:
        if not isinstance(item, dict):
            continue
        near_misses.append(
            {
                "skill_id": str(item.get("skill_id", "")),
                "reason": _first_text(item, "reason", "why_not_selected", "why", "selection_reason"),
            }
        )
    return [item for item in near_misses if item["skill_id"]]


def _normalize_coverage_notes(raw_notes: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_notes, list):
        return []
    notes: list[dict[str, Any]] = []
    for index, item in enumerate(raw_notes, start=1):
        if isinstance(item, dict):
            notes.append(item)
        elif isinstance(item, str) and item:
            notes.append(
                {
                    "requirement_id": f"note:{index}",
                    "status": "note",
                    "reason": item,
                    "skill_ids": [],
                }
            )
    return notes


def _first_text(payload: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if value:
            return str(value)
    return ""


def _load_manifest(query_wiki_root: Path) -> dict[str, Any]:
    manifest_path = query_wiki_root / "manifest.json"
    if not manifest_path.exists():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_event(cc_dir: Path, event: dict[str, Any]) -> None:
    path = cc_dir / "agent_events.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
