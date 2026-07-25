"""Codex turn lifecycle, fail-closed event auditing, and usage telemetry."""

from __future__ import annotations

import asyncio
import json
import re
import shlex
from collections.abc import Callable
from pathlib import Path
from typing import Any

from skillfabric.storage import atomic_write_text

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


class CodexOperationalAccessError(RuntimeError):
    """The turn completed without proving that the query wiki was read successfully."""

    __skillfabric_recoverable_route_failure__ = True


class CodexStructuredOutputError(ValueError):
    """The completed turn returned output that cannot form a SkillPackage."""

    __skillfabric_recoverable_route_failure__ = True


async def run_codex_attempt(
    runtime: Any,
    settings: Any,
    *,
    codex_bin: str | None,
    model: str | None,
    reasoning_effort: str,
    system_prompt: str,
    user_prompt: str,
    query_wiki_root: Path,
    codex_home: Path,
    cc_dir: Path,
    command_budget: int,
    execution_timeout_seconds: float,
    thread_config: dict[str, Any],
    output_schema: dict[str, Any],
    metadata_callback: Callable[[dict[str, str]], None] | None = None,
) -> tuple[dict[str, Any], dict[str, int], dict[str, str]]:
    """Run one isolated app-server attempt and return structured output and telemetry.

    The backend owns prompt construction and result validation. This module owns the
    SDK lifecycle so login, thread/turn ordering, event closure, interruption, and
    cleanup have one implementation shared by every Codex caller.
    """

    sdk_env = dict(settings.env)
    sdk_env["CODEX_APP_SERVER_DISABLE_MANAGED_CONFIG"] = "1"
    config = runtime.CodexConfig(
        codex_bin=codex_bin,
        cwd=str(codex_home),
        env=sdk_env,
        config_overrides=(
            "project_root_markers=[]",
            "check_for_update_on_startup=false",
        ),
    )
    codex = runtime.AsyncCodex(config=config)
    turn = None
    operation_succeeded = False
    primary_error: BaseException | None = None
    try:
        async with asyncio.timeout(execution_timeout_seconds):
            await codex.__aenter__()
            await codex.login_api_key(settings.env["OPENAI_API_KEY"])
            metadata = _sdk_metadata(codex.metadata)
            if metadata_callback is not None:
                metadata_callback(metadata)
            thread = await codex.thread_start(
                approval_mode=runtime.ApprovalMode.deny_all,
                base_instructions=system_prompt,
                config=thread_config,
                cwd=str(query_wiki_root),
                ephemeral=True,
                model=model,
            )
            turn = await thread.turn(
                user_prompt,
                cwd=str(query_wiki_root),
                effort=runtime.ReasoningEffort(reasoning_effort),
                model=model,
                output_schema=output_schema,
            )
            _write_json(
                cc_dir / "turn_state.json",
                {"schema_version": 1, "turn_started": True},
            )
            final_response, usage = await collect_codex_turn(
                turn,
                cc_dir=cc_dir,
                command_budget=command_budget,
                query_wiki_root=query_wiki_root,
            )
            operation_succeeded = True
            return _strict_json_object(final_response), usage, metadata
    except TimeoutError as exc:
        primary_error = exc
        if turn is not None:
            await _best_effort_interrupt(turn)
        _write_event(
            cc_dir,
            {"event": "sdk:timeout", "timeout_seconds": execution_timeout_seconds},
        )
        raise TimeoutError(
            f"Codex query-wiki explorer exceeded {execution_timeout_seconds:g} seconds"
        ) from exc
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            await codex.close()
        except Exception as exc:
            _write_event(
                cc_dir,
                {
                    "event": "sdk:cleanup_error",
                    "error_type": type(exc).__name__,
                    "error": _safe_error_text(str(exc), paths=(codex_home, query_wiki_root)),
                },
            )
            if operation_succeeded and primary_error is None:
                raise


async def collect_codex_turn(
    turn: Any,
    *,
    cc_dir: Path,
    command_budget: int,
    query_wiki_root: Path,
) -> tuple[str, dict[str, int]]:
    command_count = 0
    successful_read_commands = 0
    query_wiki_commands = 0
    index_read = False
    candidate_lookup = False
    evidence_categories: set[str] = set()
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
        if item_type == "commandExecution":
            audit = _command_audit(item, query_wiki_root=query_wiki_root)
            observed_violation = observed_violation or str(audit["policy_violation"])
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
        elif method == "item/completed" and item_type == "commandExecution":
            audit = _command_audit(item, query_wiki_root=query_wiki_root)
            if audit["cwd_within_query_wiki"]:
                query_wiki_commands += 1
            if audit["successful_read"]:
                successful_read_commands += 1
                evidence_categories.update(audit["evidence_categories"])
                index_read = index_read or "index" in audit["evidence_categories"]
            candidate_lookup = candidate_lookup or bool(audit["candidate_lookup"])
        elif method == "item/completed" and item_type == "agentMessage":
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
        _write_event(
            cc_dir,
            _event_summary(
                method,
                payload,
                item,
                command_count,
                query_wiki_root=query_wiki_root,
            ),
        )

    evidence_access = successful_read_commands > 0 and bool(evidence_categories)
    operational_access = {
        "schema_version": 1,
        "command_count": command_count,
        "query_wiki_commands": query_wiki_commands,
        "successful_read_commands": successful_read_commands,
        "evidence_access": evidence_access,
        "index_read": index_read,
        "candidate_lookup": candidate_lookup,
        "evidence_categories": sorted(evidence_categories),
        "policy_violation": policy_violation or None,
    }
    _write_json(cc_dir / "operational_access.json", operational_access)
    usage_payload: dict[str, int] | None = None
    if completed_turn is not None and usage is not None:
        usage_payload = _sdk_usage(
            usage,
            completed_turn=completed_turn,
            command_count=command_count,
        )
        _write_json(cc_dir / "usage.json", usage_payload)
    if budget_exceeded:
        raise RuntimeError(f"Codex query-wiki tool budget exceeded: exec_command<={command_budget}")
    if policy_violation:
        error = RuntimeError(
            f"Codex query-wiki observed disallowed Codex tool activity: {policy_violation}"
        )
        error.__skillfabric_non_retryable__ = True  # type: ignore[attr-defined]
        raise error
    if completed_turn is None:
        raise RuntimeError("Codex agent finished without a turn/completed event")
    status = _enum_value(getattr(completed_turn, "status", None))
    if status != "completed":
        error = getattr(completed_turn, "error", None)
        detail = str(getattr(error, "message", "") or status or "unknown status")
        raise RuntimeError(
            f"Codex agent turn failed: {_safe_error_text(detail, paths=(query_wiki_root,))}"
        )
    if usage_payload is None:
        raise RuntimeError("Codex agent completed without a token usage closure")
    response = final_response or fallback_response
    if not response.strip():
        raise RuntimeError("Codex agent did not return a structured SkillPackage")
    if not evidence_access:
        error = CodexOperationalAccessError(
            "Codex query-wiki explorer completed without successful Wiki evidence access"
        )
        error.usage = usage_payload  # type: ignore[attr-defined]
        error.operational_access = operational_access  # type: ignore[attr-defined]
        raise error
    return response, usage_payload


def _sdk_usage(usage: Any, *, completed_turn: Any, command_count: int) -> dict[str, int]:
    total = getattr(usage, "total", None)
    return {
        "duration_ms": _nonnegative_int(getattr(completed_turn, "duration_ms", 0)),
        "input_tokens": _nonnegative_int(getattr(total, "input_tokens", 0)),
        "cache_read_input_tokens": _nonnegative_int(getattr(total, "cached_input_tokens", 0)),
        "output_tokens": _nonnegative_int(getattr(total, "output_tokens", 0)),
        "reasoning_output_tokens": _nonnegative_int(getattr(total, "reasoning_output_tokens", 0)),
        "total_tokens": _nonnegative_int(getattr(total, "total_tokens", 0)),
        "total_calls": command_count,
        "num_turns": 1,
    }


def _event_summary(
    method: str,
    payload: Any,
    item: Any,
    command_count: int,
    *,
    query_wiki_root: Path,
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
        audit = _command_audit(item, query_wiki_root=query_wiki_root)
        event.update(
            {
                "command_category": audit["category"],
                "cwd_within_query_wiki": audit["cwd_within_query_wiki"],
                "successful_read": audit["successful_read"],
                "candidate_lookup": audit["candidate_lookup"],
                "evidence_categories": audit["evidence_categories"],
                "process_id_present": audit["process_id_present"],
            }
        )
        if audit["policy_violation"]:
            event["policy_violation"] = audit["policy_violation"]
        if audit["exit_code"] is not None:
            event["exit_code"] = audit["exit_code"]
    if method == "turn/completed":
        turn = getattr(payload, "turn", None)
        status = _enum_value(getattr(turn, "status", None))
        if status:
            event["status"] = status
        duration_ms = getattr(turn, "duration_ms", None)
        if isinstance(duration_ms, int) and not isinstance(duration_ms, bool) and duration_ms >= 0:
            event["duration_ms"] = duration_ms
    return event


def _command_audit(item: Any, *, query_wiki_root: Path) -> dict[str, Any]:
    """Summarize a command without persisting its command text or output."""

    root = query_wiki_root.resolve()
    cwd_value = getattr(item, "cwd", None)
    cwd = root if cwd_value in (None, "") else _resolve_audit_path(cwd_value, root)
    try:
        cwd_within = cwd == root or root in cwd.parents
    except (OSError, RuntimeError):
        cwd_within = False
    status = _enum_value(getattr(item, "status", None)).casefold()
    exit_code = getattr(item, "exit_code", None)
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        exit_code = None
    actions = _command_actions(getattr(item, "command_actions", None))
    action_types = {_enum_value(getattr(action, "type", None)).casefold() for action in actions}
    category = next(
        (name for name in ("read", "list", "search") if name in action_types), "unknown"
    )
    if "listfiles" in action_types or "list_files" in action_types:
        category = "list"
    command = str(getattr(item, "command", "") or "")
    if category == "unknown":
        category = _fallback_command_category(command)
    evidence_categories = _evidence_categories(actions, command=command)
    paths_within = _action_paths_within_root(actions, root=root, cwd=cwd)
    policy_violation = _command_policy_violation(
        command,
        action_types=action_types,
        paths_within_query_wiki=paths_within,
    )
    successful = status == "completed" and exit_code == 0
    query_wiki_activity = cwd_within and paths_within and not policy_violation
    return {
        "category": category,
        "cwd_within_query_wiki": cwd_within,
        "successful_read": (
            successful and query_wiki_activity and category in {"read", "list", "search"}
        ),
        "candidate_lookup": (
            successful
            and query_wiki_activity
            and (category == "search" or bool({"card", "source", "relation"} & evidence_categories))
        ),
        "evidence_categories": sorted(evidence_categories),
        "exit_code": exit_code,
        "policy_violation": policy_violation,
        "process_id_present": getattr(item, "process_id", None) not in (None, ""),
    }


def _resolve_audit_path(value: Any, root: Path) -> Path:
    path = Path(str(value))
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _command_actions(value: Any) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        return []
    return [_root_item(action) for action in value]


def _fallback_command_category(command: str) -> str:
    lowered = command.casefold()
    if re.search(r"(?:^|[\s;&|])(rg|grep)(?:[\s;&|]|$)", lowered):
        return "search"
    if re.search(r"(?:^|[\s;&|])(find|ls)(?:[\s;&|]|$)", lowered):
        return "list"
    if re.search(r"(?:^|[\s;&|])(cat|head|tail|sed|stat|wc)(?:[\s;&|]|$)", lowered):
        return "read"
    return "unknown"


def _evidence_categories(actions: list[Any], *, command: str) -> set[str]:
    values = [str(getattr(action, "path", "") or "") for action in actions]
    values.append(command)
    lowered = "\n".join(values).replace("\\", "/").casefold()
    categories: set[str] = set()
    if re.search(r"(?:^|[\s/'\"]|/)index\.md(?:$|[\s'\"])", lowered):
        categories.add("index")
    if re.search(r"(?:^|/)cards?(?:/|\b)", lowered):
        categories.add("card")
    if re.search(r"(?:^|/)sources?(?:/|\b)", lowered):
        categories.add("source")
    if "semantic_edges.jsonl" in lowered or re.search(r"(?:^|/)edges(?:/|\b)", lowered):
        categories.add("relation")
    return categories


def _action_paths_within_root(actions: list[Any], *, root: Path, cwd: Path) -> bool:
    for action in actions:
        value = getattr(action, "path", None)
        if value in (None, ""):
            continue
        try:
            path = Path(str(value))
            resolved = (cwd / path).resolve() if not path.is_absolute() else path.resolve()
            if resolved != root and root not in resolved.parents:
                return False
        except (OSError, RuntimeError, ValueError):
            return False
    return True


def _command_policy_violation(
    command: str,
    *,
    action_types: set[str],
    paths_within_query_wiki: bool,
) -> str:
    if not paths_within_query_wiki:
        return "outside_query_wiki"
    allowed_actions = {"read", "listfiles", "list_files", "search", "unknown", ""}
    if action_types - allowed_actions:
        return "write_attempt"
    tokens = _shell_tokens(command)
    if any(">" in token and set(token) <= set(";&|<>") for token in tokens):
        return "write_attempt"
    tokens = _unwrap_app_server_shell_wrapper(tokens)
    if tokens is None:
        return "unrestricted_runtime"
    if any(">" in token and set(token) <= set(";&|<>") for token in tokens):
        return "write_attempt"
    commands = _shell_command_names(tokens)
    if commands & {
        "rm",
        "mv",
        "cp",
        "touch",
        "mkdir",
        "rmdir",
        "chmod",
        "chown",
        "tee",
        "truncate",
        "dd",
        "install",
        "patch",
    }:
        return "write_attempt"
    if commands & {"curl", "wget", "ssh", "scp", "sftp", "nc", "ncat", "telnet", "ftp"}:
        return "network_attempt"
    if commands & {"python", "python3", "node", "ruby", "perl", "bash", "sh", "zsh"}:
        return "unrestricted_runtime"
    if "&" in tokens:
        return "background_job"
    return ""


def _unwrap_app_server_shell_wrapper(tokens: list[str]) -> list[str] | None:
    """Return the audited command inside the app-server's fixed shell transport."""

    if not tokens or Path(tokens[0]).name.casefold() not in {"bash", "sh", "zsh"}:
        return tokens
    if len(tokens) != 3 or tokens[1] not in {"-c", "-cl", "-lc"}:
        return None
    inner_tokens = _shell_tokens(tokens[2])
    return inner_tokens or None


def _shell_tokens(command: str) -> list[str]:
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>")
        lexer.commenters = ""
        return list(lexer)
    except ValueError:
        return []


def _shell_command_names(tokens: list[str]) -> set[str]:
    names: set[str] = set()
    expect_command = True
    for token in tokens:
        if token in {";", "&&", "||", "|", "&"}:
            expect_command = True
        elif expect_command and not any(character in token for character in "<>"):
            names.add(Path(token).name.casefold())
            expect_command = False
    return names


def _root_item(item: Any) -> Any:
    return getattr(item, "root", item)


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


async def _best_effort_interrupt(turn: Any) -> None:
    try:
        await turn.interrupt()
    except Exception:  # noqa: BLE001 - cleanup must preserve the original SDK failure.
        return


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


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
    except ValueError as exc:
        raise CodexStructuredOutputError(
            "Codex agent did not return valid SkillPackage JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise CodexStructuredOutputError("Codex agent SkillPackage response must be a JSON object")
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


def _safe_error_text(value: str, *, paths: tuple[Path, ...] = ()) -> str:
    text = re.sub(r"(?i)\bsk-[a-z0-9._-]+", "[redacted]", value)
    text = re.sub(
        r"(?i)\b([A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD)[A-Z0-9_]*)=\S+",
        r"\1=[redacted]",
        text,
    )
    text = re.sub(r"(?i)\bBearer\s+\S+", "Bearer [redacted]", text)
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


__all__ = [
    "CodexOperationalAccessError",
    "CodexStructuredOutputError",
    "collect_codex_turn",
    "run_codex_attempt",
]
