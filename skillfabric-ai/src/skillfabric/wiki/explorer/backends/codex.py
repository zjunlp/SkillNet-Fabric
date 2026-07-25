"""OpenAI Codex SDK backend for strict query-wiki exploration."""

from __future__ import annotations

import asyncio
import json
import math
import shutil
import threading
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from skillfabric.runtime.sdk_env import CodexSdkEnvironment, build_codex_sdk_env
from skillfabric.storage import atomic_write_text
from skillfabric.wiki.explorer.backends.codex_session import (
    CodexOperationalAccessError,
    CodexStructuredOutputError,
    run_codex_attempt,
)
from skillfabric.wiki.explorer.prompting import (
    EXPLORER_PROMPT_ID,
    ExplorerPromptContext,
    default_tool_budget,
    render_system_prompt,
    render_user_prompt,
    validate_required_selected_skills,
)
from skillfabric.wiki.explorer.redaction import sanitize_error_text
from skillfabric.wiki.explorer.skill_package import SkillPackage, skill_package_json_schema

CodexSdkRuntime = Any
PromptSpecBuilder = Callable[[Mapping[str, Any]], Mapping[str, str]]
CODEX_ALLOWED_TOOLS = ("exec_command",)
PERMISSION_PROFILE = "skillfabric-query-wiki"
CODEX_EXECUTION_GUIDANCE = """
<codex_execution_guidance>
- Begin with `index.md`, then use non-interactive `exec_command` reads and searches.
- Keep paths relative to the supplied query-wiki root; inspect cards, sources, and
  `semantic_edges.jsonl` only when they are needed as routing evidence.
- Use read/list/search commands such as `sed`, `head`, `find`, and `rg`; do not use
  `write_stdin`, interactive shells, redirects, file edits, network tools, or background jobs.
- A successful route must be grounded in files actually read from this query wiki. If a command
  fails, simplify the next read instead of returning a coverage gap without inspecting the wiki.
</codex_execution_guidance>
""".strip()


def build_codex_prompt_spec(
    *,
    query: str,
    query_wiki_root: str | Path,
    max_selected_skills: int,
    required_selected_skills: int | None = None,
    tool_budget: dict[str, int] | None = None,
    prompt_spec_builder: PromptSpecBuilder | None = None,
) -> dict[str, Any]:
    """Build the exact Codex prompt and schema payload used by the backend."""

    context = ExplorerPromptContext(
        query=query,
        query_wiki_root=query_wiki_root,
        max_selected_skills=max_selected_skills,
        required_selected_skills=required_selected_skills,
        allowed_tools=CODEX_ALLOWED_TOOLS,
        tool_budget=_normalize_tool_budget(
            tool_budget,
            max_selected_skills=max_selected_skills,
        ),
    )
    payload = {
        "prompt_id": EXPLORER_PROMPT_ID,
        "query_wiki_root": context.query_wiki_root,
        "max_selected_skills": context.max_selected_skills,
        "allowed_tools": list(context.allowed_tools),
        "tool_budget": dict(context.tool_budget or {}),
        "permission_profile": PERMISSION_PROFILE,
        "execution_contract": CODEX_EXECUTION_CONTRACT.to_dict(),
        "system_prompt": render_system_prompt(context) + "\n\n" + CODEX_EXECUTION_GUIDANCE,
        "user_prompt": render_user_prompt(context),
        "schema": skill_package_json_schema(),
    }
    if context.required_selected_skills is not None:
        payload["required_selected_skills"] = context.required_selected_skills
    if prompt_spec_builder is not None:
        builder_context = {**payload, "query": query}
        payload = _apply_prompt_spec_builder(
            payload,
            prompt_spec_builder,
            builder_context=builder_context,
        )
    return payload


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
    codex_bin: str | Path | None = None
    sdk_runtime: CodexSdkRuntime | None = None
    tool_budget: dict[str, int] | None = None
    required_selected_skills: int | None = None
    prompt_spec_builder: PromptSpecBuilder | None = None

    CODEX_EXECUTION_CONTRACT = CODEX_EXECUTION_CONTRACT

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
        if self.prompt_spec_builder is not None and not callable(self.prompt_spec_builder):
            raise TypeError("prompt_spec_builder must be callable")
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

        query_wiki_root = Path(query_wiki_root)
        if query_wiki_root.is_symlink():
            raise ValueError("query_wiki root must not be a symlink")
        query_wiki_root = query_wiki_root.resolve()
        if not query_wiki_root.is_dir():
            raise FileNotFoundError(f"query_wiki root does not exist: {query_wiki_root}")
        cc_dir = trace_dir / "cc_explorer"
        cc_dir.mkdir(parents=True, exist_ok=True)
        tool_budget = dict(self.tool_budget or {})
        prompt_spec = build_codex_prompt_spec(
            query=query,
            query_wiki_root=query_wiki_root,
            max_selected_skills=self.max_selected_skills,
            required_selected_skills=self.required_selected_skills,
            tool_budget=tool_budget,
            prompt_spec_builder=self.prompt_spec_builder,
        )
        system_prompt = str(prompt_spec["system_prompt"])
        user_prompt = str(prompt_spec["user_prompt"])
        atomic_write_text(cc_dir / "prompt.system.md", system_prompt)
        atomic_write_text(cc_dir / "prompt.user.md", user_prompt)
        atomic_write_text(
            cc_dir / "prompt_contract.json",
            json.dumps(
                {
                    "prompt_id": prompt_spec["prompt_id"],
                    "schema": prompt_spec["schema"],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )
        prompt_context = {
            key: prompt_spec[key]
            for key in (
                "prompt_id",
                "query_wiki_root",
                "max_selected_skills",
                "allowed_tools",
                "tool_budget",
                "permission_profile",
                "execution_contract",
            )
        }
        if "required_selected_skills" in prompt_spec:
            prompt_context["required_selected_skills"] = prompt_spec[
                "required_selected_skills"
            ]
        atomic_write_text(
            cc_dir / "prompt_context.json",
            json.dumps(prompt_context, ensure_ascii=False, indent=2) + "\n",
        )
        atomic_write_text(
            cc_dir / "prompt_spec.json",
            json.dumps(prompt_spec, ensure_ascii=False, indent=2) + "\n",
        )
        _write_json(
            cc_dir / "backend.json",
            self._backend_payload(
                runtime=None,
                metadata=None,
                prompt_id=str(prompt_spec["prompt_id"]),
            ),
        )
        _write_event(
            cc_dir,
            {
                "event": "backend:start",
                "backend": "codex",
                "allowed_tools": list(CODEX_ALLOWED_TOOLS),
                "command_budget": _command_budget(tool_budget),
                "prompt_id": prompt_spec["prompt_id"],
            },
        )
        codex_home: Path | None = None
        try:
            runtime = self.sdk_runtime or _load_sdk_runtime()
            _write_json(
                cc_dir / "backend.json",
                self._backend_payload(
                    runtime=runtime,
                    metadata=None,
                    prompt_id=str(prompt_spec["prompt_id"]),
                ),
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
                        prompt_id=str(prompt_spec["prompt_id"]),
                    ),
                    timeout_seconds=self.execution_timeout_seconds,
                )
            _write_json(
                cc_dir / "backend.json",
                self._backend_payload(
                    runtime,
                    metadata,
                    prompt_id=str(prompt_spec["prompt_id"]),
                ),
            )
            _write_json(cc_dir / "usage.json", usage)
            try:
                package = SkillPackage.from_dict(payload)
            except (TypeError, ValueError) as exc:
                raise CodexStructuredOutputError(
                    "Codex agent returned an invalid structured skill package"
                ) from exc
            _validate_operational_result(package, cc_dir=cc_dir)
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
                    paths=(
                        query_wiki_root,
                        Path(self.env_file).expanduser().resolve(),
                        *(() if codex_home is None else (codex_home,)),
                    ),
                ),
            }
            _write_json(cc_dir / "error.json", error)
            _write_event(cc_dir, {"event": "backend:error", **error})
            with suppress(AttributeError, TypeError):
                exc.__skillfabric_sanitized_error__ = error["error"]  # type: ignore[attr-defined]
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
        prompt_id: str,
    ) -> tuple[dict[str, Any], dict[str, int], dict[str, str]]:
        return await run_codex_attempt(
            runtime,
            settings,
            codex_bin=None if self.codex_bin is None else str(self.codex_bin),
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            query_wiki_root=query_wiki_root,
            codex_home=codex_home,
            cc_dir=cc_dir,
            command_budget=command_budget,
            execution_timeout_seconds=self.execution_timeout_seconds,
            thread_config=_thread_config(
                query_wiki_root=query_wiki_root,
                codex_home=codex_home,
                api_base=settings.api_base,
                codex_bin=_codex_runtime_path(self.codex_bin),
            ),
            output_schema=skill_package_json_schema(),
            metadata_callback=lambda metadata: _write_json(
                cc_dir / "backend.json",
                self._backend_payload(
                    runtime=runtime,
                    metadata=metadata,
                    prompt_id=prompt_id,
                ),
            ),
        )

    def _backend_payload(
        self,
        runtime: CodexSdkRuntime | None,
        metadata: dict[str, str] | None,
        *,
        prompt_id: str = EXPLORER_PROMPT_ID,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "backend": "codex",
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "sdk_version": str(getattr(runtime, "__version__", "unavailable")),
            "execution_contract": CODEX_EXECUTION_CONTRACT.to_dict(),
            "permission_profile": PERMISSION_PROFILE,
            "prompt_id": prompt_id,
            "allowed_tools": list(CODEX_ALLOWED_TOOLS),
            "tool_enforcement": "event-audited-fail-closed",
            "command_budget": _command_budget(dict(self.tool_budget or {})),
        }
        if metadata:
            payload["app_server"] = metadata
        return payload


def _thread_config(
    *,
    query_wiki_root: Path,
    codex_home: Path,
    api_base: str,
    codex_bin: Path | None,
) -> dict[str, Any]:
    filesystem = {
        ":minimal": "read",
        str(query_wiki_root): "read",
    }
    if codex_bin is not None:
        filesystem[str(codex_bin)] = "read"
    return {
        "openai_base_url": api_base,
        "default_permissions": PERMISSION_PROFILE,
        "permissions": {
            PERMISSION_PROFILE: {
                "filesystem": filesystem,
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


def _codex_runtime_path(codex_bin: str | Path | None) -> Path | None:
    if codex_bin is None:
        return None
    candidate = Path(codex_bin).expanduser()
    if candidate.parent == Path("."):
        resolved = shutil.which(str(candidate))
        if resolved is not None:
            return Path(resolved).resolve()
    return candidate.resolve()


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


def _apply_prompt_spec_builder(
    default_spec: dict[str, Any],
    builder: PromptSpecBuilder,
    *,
    builder_context: Mapping[str, Any],
) -> dict[str, Any]:
    override = builder(dict(builder_context))
    if not isinstance(override, Mapping):
        raise TypeError("prompt_spec_builder must return a mapping")
    allowed = {"prompt_id", "system_prompt", "user_prompt"}
    unexpected = set(override) - allowed
    if unexpected:
        raise ValueError(
            "unsupported prompt spec fields: " + ", ".join(sorted(str(item) for item in unexpected))
        )
    result = dict(default_spec)
    for key, value in override.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"prompt spec field {key} must be a non-empty string")
        result[key] = value
    return result


def _command_budget(tool_budget: dict[str, int]) -> int:
    return min(tool_budget.get("exec_command", 0), tool_budget.get("total", 0))


def _validate_operational_result(package: SkillPackage, *, cc_dir: Path) -> None:
    access_path = cc_dir / "operational_access.json"
    try:
        access = json.loads(access_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CodexOperationalAccessError(
            "Codex query-wiki explorer did not publish valid operational access evidence"
        ) from exc
    if not isinstance(access, dict) or access.get("evidence_access") is not True:
        raise CodexOperationalAccessError(
            "Codex query-wiki explorer did not prove successful Wiki evidence access"
        )
    semantic_empty = not package.selected_skills
    access["semantic_empty"] = semantic_empty
    if semantic_empty:
        access["semantic_empty_valid"] = bool(
            package.coverage_gaps
            and access.get("index_read") is True
            and access.get("candidate_lookup") is True
        )
        _write_json(access_path, access)
        if access["semantic_empty_valid"] is not True:
            raise CodexOperationalAccessError(
                "Codex empty selection requires index.md access, candidate lookup, "
                "and an explicit coverage gap"
            )
        return
    access["semantic_empty_valid"] = False
    _write_json(access_path, access)


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


def _require_int_at_least(value: Any, *, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")


def _safe_error_text(value: str, *, paths: tuple[Path, ...] = ()) -> str:
    return sanitize_error_text(
        value,
        paths=paths,
        path_replacement="[isolated-codex-home]",
    )


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
    "CodexOperationalAccessError",
    "CodexSdkRuntime",
    "CodexWikiExplorerBackend",
    "PromptSpecBuilder",
    "build_codex_prompt_spec",
]
