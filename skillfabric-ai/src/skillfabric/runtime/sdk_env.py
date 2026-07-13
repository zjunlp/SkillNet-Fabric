"""Environment helpers for Claude Code SDK adapters."""

from __future__ import annotations

import os
from pathlib import Path

from skillfabric.runtime.llm import read_env_file


def build_claude_code_sdk_env(
    env_file: str | Path,
    *,
    runtime_env_path: str | Path | None = None,
) -> dict[str, str]:
    """Resolve Claude Code SDK env with env-file values taking precedence."""

    shell_env = {key: value for key, value in os.environ.items() if value}
    env_file_values = {key: value for key, value in read_env_file(env_file).items() if value}
    env = {**shell_env, **env_file_values}
    llm_api_key = _first_env_value(
        shell_env,
        env_file_values,
        "SKILLFABRIC_LLM_API_KEY",
        "API_KEY",
        "OPENAI_API_KEY",
    )
    llm_api_base = _first_env_value(
        shell_env,
        env_file_values,
        "SKILLFABRIC_LLM_API_BASE",
        "BASE_URL",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
    )
    llm_model = _first_env_value(
        shell_env,
        env_file_values,
        "SKILLFABRIC_LLM_MODEL",
        "MODEL",
    )
    llm_reasoning_effort = _first_env_value(
        shell_env,
        env_file_values,
        "SKILLFABRIC_LLM_REASONING_EFFORT",
    )
    if llm_api_key:
        env["OPENAI_API_KEY"] = llm_api_key
        if _env_file_has_any(
            env_file_values, "SKILLFABRIC_LLM_API_KEY", "API_KEY", "OPENAI_API_KEY"
        ):
            env["ANTHROPIC_AUTH_TOKEN"] = llm_api_key
            env["ANTHROPIC_API_KEY"] = llm_api_key
        elif not env.get("ANTHROPIC_AUTH_TOKEN"):
            env["ANTHROPIC_AUTH_TOKEN"] = llm_api_key
        if not env.get("ANTHROPIC_API_KEY"):
            env["ANTHROPIC_API_KEY"] = env.get("ANTHROPIC_AUTH_TOKEN", llm_api_key)
    if llm_api_base:
        env["OPENAI_BASE_URL"] = llm_api_base
        env["OPENAI_API_BASE"] = llm_api_base
        if _env_file_has_any(
            env_file_values,
            "SKILLFABRIC_LLM_API_BASE",
            "BASE_URL",
            "OPENAI_BASE_URL",
            "OPENAI_API_BASE",
        ) or not env.get("ANTHROPIC_BASE_URL"):
            env["ANTHROPIC_BASE_URL"] = _anthropic_base_url(llm_api_base)
    if llm_model:
        model_name = _claude_model_name(llm_model)
        if _env_file_has_any(env_file_values, "SKILLFABRIC_LLM_MODEL", "MODEL") or not env.get(
            "ANTHROPIC_MODEL"
        ):
            env["ANTHROPIC_MODEL"] = model_name
        env.setdefault("ANTHROPIC_SMALL_FAST_MODEL", env["ANTHROPIC_MODEL"])
        env.setdefault("ANTHROPIC_DEFAULT_SONNET_MODEL", env["ANTHROPIC_MODEL"])
        env.setdefault("ANTHROPIC_DEFAULT_HAIKU_MODEL", env["ANTHROPIC_MODEL"])
        env.setdefault("ANTHROPIC_DEFAULT_OPUS_MODEL", env["ANTHROPIC_MODEL"])
    if llm_reasoning_effort:
        env["ANTHROPIC_REASONING_EFFORT"] = llm_reasoning_effort
    if runtime_env_path is not None:
        _pin_python_runtime(env, runtime_env_path)
    return env


def _first_env_value(shell_env: dict[str, str], env_file_values: dict[str, str], *keys: str) -> str:
    for key in keys:
        if env_file_values.get(key):
            return env_file_values[key]
    for key in keys:
        if shell_env.get(key):
            return shell_env[key]
    return ""


def _env_file_has_any(env_file_values: dict[str, str], *keys: str) -> bool:
    return any(bool(env_file_values.get(key)) for key in keys)


def _pin_python_runtime(env: dict[str, str], runtime_env_path: str | Path) -> None:
    """Pin Python and pip resolution for Claude Code Bash subprocesses."""

    runtime_env = Path(runtime_env_path).expanduser().resolve()
    bin_dir = runtime_env / "bin"
    python = bin_dir / "python"
    python3 = bin_dir / "python3"
    pip = bin_dir / "pip"
    if not python.exists():
        raise FileNotFoundError(f"runtime Python does not exist: {python}")
    if not python3.exists():
        python3 = python
    if not pip.exists():
        raise FileNotFoundError(f"runtime pip does not exist: {pip}")
    env["VIRTUAL_ENV"] = str(runtime_env)
    env["PYTHON"] = str(python)
    env["PYTHON3"] = str(python3)
    env["PIP_REQUIRE_VIRTUALENV"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env["PATH"] = _prepend_path(env.get("PATH", ""), bin_dir)


def _prepend_path(current_path: str, prefix: Path) -> str:
    prefix_text = str(prefix)
    existing = [part for part in current_path.split(os.pathsep) if part and part != prefix_text]
    return os.pathsep.join([prefix_text, *existing])


def _anthropic_base_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    return stripped[:-3] if stripped.endswith("/v1") else stripped


def _claude_model_name(model: str) -> str:
    for prefix in ("openai/responses/", "openai/"):
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model
