from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from skillfabric.runtime.sdk_env import build_claude_code_sdk_env, build_codex_sdk_env

SDK_ENV_KEYS = (
    "API_KEY",
    "BASE_URL",
    "MODEL",
    "SKILLFABRIC_LLM_API_BASE",
    "SKILLFABRIC_LLM_API_KEY",
    "SKILLFABRIC_LLM_MODEL",
    "SKILLFABRIC_LLM_REASONING_EFFORT",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_SMALL_FAST_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_REASONING_EFFORT",
    "CODEX_ACCESS_TOKEN",
    "CODEX_API_KEY",
    "CODEX_HOME",
    "CODEX_SQLITE_HOME",
)


def _cleared_sdk_env() -> dict[str, str]:
    return dict.fromkeys(SDK_ENV_KEYS, "")


class ClaudeCodeSdkEnvTests(unittest.TestCase):
    def test_skillfabric_llm_env_drives_openai_and_claude_code_aliases(self) -> None:
        with TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "SKILLFABRIC_LLM_API_BASE=http://gateway.example/v1",
                        "SKILLFABRIC_LLM_API_KEY=sk-test",
                        "SKILLFABRIC_LLM_MODEL=openai/responses/gpt-5.4-mini",
                        "SKILLFABRIC_LLM_REASONING_EFFORT=medium",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, _cleared_sdk_env(), clear=False):
                env = build_claude_code_sdk_env(env_path)

        self.assertEqual(env["OPENAI_API_KEY"], "sk-test")
        self.assertEqual(env["OPENAI_BASE_URL"], "http://gateway.example/v1")
        self.assertEqual(env["OPENAI_API_BASE"], "http://gateway.example/v1")
        self.assertEqual(env["ANTHROPIC_AUTH_TOKEN"], "sk-test")
        self.assertEqual(env["ANTHROPIC_API_KEY"], "sk-test")
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://gateway.example")
        self.assertEqual(env["ANTHROPIC_MODEL"], "gpt-5.4-mini")
        self.assertEqual(env["ANTHROPIC_SMALL_FAST_MODEL"], "gpt-5.4-mini")
        self.assertEqual(env["ANTHROPIC_DEFAULT_SONNET_MODEL"], "gpt-5.4-mini")
        self.assertEqual(env["ANTHROPIC_DEFAULT_HAIKU_MODEL"], "gpt-5.4-mini")
        self.assertEqual(env["ANTHROPIC_DEFAULT_OPUS_MODEL"], "gpt-5.4-mini")
        self.assertEqual(env["ANTHROPIC_REASONING_EFFORT"], "medium")


class CodexSdkEnvTests(unittest.TestCase):
    def test_codex_uses_the_experiment_api_and_an_isolated_home(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_path = root / ".env"
            codex_home = root / "codex-home"
            codex_home.mkdir()
            env_path.write_text(
                "\n".join(
                    [
                        "SKILLFABRIC_LLM_API_BASE=http://gateway.example/v1",
                        "SKILLFABRIC_LLM_API_KEY=sk-experiment",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "sk-personal",
                    "CODEX_API_KEY": "codex-personal",
                    "CODEX_ACCESS_TOKEN": "access-personal",
                    "CODEX_HOME": "/personal/codex-home",
                    "CODEX_SQLITE_HOME": "/personal/codex-state",
                },
                clear=False,
            ):
                settings = build_codex_sdk_env(env_path, codex_home=codex_home)

        self.assertEqual(settings.api_base, "http://gateway.example/v1")
        self.assertEqual(settings.env["OPENAI_API_KEY"], "sk-experiment")
        self.assertEqual(settings.env["CODEX_API_KEY"], "")
        self.assertEqual(settings.env["CODEX_ACCESS_TOKEN"], "")
        self.assertEqual(settings.env["CODEX_HOME"], str(codex_home.resolve()))
        self.assertEqual(settings.env["CODEX_SQLITE_HOME"], str(codex_home.resolve()))
        self.assertEqual(settings.env["HOME"], str(codex_home.resolve()))

    def test_codex_env_requires_an_api_key_and_existing_home(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_path = root / ".env"
            env_path.write_text("OPENAI_API_KEY=\n", encoding="utf-8")
            cleared = dict.fromkeys(
                ("API_KEY", "SKILLFABRIC_LLM_API_KEY", "OPENAI_API_KEY"),
                "",
            )
            with (
                patch.dict(os.environ, cleared, clear=False),
                self.assertRaisesRegex(ValueError, "missing API key"),
            ):
                build_codex_sdk_env(env_path, codex_home=root)

            with self.assertRaisesRegex(FileNotFoundError, "Codex home"):
                build_codex_sdk_env(env_path, codex_home=root / "missing")


class ClaudeCodeSdkEnvPrecedenceTests(unittest.TestCase):
    def test_explicit_model_and_reasoning_override_env_file_values(self) -> None:
        with TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "MODEL=gpt-5.4-mini\nSKILLFABRIC_LLM_REASONING_EFFORT=medium\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, _cleared_sdk_env(), clear=False):
                env = build_claude_code_sdk_env(
                    env_path,
                    model="gpt-5.6-terra",
                    reasoning_effort="xhigh",
                )

        self.assertEqual(env["ANTHROPIC_MODEL"], "gpt-5.6-terra")
        self.assertEqual(env["ANTHROPIC_REASONING_EFFORT"], "xhigh")

    def test_env_file_overrides_existing_claude_code_shell_values(self) -> None:
        with TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "API_KEY=sk-env",
                        "BASE_URL=http://env-gateway.example/v1",
                        "MODEL=openai/env-model",
                        "SKILLFABRIC_LLM_REASONING_EFFORT=medium",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                _cleared_sdk_env()
                | {
                    "ANTHROPIC_AUTH_TOKEN": "sk-shell",
                    "ANTHROPIC_API_KEY": "sk-shell-api",
                    "ANTHROPIC_BASE_URL": "http://shell-gateway.example",
                    "ANTHROPIC_MODEL": "shell-model",
                    "SKILLFABRIC_LLM_API_KEY": "sk-legacy-shell",
                    "SKILLFABRIC_LLM_API_BASE": "http://legacy-shell.example/v1",
                    "SKILLFABRIC_LLM_MODEL": "openai/legacy-shell-model",
                },
                clear=False,
            ):
                env = build_claude_code_sdk_env(env_path)

        self.assertEqual(env["OPENAI_API_KEY"], "sk-env")
        self.assertEqual(env["OPENAI_BASE_URL"], "http://env-gateway.example/v1")
        self.assertEqual(env["ANTHROPIC_AUTH_TOKEN"], "sk-env")
        self.assertEqual(env["ANTHROPIC_API_KEY"], "sk-env")
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://env-gateway.example")
        self.assertEqual(env["ANTHROPIC_MODEL"], "env-model")
        self.assertEqual(env["ANTHROPIC_REASONING_EFFORT"], "medium")


if __name__ == "__main__":
    unittest.main()
