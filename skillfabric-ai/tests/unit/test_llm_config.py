from __future__ import annotations

import json
import os
import sys
import threading
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from skillfabric.llm import LLMConfig, litellm_completion, llm_usage_context
from skillfabric.llm_jobs import LLMJobOptions, run_llm_jobs


class LLMConfigTests(unittest.TestCase):
    def test_loads_litellm_settings_from_env_file(self) -> None:
        with TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            usage_path = Path(tmp) / "usage.jsonl"
            env_path.write_text(
                "\n".join(
                    [
                        "BASE_URL=https://example.test/api",
                        "API_KEY=sk-test",
                        "MODEL=openai/test-model",
                        "MAX_TOKENS=123",
                        "SKILLFABRIC_LLM_REASONING_EFFORT=medium",
                        "USAGE_ENABLED=1",
                        f"USAGE_LOG_PATH={usage_path}",
                        "USAGE_OPERATION=wiki_build",
                        'USAGE_METADATA={"task_id":"task_a","trace_id":"trace_a"}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            cleared_llm_env = {
                "API_KEY": "",
                "BASE_URL": "",
                "MODEL": "",
                "MAX_TOKENS": "",
                "SKILLFABRIC_LLM_REASONING_EFFORT": "",
                "OPENAI_API_BASE": "",
                "OPENAI_BASE_URL": "",
                "OPENAI_API_KEY": "",
                "USAGE_ENABLED": "",
                "USAGE_LOG_PATH": "",
                "USAGE_OPERATION": "",
                "USAGE_METADATA": "",
            }
            with patch.dict(os.environ, cleared_llm_env, clear=False):
                config = LLMConfig.from_env(env_path=env_path)

            self.assertEqual(config.api_base, "https://example.test/api")
            self.assertEqual(config.api_key, "sk-test")
            self.assertEqual(config.model, "openai/test-model")
            self.assertEqual(config.max_tokens, 123)
            self.assertEqual(config.reasoning_effort, "medium")
            self.assertTrue(config.usage_enabled)
            self.assertEqual(config.usage_log_path, usage_path)
            self.assertEqual(config.usage_operation, "wiki_build")
            self.assertEqual(config.usage_metadata, {"task_id": "task_a", "trace_id": "trace_a"})

    def test_loads_primary_api_settings_from_skillnet_style_env_names(self) -> None:
        with TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "API_KEY=sk-public",
                        "BASE_URL=https://public.example/v1",
                        "MODEL=openai/public-model",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    "API_KEY": "",
                    "BASE_URL": "",
                    "MODEL": "",
                    "OPENAI_API_KEY": "",
                    "OPENAI_BASE_URL": "",
                    "OPENAI_API_BASE": "",
                    "ANTHROPIC_API_KEY": "",
                    "ANTHROPIC_AUTH_TOKEN": "",
                    "ANTHROPIC_BASE_URL": "",
                    "ANTHROPIC_MODEL": "",
                },
                clear=False,
            ):
                config = LLMConfig.from_env(env_path=env_path)

            self.assertEqual(config.api_key, "sk-public")
            self.assertEqual(config.api_base, "https://public.example/v1")
            self.assertEqual(config.model, "openai/public-model")

    def test_loads_anthropic_compatible_settings_from_claude_code_env_names(self) -> None:
        with TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "ANTHROPIC_AUTH_TOKEN=sk-cc-token",
                        "ANTHROPIC_BASE_URL=http://gateway.example",
                        "ANTHROPIC_MODEL=gpt-5.4-mini",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    "API_KEY": "",
                    "BASE_URL": "",
                    "MODEL": "",
                    "OPENAI_API_KEY": "",
                    "OPENAI_BASE_URL": "",
                    "OPENAI_API_BASE": "",
                    "ANTHROPIC_API_KEY": "",
                    "ANTHROPIC_AUTH_TOKEN": "",
                    "ANTHROPIC_BASE_URL": "",
                    "ANTHROPIC_MODEL": "",
                },
                clear=False,
            ):
                config = LLMConfig.from_env(env_path=env_path)

            self.assertEqual(config.api_key, "sk-cc-token")
            self.assertEqual(config.api_base, "http://gateway.example/v1")
            self.assertEqual(config.model, "openai/gpt-5.4-mini")
            self.assertEqual(config.credential_source, "anthropic_auth_token")

    def test_anthropic_endpoint_keeps_claude_model_on_anthropic_provider(self) -> None:
        with TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("", encoding="utf-8")

            with patch.dict(
                os.environ,
                {
                    "API_KEY": "",
                    "BASE_URL": "",
                    "MODEL": "claude-sonnet-4-5",
                    "OPENAI_API_KEY": "",
                    "OPENAI_BASE_URL": "",
                    "OPENAI_API_BASE": "",
                    "ANTHROPIC_API_KEY": "",
                    "ANTHROPIC_AUTH_TOKEN": "sk-cc-token",
                    "ANTHROPIC_BASE_URL": "http://gateway.example",
                    "ANTHROPIC_MODEL": "claude-sonnet-4-5",
                },
                clear=False,
            ):
                config = LLMConfig.from_env(env_path=env_path)

            self.assertEqual(config.api_key, "sk-cc-token")
            self.assertEqual(config.api_base, "http://gateway.example")
            self.assertEqual(config.model, "anthropic/claude-sonnet-4-5")
            self.assertEqual(config.credential_source, "anthropic_auth_token")

    def test_openai_compatible_endpoint_takes_precedence_for_public_llm_calls(self) -> None:
        with TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("", encoding="utf-8")

            with patch.dict(
                os.environ,
                {
                    "API_KEY": "",
                    "BASE_URL": "",
                    "MODEL": "gpt-5.4-mini",
                    "OPENAI_API_KEY": "sk-openai-compatible",
                    "OPENAI_BASE_URL": "http://gateway.example",
                    "OPENAI_API_BASE": "",
                    "ANTHROPIC_API_KEY": "",
                    "ANTHROPIC_AUTH_TOKEN": "sk-cc-token",
                    "ANTHROPIC_BASE_URL": "http://gateway.example",
                    "ANTHROPIC_MODEL": "gpt-5.4-mini",
                },
                clear=False,
            ):
                config = LLMConfig.from_env(env_path=env_path)

            self.assertEqual(config.api_key, "sk-openai-compatible")
            self.assertEqual(config.api_base, "http://gateway.example/v1")
            self.assertEqual(config.model, "openai/gpt-5.4-mini")
            self.assertEqual(config.credential_source, "api_key")

    def test_missing_primary_api_config_points_to_help_config(self) -> None:
        with TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("", encoding="utf-8")

            with patch.dict(
                os.environ,
                {
                    "API_KEY": "",
                    "BASE_URL": "",
                    "OPENAI_API_KEY": "",
                    "OPENAI_BASE_URL": "",
                    "OPENAI_API_BASE": "",
                    "ANTHROPIC_API_KEY": "",
                    "ANTHROPIC_AUTH_TOKEN": "",
                    "ANTHROPIC_BASE_URL": "",
                    "ANTHROPIC_MODEL": "",
                },
                clear=False,
            ):
                with self.assertRaisesRegex(ValueError, "skillfabric help config"):
                    LLMConfig.from_env(env_path=env_path)

    def test_litellm_completion_passes_project_api_config(self) -> None:
        calls: list[dict[str, object]] = []

        fake_litellm = types.SimpleNamespace()

        def fake_completion(**kwargs):
            calls.append(kwargs)
            return {"choices": [{"message": {"content": "ok"}}]}

        fake_litellm.completion = fake_completion
        original = sys.modules.get("litellm")
        sys.modules["litellm"] = fake_litellm
        try:
            response = litellm_completion(
                messages=[{"role": "user", "content": "Hello"}],
                config=LLMConfig(
                    api_base="https://example.test/api",
                    api_key="sk-test",
                    model="openai/test-model",
                    max_tokens=321,
                    reasoning_effort="medium",
                    timeout=15.0,
                ),
            )
        finally:
            if original is None:
                sys.modules.pop("litellm", None)
            else:
                sys.modules["litellm"] = original

        self.assertEqual(response["choices"][0]["message"]["content"], "ok")
        self.assertEqual(calls[0]["model"], "openai/test-model")
        self.assertEqual(calls[0]["api_base"], "https://example.test/api")
        self.assertEqual(calls[0]["api_key"], "sk-test")
        self.assertEqual(calls[0]["max_tokens"], 321)
        self.assertEqual(calls[0]["reasoning_effort"], "medium")
        self.assertEqual(calls[0]["timeout"], 15.0)
        self.assertEqual(calls[0]["request_timeout"], 15.0)
        self.assertEqual(calls[0]["force_timeout"], 15.0)
        self.assertEqual(fake_litellm.request_timeout, 15.0)

    def test_skillfabric_llm_env_names_are_primary_for_public_config(self) -> None:
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

            with patch.dict(
                os.environ,
                {
                    "API_KEY": "",
                    "BASE_URL": "",
                    "MODEL": "",
                    "OPENAI_API_KEY": "",
                    "OPENAI_BASE_URL": "",
                    "OPENAI_API_BASE": "",
                    "ANTHROPIC_API_KEY": "",
                    "ANTHROPIC_AUTH_TOKEN": "",
                    "ANTHROPIC_BASE_URL": "",
                    "ANTHROPIC_MODEL": "",
                    "SKILLFABRIC_LLM_API_BASE": "",
                    "SKILLFABRIC_LLM_API_KEY": "",
                    "SKILLFABRIC_LLM_MODEL": "",
                    "SKILLFABRIC_LLM_REASONING_EFFORT": "",
                },
                clear=False,
            ):
                config = LLMConfig.from_env(env_path=env_path)

            self.assertEqual(config.api_base, "http://gateway.example/v1")
            self.assertEqual(config.api_key, "sk-test")
            self.assertEqual(config.model, "openai/responses/gpt-5.4-mini")
            self.assertEqual(config.reasoning_effort, "medium")

    def test_litellm_completion_folds_system_messages_for_anthropic_provider(self) -> None:
        calls: list[dict[str, object]] = []
        fake_litellm = types.SimpleNamespace()

        def fake_completion(**kwargs):
            calls.append(kwargs)
            return {"choices": [{"message": {"content": "ok"}}]}

        fake_litellm.completion = fake_completion
        original = sys.modules.get("litellm")
        sys.modules["litellm"] = fake_litellm
        try:
            litellm_completion(
                messages=[
                    {"role": "system", "content": "Use strict JSON."},
                    {"role": "user", "content": "Extract this skill."},
                ],
                config=LLMConfig(
                    api_base="http://gateway.example",
                    api_key="sk-test",
                    model="anthropic/gpt-5.4-mini",
                    max_tokens=321,
                    timeout=15.0,
                ),
            )
        finally:
            if original is None:
                sys.modules.pop("litellm", None)
            else:
                sys.modules["litellm"] = original

        sent_messages = calls[0]["messages"]
        self.assertIsInstance(sent_messages, list)
        self.assertEqual([message["role"] for message in sent_messages], ["user"])
        self.assertIn("Use strict JSON.", sent_messages[0]["content"])
        self.assertIn("Extract this skill.", sent_messages[0]["content"])

    def test_litellm_completion_lets_anthropic_auth_token_flow_through_provider_env(self) -> None:
        calls: list[dict[str, object]] = []
        fake_litellm = types.SimpleNamespace()

        def fake_completion(**kwargs):
            calls.append(kwargs)
            return {"choices": [{"message": {"content": "ok"}}]}

        fake_litellm.completion = fake_completion
        original = sys.modules.get("litellm")
        sys.modules["litellm"] = fake_litellm
        try:
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "", "ANTHROPIC_AUTH_TOKEN": ""}, clear=False):
                litellm_completion(
                    messages=[{"role": "user", "content": "Hello"}],
                    config=LLMConfig(
                        api_base="http://gateway.example",
                        api_key="sk-cc-token",
                        model="anthropic/gpt-5.4-mini",
                        credential_source="anthropic_auth_token",
                        max_tokens=321,
                        timeout=15.0,
                    ),
                )
                self.assertEqual(os.environ["ANTHROPIC_AUTH_TOKEN"], "sk-cc-token")
        finally:
            if original is None:
                sys.modules.pop("litellm", None)
            else:
                sys.modules["litellm"] = original

        self.assertIsNone(calls[0]["api_key"])

    def test_litellm_completion_writes_usage_record_for_direct_call(self) -> None:
        calls: list[dict[str, object]] = []
        fake_litellm = types.SimpleNamespace()

        def fake_completion(**kwargs):
            calls.append(kwargs)
            return {"choices": [{"message": {"content": "ok"}}]}

        fake_litellm.completion = fake_completion
        fake_litellm.token_counter = lambda **kwargs: 9 if kwargs.get("messages") else 3
        fake_litellm.cost_per_token = lambda **_kwargs: (0.001, 0.002)
        original = sys.modules.get("litellm")
        sys.modules["litellm"] = fake_litellm
        try:
            with TemporaryDirectory() as tmp:
                usage_path = Path(tmp) / "usage.jsonl"
                response = litellm_completion(
                    messages=[{"role": "user", "content": "Hello"}],
                    config=LLMConfig(
                        api_base="https://example.test/api",
                        api_key="sk-test",
                        model="openai/test-model",
                        max_tokens=321,
                        timeout=15.0,
                        usage_log_path=usage_path,
                        usage_operation="route",
                    ),
                )
                records = [json.loads(line) for line in usage_path.read_text(encoding="utf-8").splitlines()]
        finally:
            if original is None:
                sys.modules.pop("litellm", None)
            else:
                sys.modules["litellm"] = original

        self.assertEqual(response["choices"][0]["message"]["content"], "ok")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["operation"], "route")
        self.assertEqual(records[0]["status"], "completed")
        self.assertGreater(records[0]["prompt_tokens"], 0)
        self.assertGreater(records[0]["completion_tokens"], 0)

    def test_litellm_completion_merges_env_and_call_usage_metadata(self) -> None:
        fake_litellm = types.SimpleNamespace()

        def fake_completion(**_kwargs):
            return {"choices": [{"message": {"content": "ok"}}]}

        fake_litellm.completion = fake_completion
        fake_litellm.token_counter = lambda **kwargs: 9 if kwargs.get("messages") else 3
        fake_litellm.cost_per_token = lambda **_kwargs: (0.001, 0.002)
        original = sys.modules.get("litellm")
        sys.modules["litellm"] = fake_litellm
        try:
            with TemporaryDirectory() as tmp:
                usage_path = Path(tmp) / "usage.jsonl"
                litellm_completion(
                    messages=[{"role": "user", "content": "Hello"}],
                    config=LLMConfig(
                        api_base="https://example.test/api",
                        api_key="sk-test",
                        model="openai/test-model",
                        usage_log_path=usage_path,
                        usage_operation="route",
                        usage_metadata={"task_id": "task_a", "trace_id": "trace_a"},
                    ),
                    usage_metadata={"skill_id": "skill:docx"},
                )
                records = [json.loads(line) for line in usage_path.read_text(encoding="utf-8").splitlines()]
        finally:
            if original is None:
                sys.modules.pop("litellm", None)
            else:
                sys.modules["litellm"] = original

        self.assertEqual(
            records[0]["metadata"],
            {"task_id": "task_a", "trace_id": "trace_a", "skill_id": "skill:docx"},
        )

    def test_litellm_usage_context_overrides_config_usage_target(self) -> None:
        fake_litellm = types.SimpleNamespace()

        def fake_completion(**_kwargs):
            return {"choices": [{"message": {"content": "ok"}}]}

        fake_litellm.completion = fake_completion
        fake_litellm.token_counter = lambda **kwargs: 9 if kwargs.get("messages") else 3
        fake_litellm.cost_per_token = lambda **_kwargs: (0.001, 0.002)
        original = sys.modules.get("litellm")
        sys.modules["litellm"] = fake_litellm
        try:
            with TemporaryDirectory() as tmp:
                usage_path = Path(tmp) / "context_usage.jsonl"
                with llm_usage_context(
                    log_path=usage_path,
                    operation="route",
                    metadata={"task_id": "task_a"},
                ):
                    litellm_completion(
                        messages=[{"role": "user", "content": "Hello"}],
                        config=LLMConfig(
                            api_base="https://example.test/api",
                            api_key="sk-test",
                            model="openai/test-model",
                            usage_log_path=None,
                            usage_operation="config_operation",
                            usage_metadata={"experiment_run_id": "run_1"},
                        ),
                    )
                records = [json.loads(line) for line in usage_path.read_text(encoding="utf-8").splitlines()]
        finally:
            if original is None:
                sys.modules.pop("litellm", None)
            else:
                sys.modules["litellm"] = original

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["operation"], "route")
        self.assertEqual(records[0]["metadata"], {"experiment_run_id": "run_1", "task_id": "task_a"})

    def test_litellm_usage_context_propagates_through_llm_job_threads(self) -> None:
        fake_litellm = types.SimpleNamespace()

        def fake_completion(**_kwargs):
            return {"choices": [{"message": {"content": "ok"}}]}

        fake_litellm.completion = fake_completion
        fake_litellm.token_counter = lambda **kwargs: 9 if kwargs.get("messages") else 3
        fake_litellm.cost_per_token = lambda **_kwargs: (0.001, 0.002)
        original = sys.modules.get("litellm")
        sys.modules["litellm"] = fake_litellm
        try:
            with TemporaryDirectory() as tmp:
                usage_path = Path(tmp) / "threaded_usage.jsonl"
                config = LLMConfig(
                    api_base="https://example.test/api",
                    api_key="sk-test",
                    model="openai/test-model",
                    usage_log_path=None,
                )

                def worker(item: str) -> str:
                    litellm_completion(
                        messages=[{"role": "user", "content": item}],
                        config=config,
                    )
                    return item

                with llm_usage_context(
                    log_path=usage_path,
                    operation="kg_build.interface_extraction",
                    metadata={"experiment_run_id": "run_1"},
                ):
                    outcomes = run_llm_jobs(
                        ["a", "b"],
                        worker,
                        options=LLMJobOptions(concurrency=2, progress_every=0),
                    )
                records = [json.loads(line) for line in usage_path.read_text(encoding="utf-8").splitlines()]
        finally:
            if original is None:
                sys.modules.pop("litellm", None)
            else:
                sys.modules["litellm"] = original

        self.assertEqual([outcome.ok for outcome in outcomes], [True, True])
        self.assertEqual(len(records), 2)
        self.assertEqual(
            {record["operation"] for record in records},
            {"kg_build.interface_extraction"},
        )
        self.assertEqual(
            {record["metadata"]["experiment_run_id"] for record in records},
            {"run_1"},
        )

    def test_litellm_completion_enforces_process_timeout_in_worker_thread(self) -> None:
        errors: list[BaseException] = []

        def call_in_thread() -> None:
            try:
                litellm_completion(
                    messages=[{"role": "user", "content": "Hello"}],
                    config=LLMConfig(
                        api_base="https://example.test/api",
                        api_key="sk-test",
                        model="openai/test-model",
                        timeout=0.05,
                    ),
                    _skillfabric_test_sleep_seconds=5.0,
                )
            except BaseException as exc:  # noqa: BLE001 - test captures timeout type.
                errors.append(exc)

        worker = threading.Thread(target=call_in_thread)
        worker.start()
        worker.join(timeout=2)

        self.assertFalse(worker.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], TimeoutError)

    def test_litellm_completion_records_failure_usage_for_timeout(self) -> None:
        errors: list[BaseException] = []
        with TemporaryDirectory() as tmp:
            usage_path = Path(tmp) / "usage.jsonl"

            def call_in_thread() -> None:
                try:
                    litellm_completion(
                        messages=[{"role": "user", "content": "Hello"}],
                        config=LLMConfig(
                            api_base="https://example.test/api",
                            api_key="sk-test",
                            model="openai/test-model",
                            timeout=0.05,
                            usage_log_path=usage_path,
                            usage_operation="validation",
                        ),
                        _skillfabric_test_sleep_seconds=5.0,
                    )
                except BaseException as exc:  # noqa: BLE001 - test captures timeout type.
                    errors.append(exc)

            worker = threading.Thread(target=call_in_thread)
            worker.start()
            worker.join(timeout=10)

            records = [json.loads(line) for line in usage_path.read_text(encoding="utf-8").splitlines()]

        self.assertFalse(worker.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], "failed")
        self.assertEqual(records[0]["operation"], "validation")
        self.assertGreater(records[0]["prompt_tokens"], 0)
        self.assertEqual(records[0]["completion_tokens"], 0)

    def test_litellm_completion_uses_env_max_tokens_by_default(self) -> None:
        calls: list[dict[str, object]] = []
        fake_litellm = types.SimpleNamespace()

        def fake_completion(**kwargs):
            calls.append(kwargs)
            return {"choices": [{"message": {"content": "ok"}}]}

        fake_litellm.completion = fake_completion
        original = sys.modules.get("litellm")
        sys.modules["litellm"] = fake_litellm
        try:
            with TemporaryDirectory() as tmp:
                env_path = Path(tmp) / ".env"
                env_path.write_text(
                    "\n".join(
                        [
                            "BASE_URL=https://example.test/api",
                            "API_KEY=sk-test",
                            "MODEL=openai/test-model",
                            "MAX_TOKENS=32768",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                litellm_completion(
                    messages=[{"role": "user", "content": "Hello"}],
                    env_path=env_path,
                    usage_operation="llm_smoke",
                )
        finally:
            if original is None:
                sys.modules.pop("litellm", None)
            else:
                sys.modules["litellm"] = original

        self.assertEqual(calls[0]["max_tokens"], 32768)

    def test_litellm_completion_records_usage_when_log_path_is_configured(self) -> None:
        calls: list[dict[str, object]] = []
        fake_litellm = types.SimpleNamespace()

        def fake_completion(**kwargs):
            calls.append(kwargs)
            return {"choices": [{"message": {"content": "ok"}}]}

        fake_litellm.completion = fake_completion
        fake_litellm.token_counter = lambda **kwargs: 10 if kwargs.get("messages") else 2
        fake_litellm.cost_per_token = lambda **_kwargs: (0.001, 0.002)
        original = sys.modules.get("litellm")
        sys.modules["litellm"] = fake_litellm
        try:
            with TemporaryDirectory() as tmp:
                usage_path = Path(tmp) / "usage.jsonl"
                env_path = Path(tmp) / ".env"
                env_path.write_text(
                    "\n".join(
                        [
                            "BASE_URL=https://example.test/api",
                            "API_KEY=sk-test",
                            "MODEL=openai/test-model",
                            f"USAGE_LOG_PATH={usage_path}",
                            "USAGE_OPERATION=custom_smoke",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                response = litellm_completion(
                    messages=[{"role": "user", "content": "Hello"}],
                    env_path=env_path,
                )
                records = [json.loads(line) for line in usage_path.read_text(encoding="utf-8").splitlines()]
        finally:
            if original is None:
                sys.modules.pop("litellm", None)
            else:
                sys.modules["litellm"] = original

        self.assertEqual(response["choices"][0]["message"]["content"], "ok")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["operation"], "custom_smoke")

    def test_litellm_completion_prices_gpt_5_4_mini_server_usage(self) -> None:
        fake_litellm = types.SimpleNamespace()

        def fake_completion(**_kwargs):
            return {
                "choices": [{"message": {"content": "OK"}}],
                "usage": {
                    "prompt_tokens": 5091,
                    "completion_tokens": 5,
                    "total_tokens": 5096,
                    "prompt_tokens_details": {"cached_tokens": 4864},
                },
            }

        fake_litellm.completion = fake_completion
        fake_litellm.token_counter = lambda **kwargs: 10 if kwargs.get("messages") else 2
        fake_litellm.cost_per_token = lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("unknown"))
        original = sys.modules.get("litellm")
        sys.modules["litellm"] = fake_litellm
        try:
            with TemporaryDirectory() as tmp:
                usage_path = Path(tmp) / "usage.jsonl"
                env_path = Path(tmp) / ".env"
                env_path.write_text(
                    "\n".join(
                        [
                            "BASE_URL=https://example.test/api",
                            "API_KEY=sk-test",
                            "MODEL=openai/responses/gpt-5.4-mini",
                            f"USAGE_LOG_PATH={usage_path}",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                litellm_completion(
                    messages=[{"role": "user", "content": "Hello"}],
                    env_path=env_path,
                    usage_operation="llm_smoke",
                )
                records = [json.loads(line) for line in usage_path.read_text(encoding="utf-8").splitlines()]
        finally:
            if original is None:
                sys.modules.pop("litellm", None)
            else:
                sys.modules["litellm"] = original

        expected_cost = ((227 * 0.75) + (4864 * 0.075) + (5 * 4.50)) / 1_000_000
        self.assertAlmostEqual(records[0]["cost_usd"], expected_cost, places=10)
        self.assertEqual(records[0]["cached_prompt_tokens"], 4864)
        self.assertEqual(records[0]["billable_prompt_tokens"], 227)
        self.assertTrue(records[0]["pricing_known"])


if __name__ == "__main__":
    unittest.main()
