from __future__ import annotations

import asyncio
import json
import os
import unittest
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from skillfabric.router.models import RouterBundle
from skillfabric.router.routing import RouterConfig, route_task
from skillfabric.router.task_atoms import TaskAtom, TaskDecomposition
from skillfabric.wiki.explorer.backends.claude_code import ClaudeCodeWikiExplorerBackend
from skillfabric.wiki.explorer.prompting import EXPLORER_PROMPT_ID
from skillfabric.wiki.explorer.search_index import load_page_index
from skillfabric.wiki.materializer import build_wiki
from skillfabric.wiki.models import WikiBuildConfig
from tests.unit.wiki_helpers import build_fixture_workspace


@dataclass(slots=True)
class _StubResultMessage:
    structured_output: dict[str, Any] | None
    is_error: bool = False
    result: str = ""
    duration_ms: int = 0
    total_cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    num_turns: int = 0
    subtype: str = ""


@dataclass(slots=True)
class _StubTextBlock:
    text: str


@dataclass(slots=True)
class _StubAssistantMessage:
    content: list[Any]


class _StubSdkRuntime:
    class PermissionResultAllow:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class PermissionResultDeny:
        def __init__(self, message: str = "") -> None:
            self.message = message

    class PermissionUpdate:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class ResultMessage(_StubResultMessage):
        pass

    class AssistantMessage(_StubAssistantMessage):
        pass

    class ClaudeAgentOptions:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    def __init__(self, structured_output: dict[str, Any], metrics: dict[str, Any] | None = None) -> None:
        self.structured_output = structured_output
        self.metrics = metrics or {}
        self.options: Any | None = None
        self.prompts: list[str] = []

    async def query(self, *, prompt: Any, options: Any) -> Any:
        self.options = options
        async for event in prompt:
            self.prompts.append(str(event["message"]["content"]))
        yield self.ResultMessage(structured_output=self.structured_output, **self.metrics)


def _route_test_atoms() -> TaskDecomposition:
    return TaskDecomposition(
        atoms=[
            TaskAtom(
                id="a1",
                kind="action",
                text="parse pdf tables",
                evidence="pdf",
            )
        ]
    )


class WikiExplorerTests(unittest.TestCase):
    def test_build_wiki_writes_page_index_only(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))

            pages = load_page_index(workspace)

            self.assertTrue(pages)
            self.assertTrue(any(page.page_type == "skill" for page in pages))
            self.assertFalse(any(page.path.startswith("debug/") for page in pages))
            self.assertFalse((workspace / "wiki" / "hot.md").exists())

    def test_claude_code_sdk_backend_writes_trace_files(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))
            runtime = _StubSdkRuntime(
                {
                    "selected_skills": [
                        {
                            "skill_id": "skill:pdf-table-parser",
                            "scope": "core",
                            "role": "Parse PDF tables.",
                            "evidence": [{"path": "skills/core/pdf-table-parser.md", "reason": "core page"}],
                        },
                        {
                            "skill_id": "skill:financial-kpi-extractor",
                            "scope": "core",
                            "role": "Extract financial KPIs.",
                            "evidence": [{"path": "skills/core/financial-kpi-extractor.md", "reason": "core page"}],
                        },
                    ],
                    "required_edges": [
                        {
                            "before": "skill:pdf-table-parser",
                            "after": "skill:financial-kpi-extractor",
                            "relation_type": "depend_on",
                            "evidence_path": "edges/bridge_edges.jsonl",
                            "reason": "Parsed tables feed KPI extraction.",
                        }
                    ],
                    "rationale": "Parser feeds extractor.",
                }
            )

            result = route_task(
                RouterConfig(
                    workspace=workspace,
                    query="extract financial KPIs from a PDF report",
                    task_atoms=_route_test_atoms(),
                    trace_id="cc-stub",
                    explorer_backend="claude-code",
                    explorer_model="test-sdk-model",
                ),
                sdk_runtime=runtime,
            )

            trace_dir = workspace / "runs" / "cc-stub"
            self.assertEqual(result.provenance, "claude_code")
            self.assertEqual(runtime.options.cwd.resolve(), (trace_dir / "query_wiki").resolve())
            self.assertEqual(runtime.options.tools, ["Read", "LS", "Glob", "Grep"])
            self.assertEqual(runtime.options.permission_mode, "default")
            self.assertEqual(runtime.options.model, "test-sdk-model")
            self.assertEqual(runtime.options.setting_sources, [])
            self.assertEqual(runtime.options.extra_args, {"disable-slash-commands": None})
            self.assertEqual(runtime.options.max_turns, 24)
            self.assertEqual(runtime.options.load_timeout_ms, 30_000)
            self.assertIsNotNone(runtime.options.stderr)
            self.assertIn("Write", runtime.options.disallowed_tools)
            self.assertIn("json_schema", runtime.options.output_format["type"])
            schema = runtime.options.output_format["schema"]
            selected_item_schema = schema["properties"]["selected_skills"]["items"]
            self.assertFalse(selected_item_schema["additionalProperties"])
            self.assertIn("scope", selected_item_schema["required"])
            self.assertEqual(selected_item_schema["properties"]["scope"]["enum"], ["core", "workflow_bridge", "graph_frontier"])
            self.assertIn("Allowed tools: Read, LS, Glob, Grep.", runtime.options.system_prompt)
            self.assertTrue((trace_dir / "query_wiki" / "manifest.json").exists())
            self.assertTrue((trace_dir / "cc_explorer" / "agent_events.jsonl").exists())
            prompt_contract = json.loads((trace_dir / "cc_explorer" / "prompt_contract.json").read_text(encoding="utf-8"))
            prompt_context = json.loads((trace_dir / "cc_explorer" / "prompt_context.json").read_text(encoding="utf-8"))
            self.assertEqual(prompt_contract["prompt_id"], EXPLORER_PROMPT_ID)
            self.assertEqual(prompt_context["prompt_id"], EXPLORER_PROMPT_ID)
            self.assertEqual(prompt_context["allowed_tools"], ["Read", "LS", "Glob", "Grep"])
            self.assertEqual(prompt_context["tool_budget"]["Read"], 10)
            self.assertEqual(prompt_context["tool_budget"]["total"], 16)
            self.assertNotIn("query", prompt_context)
            self.assertIn(EXPLORER_PROMPT_ID, (trace_dir / "cc_explorer" / "prompt.system.md").read_text(encoding="utf-8"))
            self.assertTrue((trace_dir / "cc_explorer" / "skill_package.json").exists())
            self.assertTrue((trace_dir / "cc_explorer" / "validation.json").exists())

    def test_claude_code_sdk_backend_uses_configurable_runtime_limits(self) -> None:
        with TemporaryDirectory() as tmp:
            query_wiki_root = Path(tmp) / "query_wiki"
            query_wiki_root.mkdir()
            (query_wiki_root / "manifest.json").write_text('{"skills": []}\n', encoding="utf-8")
            trace_dir = Path(tmp) / "trace"
            runtime = _StubSdkRuntime({"selected_skills": []})

            ClaudeCodeWikiExplorerBackend(
                sdk_runtime=runtime,
                max_turns=16,
                load_timeout_ms=45_000,
                execution_timeout_seconds=120.0,
            ).explore(
                query="x",
                query_wiki_root=query_wiki_root,
                bundle=RouterBundle(query="x", selected_skills=[], communities=[], workflow_hints=[], wiki_pages=[]),
                trace_dir=trace_dir,
            )

            self.assertEqual(runtime.options.max_turns, 16)
            self.assertEqual(runtime.options.load_timeout_ms, 45_000)
            prompt_context = json.loads((trace_dir / "cc_explorer" / "prompt_context.json").read_text(encoding="utf-8"))
            self.assertEqual(prompt_context["max_selected_skills"], 8)
            self.assertEqual(prompt_context["tool_budget"]["Read"], 10)

    def test_claude_code_sdk_backend_writes_sdk_usage_metrics(self) -> None:
        with TemporaryDirectory() as tmp:
            query_wiki_root = Path(tmp) / "query_wiki"
            query_wiki_root.mkdir()
            (query_wiki_root / "manifest.json").write_text('{"skills": []}\n', encoding="utf-8")
            trace_dir = Path(tmp) / "trace"
            runtime = _StubSdkRuntime(
                {"selected_skills": []},
                metrics={
                    "duration_ms": 1234,
                    "total_cost_usd": 0.25,
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "cache_creation_input_tokens": 7,
                    "cache_read_input_tokens": 11,
                    "num_turns": 2,
                    "is_error": False,
                    "subtype": "success",
                },
            )

            ClaudeCodeWikiExplorerBackend(sdk_runtime=runtime).explore(
                query="x",
                query_wiki_root=query_wiki_root,
                bundle=RouterBundle(query="x", selected_skills=[], communities=[], workflow_hints=[], wiki_pages=[]),
                trace_dir=trace_dir,
            )

            usage = json.loads((trace_dir / "cc_explorer" / "usage.json").read_text(encoding="utf-8"))
            self.assertEqual(usage["backend"], "claude-code")
            self.assertEqual(usage["runtime"], "claude-agent-sdk")
            self.assertEqual(usage["sdk_metrics"]["total_cost_usd"], 0.25)
            self.assertEqual(usage["sdk_metrics"]["input_tokens"], 100)
            self.assertEqual(usage["sdk_metrics"]["output_tokens"], 20)

    def test_claude_code_sdk_permissions_limit_reads_to_query_wiki(self) -> None:
        with TemporaryDirectory() as tmp:
            query_wiki_root = Path(tmp) / "query_wiki"
            query_wiki_root.mkdir()
            (query_wiki_root / "manifest.json").write_text('{"skills": []}\n', encoding="utf-8")
            trace_dir = Path(tmp) / "trace"
            runtime = _StubSdkRuntime({"selected_skills": []})

            backend = ClaudeCodeWikiExplorerBackend(sdk_runtime=runtime)
            backend.explore(
                query="x",
                query_wiki_root=query_wiki_root,
                bundle=RouterBundle(query="x", selected_skills=[], communities=[], workflow_hints=[], wiki_pages=[]),
                trace_dir=trace_dir,
            )

            allow_result = asyncio.run(runtime.options.can_use_tool("Read", {"file_path": "manifest.json"}, None))
            deny_result = asyncio.run(runtime.options.can_use_tool("Read", {"file_path": "/etc/passwd"}, None))
            write_result = asyncio.run(runtime.options.can_use_tool("Write", {"file_path": "leak.txt"}, None))

            self.assertIsInstance(allow_result, runtime.PermissionResultAllow)
            self.assertIsInstance(deny_result, runtime.PermissionResultDeny)
            self.assertIn("outside allowed read roots", deny_result.message)
            self.assertIsInstance(write_result, runtime.PermissionResultDeny)
            self.assertIn("not allowed", write_result.message)
            self.assertIn("updated_permissions", allow_result.kwargs)

    def test_claude_code_sdk_permissions_enforce_tool_budget(self) -> None:
        with TemporaryDirectory() as tmp:
            query_wiki_root = Path(tmp) / "query_wiki"
            query_wiki_root.mkdir()
            (query_wiki_root / "manifest.json").write_text('{"skills": []}\n', encoding="utf-8")
            trace_dir = Path(tmp) / "trace"
            runtime = _StubSdkRuntime({"selected_skills": []})

            ClaudeCodeWikiExplorerBackend(
                sdk_runtime=runtime,
                tool_budget={"Read": 1, "LS": 4, "Glob": 3, "Grep": 3, "total": 2},
            ).explore(
                query="x",
                query_wiki_root=query_wiki_root,
                bundle=RouterBundle(query="x", selected_skills=[], communities=[], workflow_hints=[], wiki_pages=[]),
                trace_dir=trace_dir,
            )

            first = asyncio.run(runtime.options.can_use_tool("Read", {"file_path": "manifest.json"}, None))
            second = asyncio.run(runtime.options.can_use_tool("Read", {"file_path": "manifest.json"}, None))

            self.assertIsInstance(first, runtime.PermissionResultAllow)
            self.assertIsInstance(second, runtime.PermissionResultDeny)
            self.assertIn("Read<=1", second.message)
            events = (trace_dir / "cc_explorer" / "agent_events.jsonl").read_text(encoding="utf-8")
            self.assertIn("sdk:tool_denied_budget", events)

    def test_claude_code_sdk_backend_passes_env_file_values_to_sdk(self) -> None:
        with TemporaryDirectory() as tmp:
            query_wiki_root = Path(tmp) / "query_wiki"
            query_wiki_root.mkdir()
            (query_wiki_root / "manifest.json").write_text('{"skills": []}\n', encoding="utf-8")
            env_path = Path(tmp) / ".env"
            env_path.write_text("ANTHROPIC_API_KEY=file-token\nCLAUDE_CODE_USE_BEDROCK=1\n", encoding="utf-8")
            trace_dir = Path(tmp) / "trace"
            runtime = _StubSdkRuntime({"selected_skills": []})

            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=True):
                ClaudeCodeWikiExplorerBackend(env_file=env_path, sdk_runtime=runtime).explore(
                    query="x",
                    query_wiki_root=query_wiki_root,
                    bundle=RouterBundle(query="x", selected_skills=[], communities=[], workflow_hints=[], wiki_pages=[]),
                    trace_dir=trace_dir,
                )

            self.assertEqual(runtime.options.env["ANTHROPIC_API_KEY"], "file-token")
            self.assertEqual(runtime.options.env["CLAUDE_CODE_USE_BEDROCK"], "1")

    def test_claude_code_sdk_env_filters_empty_values_and_preserves_auth_token(self) -> None:
        with TemporaryDirectory() as tmp:
            query_wiki_root = Path(tmp) / "query_wiki"
            query_wiki_root.mkdir()
            (query_wiki_root / "manifest.json").write_text('{"skills": []}\n', encoding="utf-8")
            trace_dir = Path(tmp) / "trace"
            runtime = _StubSdkRuntime({"selected_skills": []})

            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "", "ANTHROPIC_AUTH_TOKEN": "auth-token"}, clear=True):
                ClaudeCodeWikiExplorerBackend(env_file=Path(tmp) / "missing.env", sdk_runtime=runtime).explore(
                    query="x",
                    query_wiki_root=query_wiki_root,
                    bundle=RouterBundle(query="x", selected_skills=[], communities=[], workflow_hints=[], wiki_pages=[]),
                    trace_dir=trace_dir,
                )

            self.assertEqual(runtime.options.env["ANTHROPIC_AUTH_TOKEN"], "auth-token")
            self.assertNotIn("ANTHROPIC_API_KEY", runtime.options.env)

    def test_claude_code_sdk_env_derives_anthropic_values_from_skillfabric_env_file(self) -> None:
        with TemporaryDirectory() as tmp:
            query_wiki_root = Path(tmp) / "query_wiki"
            query_wiki_root.mkdir()
            (query_wiki_root / "manifest.json").write_text('{"skills": []}\n', encoding="utf-8")
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "BASE_URL=http://example.test/v1",
                        "API_KEY=sk-test",
                        "MODEL=openai/responses/gpt-5.4-mini",
                        "SKILLFABRIC_LLM_REASONING_EFFORT=medium",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            trace_dir = Path(tmp) / "trace"
            runtime = _StubSdkRuntime({"selected_skills": []})

            with patch.dict(
                os.environ,
                {
                    "ANTHROPIC_AUTH_TOKEN": "sk-old",
                    "ANTHROPIC_API_KEY": "sk-old",
                    "ANTHROPIC_MODEL": "old-model",
                },
                clear=True,
            ):
                ClaudeCodeWikiExplorerBackend(env_file=env_path, sdk_runtime=runtime).explore(
                    query="x",
                    query_wiki_root=query_wiki_root,
                    bundle=RouterBundle(query="x", selected_skills=[], communities=[], workflow_hints=[], wiki_pages=[]),
                    trace_dir=trace_dir,
                )

            self.assertEqual(runtime.options.env["OPENAI_API_KEY"], "sk-test")
            self.assertEqual(runtime.options.env["OPENAI_BASE_URL"], "http://example.test/v1")
            self.assertEqual(runtime.options.env["ANTHROPIC_AUTH_TOKEN"], "sk-old")
            self.assertEqual(runtime.options.env["ANTHROPIC_API_KEY"], "sk-old")
            self.assertEqual(runtime.options.env["ANTHROPIC_MODEL"], "old-model")
            self.assertEqual(runtime.options.env["ANTHROPIC_BASE_URL"], "http://example.test")
            self.assertEqual(runtime.options.effort, "medium")

    def test_claude_code_sdk_backend_runs_inside_existing_event_loop(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))
            runtime = _StubSdkRuntime(
                {
                    "selected_skills": [
                        {
                            "skill_id": "skill:pdf-table-parser",
                            "scope": "core",
                            "role": "Parse PDF tables.",
                            "evidence": [{"path": "skills/core/pdf-table-parser.md", "reason": "core page"}],
                        }
                    ],
                    "rationale": "Parser is sufficient.",
                }
            )

            async def run_route() -> Any:
                return route_task(
                    RouterConfig(
                        workspace=workspace,
                        query="parse pdf tables",
                        task_atoms=_route_test_atoms(),
                        trace_id="cc-running-loop",
                        explorer_backend="claude-code",
                    ),
                    sdk_runtime=runtime,
                )

            result = asyncio.run(run_route())

            self.assertEqual(result.provenance, "claude_code")

    def test_claude_code_sdk_drops_empty_required_edges(self) -> None:
        with TemporaryDirectory() as tmp:
            query_wiki_root = Path(tmp) / "query_wiki"
            query_wiki_root.mkdir()
            (query_wiki_root / "manifest.json").write_text('{"skills": []}\n', encoding="utf-8")
            trace_dir = Path(tmp) / "trace"
            runtime = _StubSdkRuntime({"selected_skills": [], "required_edges": [{"before": "", "after": ""}]})

            package = ClaudeCodeWikiExplorerBackend(sdk_runtime=runtime).explore(
                query="x",
                query_wiki_root=query_wiki_root,
                bundle=RouterBundle(query="x", selected_skills=[], communities=[], workflow_hints=[], wiki_pages=[]),
                trace_dir=trace_dir,
            )

            self.assertEqual(package.required_edges, [])

    def test_claude_code_sdk_normalizes_absolute_evidence_to_query_wiki_relative_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            query_wiki_root = Path(tmp) / "query_wiki"
            skill_page = query_wiki_root / "skills" / "core" / "pdf-table-parser.md"
            workflow_page = query_wiki_root / "workflows" / "parser-to-kpi.md"
            skill_page.parent.mkdir(parents=True)
            workflow_page.parent.mkdir(parents=True)
            skill_page.write_text("# Parser\n", encoding="utf-8")
            workflow_page.write_text("# Workflow\n", encoding="utf-8")
            (query_wiki_root / "manifest.json").write_text(
                json.dumps(
                    {
                        "skills": [
                            {
                                "skill_id": "skill:pdf-table-parser",
                                "scope": "core",
                                "selectable": True,
                                "page_path": "skills/core/pdf-table-parser.md",
                            }
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            trace_dir = Path(tmp) / "trace"
            runtime = _StubSdkRuntime(
                {
                    "selected_skills": [
                        {
                            "skill_id": "skill:pdf-table-parser",
                            "scope": "core",
                            "role": "Parse PDF tables.",
                            "evidence": [{"path": str(skill_page), "reason": "absolute path from SDK"}],
                        }
                    ],
                    "required_edges": [
                        {
                            "before": "skill:pdf-table-parser",
                            "after": "skill:financial-kpi-extractor",
                            "evidence_path": str(workflow_page),
                        }
                    ],
                }
            )

            package = ClaudeCodeWikiExplorerBackend(sdk_runtime=runtime).explore(
                query="x",
                query_wiki_root=query_wiki_root,
                bundle=RouterBundle(query="x", selected_skills=[], communities=[], workflow_hints=[], wiki_pages=[]),
                trace_dir=trace_dir,
            )

            self.assertEqual(package.selected_skills[0].evidence[0].path, "skills/core/pdf-table-parser.md")
            self.assertEqual(package.required_edges[0].evidence_path, "workflows/parser-to-kpi.md")

    def test_claude_code_wrapper_json_result_is_unwrapped(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))
            runtime = _StubSdkRuntime(
                {
                    "result": json.dumps(
                        {
                            "selected_skills": [
                                {
                                    "skill_id": "skill:pdf-table-parser",
                                    "evidence_paths": ["skills/core/pdf-table-parser.md"],
                                    "selection_reason": "Provides the PDF table extraction prerequisite.",
                                },
                                {
                                    "skill_id": "skill:financial-kpi-extractor",
                                    "evidence_paths": ["skills/core/financial-kpi-extractor.md"],
                                    "selection_reason": "Direct match for financial KPI extraction.",
                                },
                            ],
                            "required_edges": [
                                {
                                    "from": "skill:pdf-table-parser",
                                    "to": "skill:financial-kpi-extractor",
                                    "relation": "artifact_compatibility",
                                    "evidence_paths": ["edges/bridge_edges.jsonl"],
                                    "reason": "Parsed tables feed KPI extraction.",
                                }
                            ],
                            "rationale": "Parser feeds extractor.",
                        }
                    )
                }
            )

            result = route_task(
                RouterConfig(
                    workspace=workspace,
                    query="extract financial KPIs from a PDF report",
                    task_atoms=_route_test_atoms(),
                    trace_id="cc-wrapper-json",
                    explorer_backend="claude-code",
                ),
                sdk_runtime=runtime,
            )

            self.assertEqual(result.provenance, "claude_code")
            self.assertEqual(result.selected_skill_ids[:2], ["skill:pdf-table-parser", "skill:financial-kpi-extractor"])
            self.assertEqual(result.selected_skills[0].reason, "Provides the PDF table extraction prerequisite.")
            self.assertTrue(any(edge.edge_type == "artifact_compatibility" for edge in result.required_edges))

    def test_claude_code_assistant_json_text_is_accepted_when_structured_output_missing(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))

            class TextRuntime(_StubSdkRuntime):
                async def query(self, *, prompt: Any, options: Any) -> Any:
                    self.options = options
                    async for event in prompt:
                        self.prompts.append(str(event["message"]["content"]))
                    yield self.AssistantMessage(
                        [
                            _StubTextBlock(
                                json.dumps(
                                    {
                                        "selected_skills": [
                                            {
                                                "skill_id": "skill:pdf-table-parser",
                                                "scope": "core",
                                                "role": "Parse PDF tables.",
                                                "evidence": [
                                                    {
                                                        "path": "skills/core/pdf-table-parser.md",
                                                        "reason": "core page",
                                                    }
                                                ],
                                            }
                                        ],
                                        "required_edges": [],
                                        "ordered_hints": [],
                                        "near_misses": [],
                                        "coverage_notes": [],
                                        "rationale": "Parser covers the task.",
                                    }
                                )
                            )
                        ]
                    )
                    yield self.ResultMessage(structured_output=None)

            result = route_task(
                RouterConfig(
                    workspace=workspace,
                    query="parse pdf tables",
                    task_atoms=_route_test_atoms(),
                    trace_id="cc-assistant-json",
                    explorer_backend="claude-code",
                ),
                sdk_runtime=TextRuntime({}),
            )

            self.assertEqual(result.provenance, "claude_code")
            self.assertEqual(result.selected_skill_ids, ["skill:pdf-table-parser"])
            events = (workspace / "runs" / "cc-assistant-json" / "cc_explorer" / "agent_events.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertIn("text_chars", events)

    def test_claude_code_sdk_error_after_result_preserves_api_error(self) -> None:
        with TemporaryDirectory() as tmp:
            query_wiki_root = Path(tmp) / "query_wiki"
            query_wiki_root.mkdir()
            (query_wiki_root / "manifest.json").write_text('{"skills": []}\n', encoding="utf-8")
            trace_dir = Path(tmp) / "trace"

            class ApiErrorRuntime(_StubSdkRuntime):
                async def query(self, *, prompt: Any, options: Any) -> Any:
                    self.options = options
                    async for event in prompt:
                        self.prompts.append(str(event["message"]["content"]))
                    yield self.AssistantMessage(
                        [_StubTextBlock("API Error: The socket connection was closed unexpectedly.")]
                    )
                    yield self.ResultMessage(
                        structured_output=None,
                        is_error=True,
                        result="success",
                        subtype="success",
                    )
                    raise Exception("Claude Code returned an error result: success")

            with self.assertRaisesRegex(RuntimeError, "socket connection was closed"):
                ClaudeCodeWikiExplorerBackend(sdk_runtime=ApiErrorRuntime({})).explore(
                    query="x",
                    query_wiki_root=query_wiki_root,
                    bundle=RouterBundle(query="x", selected_skills=[], communities=[], workflow_hints=[], wiki_pages=[]),
                    trace_dir=trace_dir,
                )

            error = json.loads((trace_dir / "cc_explorer" / "error.json").read_text(encoding="utf-8"))
            self.assertIn("socket connection was closed", error["error"])
            events = (trace_dir / "cc_explorer" / "agent_events.jsonl").read_text(encoding="utf-8")
            self.assertIn("sdk:query_exception_after_result", events)
            self.assertIn("text_preview", events)
            self.assertIn("result_preview", events)

    def test_claude_code_sdk_retries_transient_api_error(self) -> None:
        with TemporaryDirectory() as tmp:
            query_wiki_root = Path(tmp) / "query_wiki"
            query_wiki_root.mkdir()
            (query_wiki_root / "manifest.json").write_text('{"skills": []}\n', encoding="utf-8")
            trace_dir = Path(tmp) / "trace"

            class FlakyRuntime(_StubSdkRuntime):
                def __init__(self) -> None:
                    super().__init__({"selected_skills": [], "rationale": "No supported skills."})
                    self.calls = 0

                async def query(self, *, prompt: Any, options: Any) -> Any:
                    self.calls += 1
                    self.options = options
                    async for event in prompt:
                        self.prompts.append(str(event["message"]["content"]))
                    if self.calls == 1:
                        yield self.AssistantMessage([_StubTextBlock("API Error: 503 Service temporarily unavailable.")])
                        yield self.ResultMessage(
                            structured_output=None,
                            is_error=True,
                            result="success",
                            subtype="success",
                        )
                        raise Exception("Claude Code returned an error result: success")
                    yield self.ResultMessage(structured_output=self.structured_output)

            runtime = FlakyRuntime()
            package = ClaudeCodeWikiExplorerBackend(sdk_runtime=runtime).explore(
                query="x",
                query_wiki_root=query_wiki_root,
                bundle=RouterBundle(query="x", selected_skills=[], communities=[], workflow_hints=[], wiki_pages=[]),
                trace_dir=trace_dir,
            )

            self.assertEqual(package.selected_skills, [])
            self.assertEqual(runtime.calls, 2)
            self.assertFalse((trace_dir / "cc_explorer" / "error.json").exists())
            events = (trace_dir / "cc_explorer" / "agent_events.jsonl").read_text(encoding="utf-8")
            self.assertIn("backend:retry", events)

    def test_claude_code_sdk_does_not_retry_max_turns(self) -> None:
        with TemporaryDirectory() as tmp:
            query_wiki_root = Path(tmp) / "query_wiki"
            query_wiki_root.mkdir()
            (query_wiki_root / "manifest.json").write_text('{"skills": []}\n', encoding="utf-8")
            trace_dir = Path(tmp) / "trace"

            class MaxTurnsRuntime(_StubSdkRuntime):
                def __init__(self) -> None:
                    super().__init__({})
                    self.calls = 0

                async def query(self, *, prompt: Any, options: Any) -> Any:
                    self.calls += 1
                    self.options = options
                    async for event in prompt:
                        self.prompts.append(str(event["message"]["content"]))
                    yield self.ResultMessage(
                        structured_output=None,
                        is_error=True,
                        result="Reached maximum number of turns (24)",
                        subtype="error_max_turns",
                    )

            runtime = MaxTurnsRuntime()
            with self.assertRaisesRegex(RuntimeError, "maximum number of turns"):
                ClaudeCodeWikiExplorerBackend(sdk_runtime=runtime).explore(
                    query="x",
                    query_wiki_root=query_wiki_root,
                    bundle=RouterBundle(query="x", selected_skills=[], communities=[], workflow_hints=[], wiki_pages=[]),
                    trace_dir=trace_dir,
                )

            self.assertEqual(runtime.calls, 1)
            events = (trace_dir / "cc_explorer" / "agent_events.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("backend:retry", events)

    def test_claude_code_backend_failure_falls_back_and_writes_error(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))

            class BrokenRuntime(_StubSdkRuntime):
                async def query(self, *, prompt: Any, options: Any) -> Any:
                    del prompt, options
                    raise RuntimeError("unavailable")
                    yield

            result = route_task(
                RouterConfig(
                    workspace=workspace,
                    query="parse pdf tables",
                    task_atoms=_route_test_atoms(),
                    trace_id="cc-fallback",
                    explorer_backend="claude-code",
                ),
                sdk_runtime=BrokenRuntime({}),
            )

            trace_dir = workspace / "runs" / "cc-fallback"
            self.assertEqual(result.provenance, "deterministic_fallback")
            self.assertTrue((trace_dir / "query_wiki" / "manifest.json").exists())
            self.assertTrue((trace_dir / "cc_explorer" / "error.json").exists())
            removed_context_name = "route_" + "context.md"
            self.assertFalse((trace_dir / removed_context_name).exists())


if __name__ == "__main__":
    unittest.main()
