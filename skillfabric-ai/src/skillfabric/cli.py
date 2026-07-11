"""SkillFabric public CLI."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "skillfabric-matplotlib"))

from skillfabric.compiled_graph.builder import (
    BuildConfig,
    BuildResult,
    _BuildDependencies,
    build_graph,
)
from skillfabric.indexing.embeddings import (
    ApiEmbeddingProvider,
    DisabledEmbeddingProvider,
    default_embedding_provider,
)
from skillfabric.orchestrator.package import (
    ExecutionPackageResult,
    finalize_execution_package,
    prepare_execution_package,
)
from skillfabric.router.bundle import build_router_bundle
from skillfabric.router.models import RouterBundle, RouterBundleConfig, RouteResult
from skillfabric.router.routing import RouterConfig, route_task
from skillfabric.router.traces import _new_trace_id as _new_agent_trace_id
from skillfabric.router.traces import validate_trace_id
from skillfabric.runtime.defaults import BuildOptions, default_build_options, default_router_options
from skillfabric.runtime.jobs import LLMJobOptions
from skillfabric.runtime.llm import (
    DEFAULT_API_BASE,
    DEFAULT_MODEL,
    llm_usage_context,
    read_env_file,
)
from skillfabric.runtime.metrics import merge_wiki_metrics
from skillfabric.runtime.progress import ProgressReporter
from skillfabric.storage import Workspace, atomic_write_text
from skillfabric.wiki.explorer.skill_package import (
    SkillPackage,
    skill_package_json_schema,
    validate_skill_package_payload,
)
from skillfabric.wiki.explorer.validation import route_from_skill_package, validate_skill_package
from skillfabric.wiki.materializer import build_wiki
from skillfabric.wiki.models import WikiBuildConfig, WikiBuildResult
from skillfabric.wiki.query_wiki import materialize_query_wiki, render_query_wiki_skill_card

INIT_FIELDS = ("API_KEY", "BASE_URL", "MODEL", "EMBEDDING_MODEL")
ENV_ALIASES = {
    "API_KEY": ("API_KEY", "OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
    "BASE_URL": ("BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE", "ANTHROPIC_BASE_URL"),
    "MODEL": ("MODEL", "ANTHROPIC_MODEL"),
    "EMBEDDING_MODEL": ("EMBEDDING_MODEL",),
}

CONFIG_HELP = """SkillFabric API configuration

Primary LLM API:
  API_KEY       OpenAI-compatible API key.
  BASE_URL      OpenAI-compatible base URL, usually ending in /v1.
  MODEL         LiteLLM model id for graph, wiki, and router LLM calls.

Embedding API:
  EMBEDDING_MODEL     Defaults to openai/text-embedding-3-small.
  EMBEDDING_API_KEY   Optional override when embeddings use a different key.
  EMBEDDING_BASE_URL  Optional override when embeddings use a different endpoint.

Standard fallbacks:
  OPENAI_API_KEY
  OPENAI_BASE_URL or OPENAI_API_BASE

Claude Code SDK path:
  ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN
  ANTHROPIC_BASE_URL

Values loaded from --env-file take precedence for SkillFabric commands; shell
environment values are used only for fields missing from the env file.
The public API uses OpenAI-compatible APIs through LiteLLM plus the optional
Claude Code SDK explorer path. Vendor-specific SDKs are not part of the public API.

Run `skillfabric init --env-file .env` to create a private env file. Do not paste
API keys into agent conversation context.
"""

WORKFLOW_HELP = """Recommended SkillFabric workflow

1. Configure API settings:
   skillfabric init --env-file .env

2. Build a graph-backed workspace and wiki:
   skillfabric build --skill-root skills --workspace .skillfabric --env-file .env

3. Route a task to the relevant skills:
   skillfabric route "summarize this repository and identify release risks" --workspace .skillfabric

4. Prepare and finalize a prompt-only execution package through an agent planner:
   skillfabric plan --agent-mode prepare --route-file .skillfabric/runs/<trace>/route.json --workspace .skillfabric

SkillFabric prepares context and execution prompts. It does not execute the final task.
"""


def main(argv: list[str] | None = None) -> None:
    """Run the SkillFabric CLI."""

    argv_list = list(sys.argv[1:] if argv is None else argv)
    parser, command_parsers = _build_parser()
    if argv_list in (["--help"], ["-h"]):
        parser.print_help()
        return
    if argv_list[:1] == ["doctor-state"] and not {"--help", "-h"}.intersection(argv_list[1:]):
        _doctor_state(argparse.Namespace(tokens=argv_list[1:]))
        return
    if argv_list[:1] == ["run-state"] and not {"--help", "-h"}.intersection(argv_list[1:]):
        _run_state(argparse.Namespace(tokens=argv_list[1:]))
        return
    args = parser.parse_args(argv_list)
    if args.command == "help":
        _help(args, command_parsers)
        return
    args.handler(args)


def _build_parser() -> tuple[argparse.ArgumentParser, dict[str, argparse.ArgumentParser]]:
    parser = argparse.ArgumentParser(
        prog="skillfabric",
        description="Build, route, and plan agent skill workflows with SkillFabric.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    command_parsers: dict[str, argparse.ArgumentParser] = {}

    init_parser = subcommands.add_parser("init", help="Configure API settings for SkillFabric")
    init_parser.add_argument("--env-file", default=".env")
    init_parser.add_argument("--check", action="store_true", help="Check configuration without writing")
    init_parser.add_argument("--json", action="store_true", help="Print machine-readable check output")
    init_parser.add_argument("--force", action="store_true", help="Prompt for existing env-file values")
    init_parser.set_defaults(handler=_init)
    command_parsers["init"] = init_parser

    help_parser = subcommands.add_parser("help", help="Show workflow, config, or command help")
    help_parser.add_argument("topic", nargs="?", default="workflow")
    command_parsers["help"] = help_parser

    build_parser = subcommands.add_parser("build", help="Build graph, indexes, execution graph, and wiki")
    build_parser.add_argument("--skill-root", required=True)
    build_parser.add_argument("--workspace", default=".skillfabric")
    build_parser.add_argument("--env-file", default=".env")
    _add_runtime_options(build_parser)
    build_parser.add_argument("--skip-wiki", action="store_true")
    build_parser.add_argument("--wiki-summary-mode", choices=["off", "all"])
    build_parser.add_argument("--llm-concurrency", type=int)
    build_parser.add_argument("--llm-rate-limit-per-minute", type=float)
    build_parser.add_argument("--llm-max-retries", type=int)
    build_parser.add_argument("--llm-retry-backoff-seconds", type=float)
    build_parser.add_argument("--llm-progress-every", type=int)
    build_parser.add_argument("--llm-batch-size", type=int)
    build_parser.add_argument(
        "--embedding-provider",
        choices=["api", "disabled"],
        help=(
            "Embedding provider for dense retrieval. Use api for OpenAI-compatible "
            "embeddings, or disabled when dense retrieval is not needed."
        ),
    )
    build_parser.add_argument(
        "--embedding-model",
        help="OpenAI-compatible embedding model id. Defaults to env EMBEDDING_MODEL.",
    )
    build_parser.set_defaults(handler=_build)
    command_parsers["build"] = build_parser

    route_parser = subcommands.add_parser("route", help="Route a task to selected skills")
    _add_route_options(route_parser)
    route_parser.set_defaults(handler=_route)
    command_parsers["route"] = route_parser

    plan_parser = subcommands.add_parser(
        "plan",
        help="Prepare or finalize a prompt-only execution handoff from a task or route",
    )
    plan_parser.add_argument("query", nargs="?")
    plan_parser.add_argument("--route-file")
    _add_route_options(plan_parser, include_query=False, agent_mode_choices=("prepare", "finalize", "latest"))
    plan_parser.add_argument("--renderer", choices=["claude-code", "codex"])
    plan_parser.add_argument("--planner-output-file", help=argparse.SUPPRESS)
    plan_parser.add_argument("--package-root", help=argparse.SUPPRESS)
    plan_parser.set_defaults(handler=_plan)
    command_parsers["plan"] = plan_parser

    doctor_state_parser = subcommands.add_parser(
        "doctor-state",
        help=argparse.SUPPRESS,
        description="Return non-secret readiness status for the Claude Code /skillfabric:doctor command.",
    )
    doctor_state_parser.add_argument("tokens", nargs=argparse.REMAINDER)
    doctor_state_parser.set_defaults(handler=_doctor_state)
    command_parsers["doctor-state"] = doctor_state_parser

    run_state_parser = subcommands.add_parser(
        "run-state",
        help=argparse.SUPPRESS,
        description="Return the next state for the Claude Code /skillfabric:run command.",
    )
    run_state_parser.add_argument("tokens", nargs=argparse.REMAINDER)
    run_state_parser.set_defaults(handler=_run_state)
    command_parsers["run-state"] = run_state_parser

    query_wiki_parser = subcommands.add_parser("query-wiki", help="Inspect a prepared query_wiki directory")
    query_wiki_subcommands = query_wiki_parser.add_subparsers(dest="query_wiki_command", required=True)
    card_parser = query_wiki_subcommands.add_parser(
        "card",
        help="Print one skill card and generated header without raw Source",
    )
    card_parser.add_argument("query_wiki_root")
    card_parser.add_argument("skill_id")
    card_parser.set_defaults(handler=_query_wiki_card)
    command_parsers["query-wiki"] = query_wiki_parser

    return parser, command_parsers


def _add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--progress-json", action="store_true")
    parser.add_argument("--quiet", action="store_true")


def _add_route_options(
    parser: argparse.ArgumentParser,
    *,
    include_query: bool = True,
    agent_mode_choices: tuple[str, ...] = ("prepare", "finalize"),
) -> None:
    if include_query:
        parser.add_argument("query")
    parser.add_argument("--workspace", default=".skillfabric")
    parser.add_argument("--env-file", default=".env")
    _add_runtime_options(parser)
    parser.add_argument("--max-selected-skills", type=int)
    parser.add_argument("--trace-id")
    parser.add_argument("--skip-llm-router", action="store_true")
    parser.add_argument(
        "--explorer-backend",
        choices=["claude-code", "fallback"],
        default=None,
    )
    parser.add_argument("--explorer-model")
    parser.add_argument("--strict-explorer", action="store_true")
    parser.add_argument("--seed-limit", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument("--expanded-limit", type=int, default=32, help=argparse.SUPPRESS)
    parser.add_argument("--workflow-confidence-threshold", type=float, default=0.95, help=argparse.SUPPRESS)
    parser.add_argument("--max-workflow-hints", type=int, default=12, help=argparse.SUPPRESS)
    parser.add_argument("--agent-mode", choices=agent_mode_choices, help=argparse.SUPPRESS)
    parser.add_argument("--skill-package-file", help=argparse.SUPPRESS)


def _help(args: argparse.Namespace, command_parsers: dict[str, argparse.ArgumentParser]) -> None:
    topic = str(args.topic or "workflow")
    if topic in {"workflow", "quickstart"}:
        print(WORKFLOW_HELP.rstrip())
        return
    if topic == "config":
        print(CONFIG_HELP.rstrip())
        return
    if topic in {"build", "route", "plan", "query-wiki"}:
        command_parsers[topic].print_help()
        return
    raise SystemExit(f"unknown help topic: {topic}")


def _query_wiki_card(args: argparse.Namespace) -> None:
    try:
        print(render_query_wiki_skill_card(args.query_wiki_root, args.skill_id), end="")
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


def _init(args: argparse.Namespace) -> None:
    if args.check:
        payload = _init_check_payload(args.env_file)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return
        if payload["configured"]:
            print(f"SkillFabric API configuration is complete for {payload['env_file']}.")
            return
        print(
            "SkillFabric API configuration is incomplete. "
            f"Missing: {', '.join(payload['missing'])}. "
            f"Run `skillfabric init --env-file {payload['env_file']}`."
        )
        return

    env_path = Path(args.env_file)
    current = read_env_file(env_path)
    updates: dict[str, str] = {}
    if args.force or not current.get("API_KEY"):
        api_key = getpass.getpass("API_KEY (input hidden; leave blank to keep existing): ").strip()
        if api_key:
            updates["API_KEY"] = api_key
    updates["BASE_URL"] = _prompt_env_value(
        "BASE_URL",
        current,
        default=current.get("BASE_URL") or DEFAULT_API_BASE,
        force=args.force,
    )
    updates["MODEL"] = _prompt_env_value(
        "MODEL",
        current,
        default=current.get("MODEL") or DEFAULT_MODEL,
        force=args.force,
    )
    updates["EMBEDDING_MODEL"] = _prompt_env_value(
        "EMBEDDING_MODEL",
        current,
        default=current.get("EMBEDDING_MODEL") or "openai/text-embedding-3-small",
        force=args.force,
    )
    updates = {key: value for key, value in updates.items() if value}
    _write_env_values(env_path, updates)
    os.chmod(env_path, 0o600)
    payload = _init_check_payload(env_path)
    if payload["configured"]:
        print(f"SkillFabric API configuration written to {env_path}.")
    else:
        print(
            "SkillFabric API configuration is still incomplete. "
            f"Missing: {', '.join(payload['missing'])}."
        )


def _prompt_env_value(
    key: str,
    current: dict[str, str],
    *,
    default: str,
    force: bool,
) -> str:
    if current.get(key) and not force:
        return current[key]
    value = input(f"{key} [{default}]: ").strip()
    return value or default


def _init_check_payload(env_file: str | Path) -> dict[str, object]:
    env_path = Path(env_file)
    env_values = read_env_file(env_path)
    sources: dict[str, str] = {}
    for field in INIT_FIELDS:
        sources[field] = _configured_source(env_values, field)
    missing = [field for field in INIT_FIELDS if not sources[field]]
    return {
        "env_file": str(env_path),
        "configured": not missing,
        "missing": missing,
        "present": {field: bool(sources[field]) for field in INIT_FIELDS},
        "sources": sources,
    }


def _configured_source(env_values: dict[str, str], field: str) -> str:
    for key in ENV_ALIASES[field]:
        if env_values.get(key):
            return "env_file"
    for key in ENV_ALIASES[field]:
        if os.environ.get(key):
            return "shell"
    return ""


def _doctor_state(args: argparse.Namespace) -> None:
    print(json.dumps(_doctor_state_payload(args), ensure_ascii=False, indent=2))


def _doctor_state_payload(args: argparse.Namespace) -> dict[str, object]:
    parsed = _parse_state_tokens(getattr(args, "tokens", []))
    env_file = parsed["env_file"]
    workspace = Workspace(parsed["workspace"])
    config = _init_check_payload(env_file)
    status_path = workspace.root / "status.json"
    status_summary = _workspace_status_summary(status_path)
    workspace_ready = status_summary.get("exists") is True and status_summary.get("stage") == "complete"
    api_configured = bool(config["configured"])
    next_command = _doctor_next_command(api_configured, workspace_ready)
    return {
        "cli_available": True,
        "env_file": str(env_file),
        "api_configured": api_configured,
        "missing": config["missing"],
        "present": config["present"],
        "sources": config["sources"],
        "workspace": str(workspace.root),
        "status_path": str(status_path),
        "workspace_ready": workspace_ready,
        "workspace_status": status_summary,
        "next_command": next_command,
        "instructions": [
            "Report only field names, boolean status, paths, and non-secret status metadata.",
            "Do not scan the workspace or print env-file contents.",
            "Do not build, route, plan, or execute from doctor.",
        ],
    }


def _workspace_status_summary(status_path: Path) -> dict[str, object]:
    if not status_path.exists():
        return {"exists": False, "stage": "not_built"}
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"exists": True, "stage": "invalid_json"}
    if not isinstance(payload, dict):
        return {"exists": True, "stage": "invalid_json"}
    stage = _workspace_status_stage(payload)
    return {
        "exists": True,
        "stage": stage,
        "build_id": payload.get("build_id"),
        "skill_count": payload.get("skill_count"),
        "warnings_count": _count_warnings(payload.get("warnings")),
    }


def _workspace_status_stage(payload: dict[str, object]) -> str:
    explicit = payload.get("stage") or payload.get("status")
    if explicit:
        return str(explicit)
    artifacts = payload.get("artifacts")
    if payload.get("build_id") and payload.get("skill_count") is not None and isinstance(artifacts, dict):
        return "complete"
    return "unknown"


def _count_warnings(value: object) -> int:
    return len(value) if isinstance(value, list) else 0


def _doctor_next_command(api_configured: bool, workspace_ready: bool) -> str:
    if not api_configured:
        return "/skillfabric:doctor after configuring .env with skillfabric init --env-file .env"
    if not workspace_ready:
        return "/skillfabric:build"
    return "/skillfabric:prepare or /skillfabric:run"


def _parse_state_tokens(tokens: list[str]) -> dict[str, str]:
    workspace = ".skillfabric"
    env_file = ".env"
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            break
        if token in {"--workspace", "--env-file"}:
            if index + 1 >= len(tokens):
                raise SystemExit(f"{token} requires a value")
            value = tokens[index + 1]
            if token == "--workspace":
                workspace = value
            else:
                env_file = value
            index += 2
            continue
        if token.startswith("--workspace="):
            workspace = token.split("=", 1)[1]
            index += 1
            continue
        if token.startswith("--env-file="):
            env_file = token.split("=", 1)[1]
            index += 1
            continue
        index += 1
    return {"workspace": workspace, "env_file": env_file}


def _write_env_values(env_path: Path, updates: dict[str, str]) -> None:
    env_path.parent.mkdir(parents=True, exist_ok=True)
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    remaining = dict(updates)
    output: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            output.append(line)
            continue
        key, _, _value = line.partition("=")
        normalized = key.strip()
        if normalized in remaining:
            output.append(f"{normalized}={remaining.pop(normalized)}")
        else:
            output.append(line)
    for key in INIT_FIELDS:
        if key in remaining:
            output.append(f"{key}={remaining.pop(key)}")
    for key, value in remaining.items():
        output.append(f"{key}={value}")
    atomic_write_text(env_path, "\n".join(output).rstrip() + "\n")


def _build(args: argparse.Namespace) -> None:
    options = _build_options_from_args(args)
    _preflight_build_api_config(args, options=options)
    reporter = _progress_reporter(args)
    try:
        with reporter.phase("build"):
            graph_result = build_graph(
                BuildConfig(
                    skill_root=args.skill_root,
                    workspace=args.workspace,
                    llm_env_path=args.env_file,
                    llm_options=_llm_job_options_from_args(args, options=options),
                ),
                dependencies=_BuildDependencies(
                    embedding_provider=_embedding_provider_from_args(args, options=options),
                ),
            )
            wiki_result: WikiBuildResult | None = None
            if not args.skip_wiki:
                with llm_usage_context(
                    log_path=graph_result.workspace.reports_dir / "llm_usage.jsonl",
                    metadata={"build_id": graph_result.graph.build_id},
                ):
                    wiki_result = build_wiki(
                        WikiBuildConfig(
                            workspace=graph_result.workspace.root,
                            env_file=args.env_file,
                            use_llm_summaries=_use_llm_wiki_summaries(args, options),
                            llm_concurrency=options.llm_concurrency,
                            llm_rate_limit_per_minute=args.llm_rate_limit_per_minute,
                            llm_max_retries=args.llm_max_retries,
                            llm_retry_backoff_seconds=args.llm_retry_backoff_seconds,
                            llm_progress_every=_llm_progress_every(args),
                            llm_batch_size=options.llm_batch_size,
                        )
                    )
                merge_wiki_metrics(graph_result.workspace, wiki_result)
    except Exception as exc:
        _write_build_failure_status(args.workspace, exc)
        raise
    print(json.dumps(_build_summary(graph_result, wiki_result), ensure_ascii=False, indent=2))


def _write_build_failure_status(workspace_root: str | Path, exc: Exception) -> None:
    workspace = Workspace(workspace_root)
    checkpoint = workspace.read_json(workspace.checkpoint_path, default={}) or {}
    payload = {
        "status": "failed",
        "stage": checkpoint.get("stage") if isinstance(checkpoint, dict) else None,
        "build_id": checkpoint.get("build_id") if isinstance(checkpoint, dict) else None,
        "config_digest": checkpoint.get("config_digest") if isinstance(checkpoint, dict) else None,
        "error_type": type(exc).__name__,
        "error": _safe_error_summary(exc),
    }
    workspace.write_json(workspace.status_path, payload)


def _safe_error_summary(exc: Exception, *, limit: int = 500) -> str:
    text = str(exc).replace("\n", " ").strip()
    if len(text) > limit:
        text = text[: limit - 3].rstrip() + "..."
    return text


def _preflight_build_api_config(args: argparse.Namespace, *, options: BuildOptions) -> None:
    effective_embedding = options.embedding_provider
    payload = _init_check_payload(args.env_file)
    required = ["API_KEY", "BASE_URL", "MODEL"]
    if effective_embedding == "api":
        required.append("EMBEDDING_MODEL")
    missing = [str(item) for item in payload["missing"] if item in required]
    if missing:
        raise SystemExit(
            "missing API configuration: "
            f"{', '.join(missing)}. Run `skillfabric init --env-file {args.env_file}`."
        )


def _build_summary(
    graph_result: BuildResult,
    wiki_result: WikiBuildResult | None,
) -> dict[str, object]:
    workspace = graph_result.workspace
    artifacts: dict[str, object] = {
        "registry": str(workspace.graph_dir / "registry.jsonl"),
        "graph": str(workspace.graph_dir / "compiled.json"),
        "status": str(workspace.status_path),
    }
    if wiki_result is not None:
        artifacts["wiki"] = {
            "index": str(workspace.wiki_dir / "index.md"),
            "health_report": str(workspace.reports_dir / "wiki_health_report.md"),
            "pages_written": wiki_result.pages_written,
        }
    warnings: list[str] = []
    for key, value in graph_result.stats.items():
        if key.endswith("warnings") and isinstance(value, list):
            warnings.extend(str(item) for item in value if item)
        elif key.endswith("warning") and value:
            warnings.append(str(value))
    return {
        "workspace": str(workspace.root),
        "skill_count": int(graph_result.stats.get("skill_count", len(graph_result.skills))),
        "graph": {
            "node_count": len(graph_result.graph.nodes),
            "edge_count": len(graph_result.graph.edges),
        },
        "artifacts": artifacts,
        "cache": {
            "skipped_unchanged": int(graph_result.stats.get("skipped_unchanged", 0) or 0),
            "wiki_summary_cache_hits": int(wiki_result.cache_hits if wiki_result is not None else 0),
        },
        "warnings": warnings,
    }


def _route(args: argparse.Namespace) -> None:
    if args.agent_mode == "prepare":
        print(json.dumps(_route_agent_prepare(args), ensure_ascii=False, indent=2))
        return
    if args.agent_mode == "finalize":
        print(json.dumps(_route_agent_finalize(args).to_dict(), ensure_ascii=False, indent=2))
        return
    result = route_task(_router_config_from_args(args, query=args.query))
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


def _route_agent_prepare(args: argparse.Namespace) -> dict[str, object]:
    options = default_router_options()
    max_selected_skills = max(1, args.max_selected_skills or options.max_selected_skills)
    workspace = Workspace(args.workspace)
    workspace.ensure()
    trace_id = validate_trace_id(args.trace_id or _new_agent_trace_id(args.query))
    trace_dir = workspace.runs_dir / trace_id
    trace_dir.mkdir(parents=True, exist_ok=True)
    bundle = build_router_bundle(
        RouterBundleConfig(
            workspace=workspace.root,
            query=args.query,
            env_file=args.env_file,
            seed_limit=args.seed_limit or options.seed_limit,
            expanded_limit=args.expanded_limit or options.expanded_limit,
            workflow_confidence_threshold=args.workflow_confidence_threshold,
            max_workflow_hints=args.max_workflow_hints,
        )
    )
    query_wiki = materialize_query_wiki(
        workspace,
        bundle,
        trace_dir=trace_dir,
        max_selected_skills=max_selected_skills,
    )
    schema = skill_package_json_schema()
    router_bundle_path = trace_dir / "router_bundle.json"
    request_path = trace_dir / "agent_route_request.json"
    skill_package_path = trace_dir / "agent_skill_package.json"
    atomic_write_text(router_bundle_path, json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2) + "\n")
    request = {
        "task": args.query,
        "trace_id": trace_id,
        "trace_dir": str(trace_dir),
        "query_wiki_root": str(query_wiki.root),
        "explorer_prompt": str(query_wiki.root / "EXPLORER.md"),
        "skill_package_output": str(skill_package_path),
        "max_selected_skills": max_selected_skills,
        "expected_schema": schema,
    }
    atomic_write_text(request_path, json.dumps(request, ensure_ascii=False, indent=2) + "\n")
    return {
        "trace_id": trace_id,
        "trace_dir": str(trace_dir),
        "query_wiki_root": str(query_wiki.root),
        "router_bundle": str(router_bundle_path),
        "agent_route_request": str(request_path),
        "skill_package_file": str(skill_package_path),
        "max_selected_skills": max_selected_skills,
        "expected_schema": schema,
    }


def _route_agent_finalize(args: argparse.Namespace) -> RouteResult:
    if not args.trace_id:
        raise SystemExit("route --agent-mode finalize requires --trace-id")
    if not args.skill_package_file:
        raise SystemExit("route --agent-mode finalize requires --skill-package-file")
    workspace = Workspace(args.workspace)
    trace_id = validate_trace_id(args.trace_id)
    trace_dir = workspace.runs_dir / trace_id
    query_wiki_root = trace_dir / "query_wiki"
    bundle_path = trace_dir / "router_bundle.json"
    if not bundle_path.exists():
        raise SystemExit(f"missing router bundle from prepare phase: {bundle_path}")
    if not query_wiki_root.exists():
        raise SystemExit(f"missing query_wiki from prepare phase: {query_wiki_root}")
    bundle = RouterBundle.from_dict(json.loads(bundle_path.read_text(encoding="utf-8")))
    request_path = trace_dir / "agent_route_request.json"
    if not request_path.exists():
        raise SystemExit(f"missing agent route request from prepare phase: {request_path}")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    prepared_query = str(request.get("task", ""))
    if not prepared_query or prepared_query != bundle.query:
        raise SystemExit("agent route request task does not match the prepared router bundle")
    if args.query != prepared_query:
        raise SystemExit("route --agent-mode finalize query does not match the prepare phase")
    prepared_limit = request.get("max_selected_skills")
    if isinstance(prepared_limit, bool) or not isinstance(prepared_limit, int) or prepared_limit < 1:
        raise SystemExit("invalid max_selected_skills in agent route request")
    if args.max_selected_skills is not None and max(1, args.max_selected_skills) != prepared_limit:
        raise SystemExit(
            "route --agent-mode finalize max-selected-skills conflicts with the prepare phase"
        )
    package_payload = _read_agent_skill_package(args.skill_package_file, trace_dir)
    schema_errors = validate_skill_package_payload(package_payload)
    if schema_errors:
        atomic_write_text(
            trace_dir / "agent_route_validation.json",
            json.dumps(
                {
                    "valid": False,
                    "valid_package": SkillPackage().to_dict(),
                    "errors": schema_errors,
                    "warnings": [],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )
        raise SystemExit(f"invalid SkillPackage schema: {'; '.join(schema_errors)}")
    package = SkillPackage.from_dict(package_payload)
    validation = validate_skill_package(package, query_wiki_root)
    atomic_write_text(
        trace_dir / "agent_route_validation.json",
        json.dumps(validation.to_dict(), ensure_ascii=False, indent=2) + "\n",
    )
    if validation.errors or not validation.valid:
        details = "; ".join(validation.errors) or "SkillPackage selected no valid skills"
        raise SystemExit(f"invalid SkillPackage: {details}")
    warnings = [*bundle.warnings, *validation.warnings]
    result = route_from_skill_package(
        validation.valid_package,
        bundle,
        query=args.query,
        trace_id=trace_id,
        trace_dir=trace_dir,
        warnings=warnings,
        max_selected_skills=prepared_limit,
    )
    atomic_write_text(trace_dir / "route.json", json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n")
    return result


def _plan(args: argparse.Namespace) -> None:
    if args.agent_mode == "latest":
        print(json.dumps(_latest_execution_package_payload(args.workspace), ensure_ascii=False, indent=2))
        return
    if args.agent_mode == "prepare":
        route = _route_from_args_or_file(args)
        prepared = prepare_execution_package(
            args.workspace,
            route,
            renderer=args.renderer or "claude-code",
        )
        print(json.dumps(prepared.to_dict(), ensure_ascii=False, indent=2))
        return
    if args.agent_mode == "finalize":
        if not args.package_root:
            raise SystemExit("plan --agent-mode finalize requires --package-root")
        if not args.planner_output_file:
            raise SystemExit("plan --agent-mode finalize requires --planner-output-file")
        planner_output = _read_planner_output(args.planner_output_file, Path(args.package_root))
        try:
            result = finalize_execution_package(args.package_root, planner_output, renderer=args.renderer)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(json.dumps(_plan_payload(result), ensure_ascii=False, indent=2))
        return
    raise SystemExit(
        "skillfabric plan now requires --agent-mode prepare, --agent-mode finalize, or --agent-mode latest; "
        "direct deterministic planning was removed."
    )


def _run_state(args: argparse.Namespace) -> None:
    print(json.dumps(_run_state_payload(args), ensure_ascii=False, indent=2))


def _run_state_payload(args: argparse.Namespace) -> dict[str, object]:
    parsed = _parse_run_state_tokens(getattr(args, "tokens", []))
    latest = _latest_execution_package_payload(parsed["workspace"])
    query = str(parsed["query"]).strip()
    if latest.get("found"):
        latest_task = str(latest.get("task") or "").strip()
        if query and _normalize_task_text(query) != _normalize_task_text(latest_task):
            return {
                "action": "prepare_required",
                "workspace": latest["workspace"],
                "env_file": parsed["env_file"],
                "prepared_prompt_found": True,
                "existing_prompt_ignored": True,
                "existing_trace_id": latest["trace_id"],
                "existing_task": latest_task,
                "task": query,
                "message": "A finalized prompt exists, but the user supplied a different task. Prepare this task before execution.",
            }
        return {
            "action": "reuse_prompt",
            "workspace": latest["workspace"],
            "env_file": parsed["env_file"],
            "prepared_prompt_found": True,
            "prompt_path": latest["prompt_path"],
            "package_root": latest["package_root"],
            "planner_validation_path": latest["planner_validation_path"],
            "route_file": latest["route_file"],
            "trace_id": latest["trace_id"],
            "task": latest["task"],
            "selected_skills": latest["selected_skills"],
            "instructions": [
                "Read prompt_path before loading native skills, searching, fetching, or answering.",
                "Do not discover .skillfabric/runs with find, grep, rg, ls, or directory scans.",
            ],
        }
    if not query:
        return {
            "action": "missing_task",
            "workspace": str(Workspace(parsed["workspace"]).root),
            "env_file": parsed["env_file"],
            "prepared_prompt_found": False,
            "message": "No finalized execution prompt exists. Ask the user for a task or run /skillfabric:prepare first.",
        }
    return {
        "action": "prepare_required",
        "workspace": str(Workspace(parsed["workspace"]).root),
        "env_file": parsed["env_file"],
        "prepared_prompt_found": False,
        "task": query,
        "message": "No finalized execution prompt exists for reuse. Run the normal prepare pipeline for this task, then execute the finalized prompt.",
    }


def _parse_run_state_tokens(tokens: list[str]) -> dict[str, str]:
    workspace = ".skillfabric"
    env_file = ".env"
    query_parts: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            query_parts.extend(tokens[index + 1 :])
            break
        if token in {"--workspace", "--env-file"}:
            if index + 1 >= len(tokens):
                raise SystemExit(f"run-state {token} requires a value")
            value = tokens[index + 1]
            if token == "--workspace":
                workspace = value
            else:
                env_file = value
            index += 2
            continue
        if token.startswith("--workspace="):
            workspace = token.split("=", 1)[1]
            index += 1
            continue
        if token.startswith("--env-file="):
            env_file = token.split("=", 1)[1]
            index += 1
            continue
        query_parts.append(token)
        index += 1
    return {"workspace": workspace, "env_file": env_file, "query": " ".join(query_parts).strip()}


def _normalize_task_text(text: str) -> str:
    return " ".join(text.split()).casefold()


def _latest_execution_package_payload(workspace_root: str | Path) -> dict[str, object]:
    workspace = Workspace(workspace_root)
    candidates: list[tuple[float, Path]] = []
    if workspace.runs_dir.exists():
        for prompt_path in workspace.runs_dir.glob("*/execution_package/execution_prompt.md"):
            package_root = prompt_path.parent
            validation_path = package_root / "planner_validation.json"
            if _planner_validation_is_valid(validation_path):
                candidates.append((prompt_path.stat().st_mtime, package_root))
    if not candidates:
        return {
            "found": False,
            "workspace": str(workspace.root),
            "message": "no finalized execution package found",
        }

    _mtime, package_root = max(candidates, key=lambda item: item[0])
    route_path = package_root / "route.json"
    route_payload = json.loads(route_path.read_text(encoding="utf-8")) if route_path.exists() else {}
    prompt_path = package_root / "execution_prompt.md"
    return {
        "found": True,
        "workspace": str(workspace.root),
        "package_root": str(package_root),
        "prompt_path": str(prompt_path),
        "planner_validation_path": str(package_root / "planner_validation.json"),
        "route_file": str(route_path),
        "trace_id": str(route_payload.get("trace_id") or package_root.parent.name),
        "task": str(route_payload.get("query") or ""),
        "selected_skills": [
            item.get("skill_id", "")
            for item in route_payload.get("selected_skills", [])
            if isinstance(item, dict) and item.get("skill_id")
        ],
    }


def _planner_validation_is_valid(validation_path: Path) -> bool:
    if not validation_path.exists():
        return False
    try:
        payload = json.loads(validation_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(isinstance(payload, dict) and payload.get("valid") is True)


def _read_agent_skill_package(skill_package_file: str, trace_dir: Path) -> dict[str, object]:
    if skill_package_file == "-":
        raw = sys.stdin.read()
        if not raw.strip():
            raise SystemExit("route --agent-mode finalize received empty SkillPackage JSON on stdin")
        payload = _parse_json_payload(raw)
        atomic_write_text(
            trace_dir / "agent_skill_package.json",
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        )
        return payload
    skill_package_path = _resolve_inside_root(
        trace_dir,
        skill_package_file,
        outside_message="skill package file must be inside trace directory",
    )
    if not skill_package_path.exists():
        raise SystemExit(f"missing skill package file: {skill_package_path}")
    return _parse_json_payload(skill_package_path.read_text(encoding="utf-8"))


def _read_planner_output(planner_output_file: str, package_root: Path) -> dict[str, object]:
    if planner_output_file == "-":
        raw = sys.stdin.read()
        if not raw.strip():
            raise SystemExit("plan --agent-mode finalize received empty planner output JSON on stdin")
        return _parse_json_payload(raw)
    output_path = _resolve_inside_root(
        package_root,
        planner_output_file,
        outside_message="planner output file must be inside package root",
    )
    if not output_path.exists():
        raise SystemExit(f"missing planner output file: {output_path}")
    return _parse_json_payload(output_path.read_text(encoding="utf-8"))


def _parse_json_payload(raw: str) -> dict[str, object]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise SystemExit("expected a JSON object")
    return payload


def _resolve_inside_root(root: Path, path: str | Path, *, outside_message: str) -> Path:
    root_resolved = root.resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate_resolved = candidate.resolve()
        try:
            candidate_resolved.relative_to(root_resolved)
            candidate = candidate_resolved
        except (OSError, ValueError):
            candidate = root / candidate
    try:
        resolved = candidate.resolve()
        resolved.relative_to(root_resolved)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"{outside_message}: {root}") from exc
    return resolved


def _plan_payload(result: ExecutionPackageResult) -> dict[str, object]:
    return result.to_dict()


def _route_from_args_or_file(args: argparse.Namespace) -> RouteResult:
    if args.route_file:
        route_payload = json.loads(Path(args.route_file).read_text(encoding="utf-8"))
        return RouteResult.from_dict(route_payload)
    if not args.query:
        raise SystemExit("plan requires a query or --route-file")
    return route_task(_router_config_from_args(args, query=args.query))


def _router_config_from_args(args: argparse.Namespace, *, query: str) -> RouterConfig:
    options = default_router_options()
    explorer_backend = args.explorer_backend or options.explorer_backend
    use_llm_router = bool(options.use_llm_router)
    if args.skip_llm_router or explorer_backend == "fallback":
        use_llm_router = False
    elif explorer_backend == "claude-code":
        use_llm_router = True
    return RouterConfig(
        workspace=args.workspace,
        query=query,
        env_file=args.env_file,
        use_llm_router=use_llm_router,
        max_selected_skills=args.max_selected_skills or options.max_selected_skills,
        seed_limit=args.seed_limit or options.seed_limit,
        expanded_limit=args.expanded_limit or options.expanded_limit,
        workflow_confidence_threshold=args.workflow_confidence_threshold,
        max_workflow_hints=args.max_workflow_hints,
        trace_id=args.trace_id,
        explorer_backend=explorer_backend,
        explorer_model=args.explorer_model,
        strict_explorer=args.strict_explorer,
    )


def _embedding_provider_from_args(args: argparse.Namespace, *, options: BuildOptions):
    provider = options.embedding_provider
    if provider == "api":
        return ApiEmbeddingProvider.from_env(
            env_path=args.env_file,
            model_id=args.embedding_model,
        )
    if provider == "disabled":
        return DisabledEmbeddingProvider()
    return default_embedding_provider(env_path=args.env_file)


def _build_options_from_args(args: argparse.Namespace) -> BuildOptions:
    defaults = default_build_options()
    env_values = read_env_file(args.env_file)
    env_embedding_provider = (
        args.embedding_provider
        or env_values.get("EMBEDDING_PROVIDER")
        or os.environ.get("EMBEDDING_PROVIDER", "")
        or ""
    ).strip().lower()
    if env_embedding_provider and env_embedding_provider not in {"api", "disabled"}:
        raise SystemExit(
            "unsupported embedding provider: "
            f"{env_embedding_provider}. Use 'api' or 'disabled'."
        )
    embedding_provider = (
        args.embedding_provider
        or env_embedding_provider
        or defaults.embedding_provider
    )
    return BuildOptions(
        embedding_provider=embedding_provider,
        wiki_summary_mode=args.wiki_summary_mode or defaults.wiki_summary_mode,
        llm_concurrency=args.llm_concurrency if args.llm_concurrency is not None else defaults.llm_concurrency,
        llm_batch_size=args.llm_batch_size if args.llm_batch_size is not None else defaults.llm_batch_size,
    )


def _use_llm_wiki_summaries(args: argparse.Namespace, options: BuildOptions) -> bool:
    mode = args.wiki_summary_mode or options.wiki_summary_mode
    return mode == "all"


def _progress_reporter(args: argparse.Namespace) -> ProgressReporter:
    return ProgressReporter(
        enabled=bool(args.progress_json),
        json_mode=bool(args.progress_json),
        quiet=bool(args.quiet),
    )


def _llm_progress_every(args: argparse.Namespace) -> int | None:
    if args.llm_progress_every is not None:
        return args.llm_progress_every
    if args.progress_json or args.quiet:
        return 0
    return None


def _llm_job_options_from_args(args: argparse.Namespace, *, options: BuildOptions) -> LLMJobOptions:
    return LLMJobOptions.from_env(
        env_path=args.env_file,
        concurrency=options.llm_concurrency,
        rate_limit_per_minute=args.llm_rate_limit_per_minute,
        max_retries=args.llm_max_retries,
        retry_backoff_seconds=args.llm_retry_backoff_seconds,
        progress_every=_llm_progress_every(args),
        batch_size=options.llm_batch_size,
    )


if __name__ == "__main__":
    main()
