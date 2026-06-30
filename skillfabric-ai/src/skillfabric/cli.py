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

from skillfabric.compiled_graph.builder import BuildConfig, BuildResult, build_graph
from skillfabric.indexing.embeddings import (
    ApiEmbeddingProvider,
    DisabledEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    default_embedding_provider,
)
from skillfabric.llm import (
    DEFAULT_API_BASE,
    DEFAULT_MODEL,
    llm_usage_context,
    read_env_file,
)
from skillfabric.metrics import merge_wiki_metrics
from skillfabric.orchestrator.package import (
    ExecutionPackageResult,
    build_execution_package,
    finalize_execution_package,
    prepare_execution_package,
)
from skillfabric.orchestrator.renderers.claude_code import render_claude_code_entry_prompt
from skillfabric.orchestrator.renderers.codex import render_codex_entry_prompt
from skillfabric.progress import ProgressReporter
from skillfabric.router.bundle import build_router_bundle
from skillfabric.router.models import RouterBundle, RouterBundleConfig, RouteResult
from skillfabric.router.routing import RouterConfig, route_task
from skillfabric.router.traces import _new_trace_id as _new_agent_trace_id
from skillfabric.runtime_defaults import BuildOptions, default_build_options, default_router_options
from skillfabric.storage import Workspace, atomic_write_text
from skillfabric.wiki.explorer.skill_package import SkillPackage, skill_package_json_schema
from skillfabric.wiki.explorer.validation import route_from_skill_package, validate_skill_package
from skillfabric.wiki.materializer import build_wiki
from skillfabric.wiki.models import WikiBuildConfig, WikiBuildResult
from skillfabric.wiki.query_wiki import materialize_query_wiki, render_query_wiki_skill_card

PUBLIC_COMMANDS = ("init", "help", "build", "route", "plan", "query-wiki")
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

Local embeddings:
  EMBEDDING_PROVIDER=local
  EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
  EMBEDDING_MODEL_PATH=/path/to/local/sentence-transformer

Standard fallbacks:
  OPENAI_API_KEY
  OPENAI_BASE_URL or OPENAI_API_BASE

Claude Code SDK path:
  ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN
  ANTHROPIC_BASE_URL

Shell environment values take precedence over values loaded from --env-file.
SkillFabric v1 commits to OpenAI-compatible APIs through LiteLLM plus the optional
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
   skillfabric route "extract financial KPIs from a PDF report" --workspace .skillfabric

4. Plan the execution workflow and prompt for Claude Code or Codex:
   skillfabric plan "extract financial KPIs from a PDF report" --workspace .skillfabric

SkillFabric prepares context and execution prompts. It does not execute the final task.
"""


def main(argv: list[str] | None = None) -> None:
    """Run the SkillFabric CLI."""

    argv_list = list(sys.argv[1:] if argv is None else argv)
    parser, command_parsers = _build_parser()
    if argv_list in (["--help"], ["-h"]):
        parser.print_help()
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
    build_parser.add_argument("--skip-llm-validation", action="store_true", help=argparse.SUPPRESS)
    build_parser.add_argument("--wiki-summary-mode", choices=["off", "all"])
    build_parser.add_argument("--similar-top-k", type=int)
    build_parser.add_argument("--candidate-top-k", type=int)
    build_parser.add_argument("--llm-concurrency", type=int)
    build_parser.add_argument("--llm-rate-limit-per-minute", type=float)
    build_parser.add_argument("--llm-max-retries", type=int)
    build_parser.add_argument("--llm-retry-backoff-seconds", type=float)
    build_parser.add_argument("--llm-progress-every", type=int)
    build_parser.add_argument("--llm-batch-size", type=int)
    build_parser.add_argument(
        "--embedding-provider",
        choices=["api", "local", "disabled"],
        help=(
            "Embedding provider for dense retrieval. Use api for OpenAI-compatible "
            "embeddings, local for SentenceTransformers, or disabled for deterministic smoke checks."
        ),
    )
    build_parser.add_argument(
        "--embedding-model",
        help=(
            "Embedding model id. Defaults to env EMBEDDING_MODEL for api, or "
            "BAAI/bge-large-en-v1.5 for local embeddings."
        ),
    )
    build_parser.add_argument(
        "--embedding-model-path",
        help="Local SentenceTransformers model path. Implies --embedding-provider local.",
    )
    build_parser.set_defaults(handler=_build)
    command_parsers["build"] = build_parser

    route_parser = subcommands.add_parser("route", help="Route a task to selected skills")
    _add_route_options(route_parser)
    route_parser.set_defaults(handler=_route)
    command_parsers["route"] = route_parser

    plan_parser = subcommands.add_parser(
        "plan",
        help="Plan a Claude Code/Codex workflow and execution prompt from a task or route",
    )
    plan_parser.add_argument("query", nargs="?")
    plan_parser.add_argument("--route-file")
    _add_route_options(plan_parser, include_query=False)
    plan_parser.add_argument("--renderer", choices=["claude-code", "codex"], default="claude-code")
    plan_parser.add_argument("--planner-output-file", help=argparse.SUPPRESS)
    plan_parser.add_argument("--package-root", help=argparse.SUPPRESS)
    plan_parser.set_defaults(handler=_plan)
    command_parsers["plan"] = plan_parser

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


def _add_route_options(parser: argparse.ArgumentParser, *, include_query: bool = True) -> None:
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
    parser.add_argument("--expanded-limit", type=int, default=50, help=argparse.SUPPRESS)
    parser.add_argument("--workflow-confidence-threshold", type=float, default=0.95, help=argparse.SUPPRESS)
    parser.add_argument("--max-workflow-hints", type=int, default=12, help=argparse.SUPPRESS)
    parser.add_argument("--agent-mode", choices=["prepare", "finalize"], help=argparse.SUPPRESS)
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
        if os.environ.get(key):
            return "shell"
    for key in ENV_ALIASES[field]:
        if env_values.get(key):
            return "env_file"
    return ""


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
                    similar_top_k=options.similar_top_k,
                    candidate_top_k=options.candidate_top_k,
                    llm_env_path=args.env_file,
                    skip_llm_validation=options.skip_llm_validation,
                    llm_concurrency=options.llm_concurrency,
                    llm_rate_limit_per_minute=args.llm_rate_limit_per_minute,
                    llm_max_retries=args.llm_max_retries,
                    llm_retry_backoff_seconds=args.llm_retry_backoff_seconds,
                    llm_progress_every=_llm_progress_every(args),
                    llm_batch_size=options.llm_batch_size,
                    embedding_provider=_embedding_provider_from_args(args, options=options),
                )
            )
            wiki_result: WikiBuildResult | None = None
            if not args.skip_wiki:
                with llm_usage_context(
                    log_path=graph_result.workspace.root / "llm_usage.jsonl",
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
    effective_skip_llm = options.skip_llm_validation
    if effective_skip_llm and effective_embedding in {"disabled", "local"}:
        return
    payload = _init_check_payload(args.env_file)
    required = ["API_KEY", "BASE_URL"]
    if not effective_skip_llm:
        required.append("MODEL")
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
        "registry": str(workspace.registry_dir / "skills.jsonl"),
        "graph": str(workspace.graph_dir / "compiled_skill_graph.json"),
        "status": str(workspace.status_path),
    }
    if wiki_result is not None:
        artifacts["wiki"] = {
            "index": str(workspace.wiki_dir / "index.md"),
            "health_report": str(workspace.wiki_dir / "wiki_health_report.md"),
            "pages_written": wiki_result.pages_written,
        }
    warnings = [
        str(value)
        for key, value in graph_result.stats.items()
        if key.endswith("warning") and value
    ]
    return {
        "workspace": str(workspace.root),
        "skill_count": int(graph_result.stats.get("skill_count", len(graph_result.skills))),
        "graph": {
            "node_count": len(graph_result.graph.nodes),
            "edge_count": len(graph_result.graph.edges),
            "community_count": len(graph_result.communities),
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
    workspace = Workspace(args.workspace)
    workspace.ensure()
    trace_id = args.trace_id or _new_agent_trace_id(args.query)
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
    query_wiki = materialize_query_wiki(workspace, bundle, trace_dir=trace_dir)
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
        "expected_schema": schema,
    }


def _route_agent_finalize(args: argparse.Namespace) -> RouteResult:
    options = default_router_options()
    if not args.trace_id:
        raise SystemExit("route --agent-mode finalize requires --trace-id")
    if not args.skill_package_file:
        raise SystemExit("route --agent-mode finalize requires --skill-package-file")
    workspace = Workspace(args.workspace)
    trace_dir = workspace.runs_dir / args.trace_id
    query_wiki_root = trace_dir / "query_wiki"
    bundle_path = trace_dir / "router_bundle.json"
    if not bundle_path.exists():
        raise SystemExit(f"missing router bundle from prepare phase: {bundle_path}")
    if not query_wiki_root.exists():
        raise SystemExit(f"missing query_wiki from prepare phase: {query_wiki_root}")
    bundle = RouterBundle.from_dict(json.loads(bundle_path.read_text(encoding="utf-8")))
    package_payload = _read_agent_skill_package(args.skill_package_file, trace_dir)
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
        trace_id=args.trace_id,
        trace_dir=trace_dir,
        warnings=warnings,
        max_selected_skills=max(1, args.max_selected_skills or options.max_selected_skills),
    )
    atomic_write_text(trace_dir / "route.json", json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n")
    return result


def _plan(args: argparse.Namespace) -> None:
    if args.agent_mode == "prepare":
        route = _route_from_args_or_file(args)
        prepared = prepare_execution_package(args.workspace, route, renderer=args.renderer)
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
    route = _route_from_args_or_file(args)
    result = build_execution_package(args.workspace, route, renderer=args.renderer)
    print(json.dumps(_plan_payload(result), ensure_ascii=False, indent=2))


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
        candidate = root / candidate
    try:
        resolved = candidate.resolve()
        resolved.relative_to(root_resolved)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"{outside_message}: {root}") from exc
    return resolved


def _plan_payload(result: ExecutionPackageResult) -> dict[str, object]:
    if result.renderer == "codex":
        entry_prompt = render_codex_entry_prompt(result.spec, execution_package_root=result.root)
    else:
        entry_prompt = render_claude_code_entry_prompt(result.spec, execution_package_root=result.root)
    return {
        **result.to_dict(),
        "entry_prompt": entry_prompt.to_dict(),
    }


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
    if not provider and args.embedding_model_path:
        provider = "local"
    if provider == "api":
        return ApiEmbeddingProvider.from_env(
            env_path=args.env_file,
            model_id=args.embedding_model,
        )
    if provider == "local":
        return SentenceTransformerEmbeddingProvider(
            model_path=args.embedding_model_path,
            model_id=args.embedding_model or "BAAI/bge-large-en-v1.5",
        )
    if provider == "disabled":
        return DisabledEmbeddingProvider()
    return default_embedding_provider(env_path=args.env_file)


def _build_options_from_args(args: argparse.Namespace) -> BuildOptions:
    defaults = default_build_options()
    embedding_provider = args.embedding_provider or ("local" if args.embedding_model_path else defaults.embedding_provider)
    return BuildOptions(
        skip_llm_validation=bool(args.skip_llm_validation or defaults.skip_llm_validation),
        embedding_provider=embedding_provider,
        wiki_summary_mode=args.wiki_summary_mode or defaults.wiki_summary_mode,
        similar_top_k=args.similar_top_k if args.similar_top_k is not None else defaults.similar_top_k,
        candidate_top_k=args.candidate_top_k if args.candidate_top_k is not None else defaults.candidate_top_k,
        llm_concurrency=args.llm_concurrency if args.llm_concurrency is not None else defaults.llm_concurrency,
        llm_batch_size=args.llm_batch_size if args.llm_batch_size is not None else defaults.llm_batch_size,
    )


def _use_llm_wiki_summaries(args: argparse.Namespace, options: BuildOptions) -> bool:
    mode = args.wiki_summary_mode or options.wiki_summary_mode
    return mode == "all" and not options.skip_llm_validation


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


if __name__ == "__main__":
    main()
