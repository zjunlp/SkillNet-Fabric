"""SkillFabric command-line interface."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path
from typing import Any

from skillfabric.compiled_graph.builder import (
    BuildConfig,
    BuildResult,
    _BuildDependencies,
    build_graph,
)
from skillfabric.compiled_graph.models import GRAPH_SCHEMA_VERSION
from skillfabric.indexing.embeddings import ApiEmbeddingProvider
from skillfabric.orchestrator.package import (
    DEFAULT_PLANNER_CONTEXT_MAX_TOKENS,
    plan_execution_package,
)
from skillfabric.router.config import RouterConfig
from skillfabric.router.models import RouteResult
from skillfabric.router.routing import route_task
from skillfabric.router.traces import _new_trace_id, validate_trace_id
from skillfabric.runtime.defaults import default_build_options, default_router_options
from skillfabric.runtime.jobs import LLMJobOptions
from skillfabric.runtime.llm import llm_usage_context, read_env_file
from skillfabric.runtime.metrics import merge_wiki_metrics
from skillfabric.runtime.progress import ProgressReporter
from skillfabric.storage import Workspace, atomic_write_text
from skillfabric.wiki.materializer import build_wiki
from skillfabric.wiki.models import WikiBuildConfig, WikiBuildResult
from skillfabric.wiki.query_wiki import render_query_wiki_skill_card

PUBLIC_COMMANDS = (
    "init",
    "help",
    "build",
    "route",
    "plan",
    "query-wiki",
    "doctor-state",
    "run-state",
)
_CONFIG_ALIASES = {
    "API_KEY": ("API_KEY", "OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
    "BASE_URL": ("BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE", "ANTHROPIC_BASE_URL"),
    "MODEL": ("MODEL", "ANTHROPIC_MODEL"),
    "EMBEDDING_MODEL": ("EMBEDDING_MODEL",),
}


def main(argv: list[str] | None = None) -> None:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if argv_list[:1] == ["doctor-state"] and not {"-h", "--help"}.intersection(argv_list[1:]):
        _doctor_state(argparse.Namespace(tokens=argv_list[1:]))
        return
    if argv_list[:1] == ["run-state"] and not {"-h", "--help"}.intersection(argv_list[1:]):
        _run_state(argparse.Namespace(tokens=argv_list[1:]))
        return
    parser, command_parsers = _build_parser()
    args = parser.parse_args(argv_list)
    if args.command == "help":
        _help(args, command_parsers)
        return
    args.handler(args)


def _build_parser() -> tuple[argparse.ArgumentParser, dict[str, argparse.ArgumentParser]]:
    parser = argparse.ArgumentParser(
        prog="skillfabric",
        description="Build semantic skill graphs, route tasks, and prepare execution prompts.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    commands: dict[str, argparse.ArgumentParser] = {}

    init_parser = subcommands.add_parser("init", help="Configure API settings")
    init_parser.add_argument("--env-file", default=".env")
    init_parser.add_argument("--check", action="store_true")
    init_parser.add_argument("--json", action="store_true")
    init_parser.add_argument("--force", action="store_true")
    init_parser.set_defaults(handler=_init)
    commands["init"] = init_parser

    help_parser = subcommands.add_parser("help", help="Show workflow or command help")
    help_parser.add_argument("topic", nargs="?", default="workflow")
    commands["help"] = help_parser

    build_parser = subcommands.add_parser("build", help="Build graph and wiki artifacts")
    build_parser.add_argument("--skill-root", required=True)
    build_parser.add_argument("--workspace", default=".skillfabric")
    build_parser.add_argument("--env-file", default=".env")
    build_parser.add_argument("--skip-wiki", action="store_true")
    build_parser.add_argument("--wiki-summary-mode", choices=("off", "all"))
    build_parser.add_argument("--embedding-model")
    _add_llm_options(build_parser)
    _add_progress_options(build_parser)
    build_parser.set_defaults(handler=_build)
    commands["build"] = build_parser

    route_parser = subcommands.add_parser("route", help="Route a task through the query wiki")
    route_parser.add_argument("query")
    _add_route_options(route_parser)
    route_parser.set_defaults(handler=_route)
    commands["route"] = route_parser

    plan_parser = subcommands.add_parser("plan", help="Generate one execution prompt")
    plan_parser.add_argument("query", nargs="?")
    plan_parser.add_argument("--route-file")
    plan_parser.add_argument("--workspace", default=".skillfabric")
    plan_parser.add_argument("--env-file", default=".env")
    plan_parser.add_argument("--package-root")
    plan_parser.add_argument(
        "--planner-context-max-tokens",
        type=int,
        default=DEFAULT_PLANNER_CONTEXT_MAX_TOKENS,
    )
    _add_route_tuning(plan_parser)
    plan_parser.set_defaults(handler=_plan)
    commands["plan"] = plan_parser

    query_wiki_parser = subcommands.add_parser("query-wiki", help="Inspect a query wiki")
    query_wiki_commands = query_wiki_parser.add_subparsers(
        dest="query_wiki_command",
        required=True,
    )
    card_parser = query_wiki_commands.add_parser("card", help="Print one bounded skill card")
    card_parser.add_argument("query_wiki_root")
    card_parser.add_argument("skill_id")
    card_parser.set_defaults(handler=_query_wiki_card)
    commands["query-wiki"] = query_wiki_parser

    doctor_parser = subcommands.add_parser(
        "doctor-state",
        help="Report plugin configuration and workspace readiness",
    )
    doctor_parser.add_argument("tokens", nargs=argparse.REMAINDER)
    doctor_parser.set_defaults(handler=_doctor_state)
    commands["doctor-state"] = doctor_parser

    run_parser = subcommands.add_parser(
        "run-state",
        help="Resolve the latest finalized execution package",
    )
    run_parser.add_argument("tokens", nargs=argparse.REMAINDER)
    run_parser.set_defaults(handler=_run_state)
    commands["run-state"] = run_parser
    return parser, commands


def _add_llm_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--llm-concurrency", type=int)
    parser.add_argument("--llm-rate-limit-per-minute", type=float)
    parser.add_argument("--llm-max-retries", type=int)
    parser.add_argument("--llm-retry-backoff-seconds", type=float)
    parser.add_argument("--llm-progress-every", type=int)
    parser.add_argument("--llm-batch-size", type=int)


def _add_progress_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--progress-json", action="store_true")
    parser.add_argument("--quiet", action="store_true")


def _add_route_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", default=".skillfabric")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--trace-id")
    _add_route_tuning(parser)
    _add_progress_options(parser)


def _add_route_tuning(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-selected-skills", type=int)
    parser.add_argument("--seed-limit", type=int)
    parser.add_argument("--expanded-limit", type=int)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--explorer-model")
    parser.add_argument("--explorer-max-turns", type=int)
    parser.add_argument("--explorer-load-timeout-ms", type=int)
    parser.add_argument("--explorer-timeout-seconds", type=float)


def _help(
    args: argparse.Namespace,
    command_parsers: dict[str, argparse.ArgumentParser],
) -> None:
    topic = str(args.topic)
    if topic in {"workflow", "quickstart"}:
        print(
            "1. skillfabric init --env-file .env\n"
            "2. skillfabric build --skill-root skills --workspace .skillfabric --env-file .env\n"
            '3. skillfabric plan "your task" --workspace .skillfabric --env-file .env'
        )
        return
    if topic in command_parsers:
        command_parsers[topic].print_help()
        return
    raise SystemExit(f"unknown help topic: {topic}")


def _init(args: argparse.Namespace) -> None:
    env_path = Path(args.env_file)
    status = _configuration_status(env_path)
    if args.check:
        if args.json:
            print(json.dumps(status, ensure_ascii=False, indent=2))
        else:
            state = "ready" if status["configured"] else "missing configuration"
            print(f"SkillFabric API configuration: {state}")
            if status["missing"]:
                print(f"Missing: {', '.join(status['missing'])}")
        return

    existing = read_env_file(env_path)
    updates: dict[str, str] = {}
    prompts = (
        ("API_KEY", True),
        ("BASE_URL", False),
        ("MODEL", False),
        ("EMBEDDING_MODEL", False),
    )
    for field, secret in prompts:
        if existing.get(field) and not args.force:
            continue
        value = (
            getpass.getpass(f"{field} (input hidden): ") if secret else input(f"{field}: ")
        ).strip()
        if value:
            updates[field] = value
    if updates:
        _write_env_values(env_path, updates)
    result = _configuration_status(env_path)
    if not result["configured"]:
        raise SystemExit(f"configuration incomplete; missing: {', '.join(result['missing'])}")
    print(f"SkillFabric API configuration ready: {env_path}")


def _configuration_status(env_path: Path) -> dict[str, Any]:
    values = read_env_file(env_path)
    present: dict[str, bool] = {}
    sources: dict[str, str] = {}
    for field, aliases in _CONFIG_ALIASES.items():
        file_key = next((key for key in aliases if values.get(key)), None)
        shell_key = next((key for key in aliases if os.environ.get(key)), None)
        present[field] = bool(file_key or shell_key)
        sources[field] = "env_file" if file_key else "environment" if shell_key else "missing"
    missing = [field for field in _CONFIG_ALIASES if not present[field]]
    return {
        "env_file": str(env_path),
        "configured": not missing,
        "present": present,
        "sources": sources,
        "missing": missing,
    }


def _write_env_values(path: Path, updates: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    output: list[str] = []
    remaining = dict(updates)
    for line in lines:
        key, separator, _value = line.partition("=")
        normalized = key.strip()
        if separator and normalized in remaining:
            output.append(f"{normalized}={remaining.pop(normalized)}")
        else:
            output.append(line)
    output.extend(f"{key}={value}" for key, value in remaining.items())
    atomic_write_text(path, "\n".join(output).rstrip() + "\n")
    path.chmod(0o600)


def _build(args: argparse.Namespace) -> None:
    options = default_build_options()
    _require_api_configuration(Path(args.env_file))
    jobs = _llm_options(args)
    reporter = _progress_reporter(args)
    with reporter.phase("build"):
        result = build_graph(
            BuildConfig(
                skill_root=args.skill_root,
                workspace=args.workspace,
                llm_env_path=args.env_file,
                llm_options=jobs,
            ),
            dependencies=_BuildDependencies(
                embedding_provider=ApiEmbeddingProvider.from_env(
                    env_path=args.env_file,
                    model_id=args.embedding_model,
                )
            ),
        )
        wiki_result: WikiBuildResult | None = None
        if not args.skip_wiki:
            mode = args.wiki_summary_mode or options.wiki_summary_mode
            with llm_usage_context(
                log_path=result.workspace.reports_dir / "llm_usage.jsonl",
                metadata={"build_id": result.graph.build_id},
            ):
                wiki_result = build_wiki(
                    WikiBuildConfig(
                        workspace=result.workspace.root,
                        env_file=args.env_file,
                        use_llm_summaries=mode == "all",
                        llm_options=jobs,
                    )
                )
            merge_wiki_metrics(result.workspace, wiki_result)
    print(json.dumps(_build_summary(result, wiki_result), ensure_ascii=False, indent=2))


def _require_api_configuration(env_path: Path) -> None:
    status = _configuration_status(env_path)
    if status["missing"]:
        raise SystemExit(
            "missing API configuration: "
            f"{', '.join(status['missing'])}. Run `skillfabric init --env-file {env_path}`."
        )


def _build_summary(result: BuildResult, wiki: WikiBuildResult | None) -> dict[str, Any]:
    workspace = result.workspace
    artifacts: dict[str, Any] = {
        "registry": str(workspace.graph_dir / "registry.jsonl"),
        "contracts": str(workspace.graph_dir / "contracts.jsonl"),
        "relation_decisions": str(workspace.graph_dir / "relation_decisions.jsonl"),
        "graph": str(workspace.graph_dir / "graph.json"),
        "bm25": str(workspace.graph_dir / "bm25.sqlite"),
        "embeddings": str(workspace.graph_dir / "embeddings.json"),
        "build_summary": str(workspace.reports_dir / "build_summary.json"),
        "llm_usage": str(workspace.reports_dir / "llm_usage.jsonl"),
        "status": str(workspace.status_path),
    }
    if wiki is not None:
        artifacts["wiki"] = {
            "index": str(workspace.wiki_dir / "index.md"),
            "pages_written": wiki.pages_written,
        }
    return {
        "workspace": str(workspace.root),
        "build_id": result.graph.build_id,
        "skill_count": len(result.graph.nodes),
        "graph": {
            "schema_version": result.graph.schema_version,
            "node_count": len(result.graph.nodes),
            "edge_count": len(result.graph.edges),
            "edge_counts": result.stats.get("edge_counts", {}),
        },
        "artifacts": artifacts,
    }


def _route(args: argparse.Namespace) -> None:
    payload = route_task(_router_config(args, query=args.query)).to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _plan(args: argparse.Namespace) -> None:
    route, query, default_root = _plan_route_context(args)
    workspace = Workspace(args.workspace)
    package_root = (
        _inside(workspace.runs_dir, args.package_root) if args.package_root else default_root
    )
    result = plan_execution_package(
        workspace,
        route,
        query=query,
        env_file=args.env_file,
        package_root=package_root,
        planner_context_max_tokens=args.planner_context_max_tokens,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


def _plan_route_context(args: argparse.Namespace) -> tuple[RouteResult, str, Path]:
    workspace = Workspace(args.workspace)
    if args.route_file:
        route_path = _inside(workspace.runs_dir, args.route_file)
        route = RouteResult.from_dict(_read_json_file(route_path))
        query_payload = _read_json_file(route_path.parent / "query.json")
        try:
            prepared_query = _required_artifact_string(
                query_payload.get("query"),
                label="route query artifact",
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if args.query and args.query != prepared_query:
            raise SystemExit("plan query differs from the route query")
        return route, prepared_query, route_path.parent / "execution_package"
    if not args.query:
        raise SystemExit("plan requires a query or --route-file")
    trace_id = _new_trace_id(args.query)
    route = route_task(_router_config(args, query=args.query, trace_id=trace_id))
    trace_dir = _trace_dir(workspace, trace_id)
    return route, args.query, trace_dir / "execution_package"


def _query_wiki_card(args: argparse.Namespace) -> None:
    try:
        print(render_query_wiki_skill_card(args.query_wiki_root, args.skill_id), end="")
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


def _doctor_state(args: argparse.Namespace) -> None:
    parsed = _parse_control_tokens(args.tokens)
    workspace = Workspace(parsed["workspace"])
    config = _configuration_status(Path(parsed["env_file"]))
    status = workspace.read_json(workspace.status_path, default={}) or {}
    ready = bool(
        isinstance(status, dict)
        and status.get("state") == "ready"
        and status.get("schema_version") == GRAPH_SCHEMA_VERSION
    )
    stats = status.get("stats", {}) if isinstance(status, dict) else {}
    payload = {
        "api_configured": config["configured"],
        "missing_configuration": config["missing"],
        "workspace": str(workspace.root),
        "workspace_ready": ready,
        "build_id": status.get("build_id", "") if isinstance(status, dict) else "",
        "skill_count": stats.get("skill_count", 0) if isinstance(stats, dict) else 0,
        "next_action": "ready"
        if config["configured"] and ready
        else "build"
        if config["configured"]
        else "init",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _run_state(args: argparse.Namespace) -> None:
    parsed = _parse_control_tokens(args.tokens)
    latest = _latest_execution_package(parsed["workspace"])
    query = parsed["query"]
    if latest.get("found"):
        previous = latest["task"]
        if query and _normalize_text(query) != _normalize_text(previous):
            payload = {
                "action": "prepare_required",
                "workspace": latest["workspace"],
                "env_file": parsed["env_file"],
                "task": query,
                "existing_task": previous,
            }
        else:
            payload = {
                "action": "reuse_prompt",
                "workspace": latest["workspace"],
                "env_file": parsed["env_file"],
                "task": previous,
                "prompt_path": latest["prompt_path"],
                "package_root": latest["package_root"],
                "selected_skills": latest["selected_skills"],
            }
    elif query:
        payload = {
            "action": "prepare_required",
            "workspace": str(Workspace(parsed["workspace"]).root),
            "env_file": parsed["env_file"],
            "task": query,
        }
    else:
        payload = {
            "action": "missing_task",
            "workspace": str(Workspace(parsed["workspace"]).root),
            "env_file": parsed["env_file"],
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _latest_execution_package(workspace_root: str | Path) -> dict[str, Any]:
    workspace = Workspace(workspace_root)
    candidates: list[tuple[float, Path]] = []
    if workspace.runs_dir.exists():
        candidates.extend(
            (prompt.stat().st_mtime, prompt.parent)
            for prompt in workspace.runs_dir.glob("*/execution_package/execution_prompt.md")
            if _is_finalized_execution_package(prompt.parent)
        )
    if not candidates:
        return {"found": False, "workspace": str(workspace.root)}
    _mtime, root = max(candidates, key=lambda item: item[0])
    request = _read_json_file(root / "planner_request.json")
    task = _required_artifact_string(
        request.get("task"),
        label="planner request task",
    )
    try:
        route = RouteResult.from_dict(_read_json_file(root / "route.json"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid route result in {root / 'route.json'}: {exc}") from exc
    return {
        "found": True,
        "workspace": str(workspace.root),
        "package_root": str(root),
        "prompt_path": str(root / "execution_prompt.md"),
        "planner_validation_path": str(root / "planner_validation.json"),
        "route_file": str(root / "route.json"),
        "trace_id": root.parent.name,
        "task": task,
        "selected_skills": list(route.selected_skill_ids),
    }


def _is_finalized_execution_package(root: Path) -> bool:
    required = (
        root / "execution_prompt.md",
        root / "planner_output.json",
        root / "planner_validation.json",
        root / "planner_request.json",
        root / "route.json",
    )
    return all(path.is_file() for path in required) and _valid_planner_validation(
        root / "planner_validation.json"
    )


def _valid_planner_validation(path: Path) -> bool:
    try:
        payload = _read_json_file(path)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return False
    return (
        set(payload) == {"valid", "errors"}
        and payload.get("valid") is True
        and payload.get("errors") == []
    )


def _router_config(
    args: argparse.Namespace,
    *,
    query: str,
    trace_id: str | None = None,
) -> RouterConfig:
    defaults = default_router_options()
    return RouterConfig(
        workspace=args.workspace,
        query=query,
        env_file=args.env_file,
        trace_id=trace_id if trace_id is not None else getattr(args, "trace_id", None),
        max_selected_skills=(
            defaults.max_selected_skills
            if args.max_selected_skills is None
            else args.max_selected_skills
        ),
        seed_limit=defaults.seed_limit if args.seed_limit is None else args.seed_limit,
        expanded_limit=(
            defaults.expanded_limit if args.expanded_limit is None else args.expanded_limit
        ),
        max_depth=defaults.max_depth if args.max_depth is None else args.max_depth,
        explorer_model=args.explorer_model,
        explorer_max_turns=(
            defaults.explorer_max_turns
            if args.explorer_max_turns is None
            else args.explorer_max_turns
        ),
        explorer_load_timeout_ms=(
            defaults.explorer_load_timeout_ms
            if args.explorer_load_timeout_ms is None
            else args.explorer_load_timeout_ms
        ),
        explorer_timeout_seconds=(
            defaults.explorer_timeout_seconds
            if args.explorer_timeout_seconds is None
            else args.explorer_timeout_seconds
        ),
    )


def _llm_options(args: argparse.Namespace) -> LLMJobOptions:
    return LLMJobOptions.from_env(
        env_path=args.env_file,
        concurrency=args.llm_concurrency,
        rate_limit_per_minute=args.llm_rate_limit_per_minute,
        max_retries=args.llm_max_retries,
        retry_backoff_seconds=args.llm_retry_backoff_seconds,
        progress_every=args.llm_progress_every,
        batch_size=args.llm_batch_size,
    )


def _progress_reporter(args: argparse.Namespace) -> ProgressReporter:
    return ProgressReporter(
        enabled=bool(args.progress_json),
        json_mode=bool(args.progress_json),
        quiet=bool(args.quiet),
    )


def _read_json_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _inside(root: Path, value: str | Path) -> Path:
    root = root.resolve()
    path = Path(value)
    candidate = (root / path if not path.is_absolute() else path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"path must stay inside {root}: {value}") from exc
    return candidate


def _trace_dir(workspace: Workspace, trace_id: str) -> Path:
    return _inside(
        workspace.runs_dir,
        workspace.runs_dir / validate_trace_id(trace_id),
    )


def _parse_control_tokens(tokens: list[str]) -> dict[str, str]:
    workspace = ".skillfabric"
    env_file = ".env"
    query: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in {"--workspace", "--env-file"}:
            if index + 1 >= len(tokens):
                raise SystemExit(f"{token} requires a value")
            if token == "--workspace":
                workspace = tokens[index + 1]
            else:
                env_file = tokens[index + 1]
            index += 2
            continue
        if token.startswith("--workspace="):
            workspace = token.split("=", 1)[1]
        elif token.startswith("--env-file="):
            env_file = token.split("=", 1)[1]
        else:
            query.append(token)
        index += 1
    return {
        "workspace": workspace,
        "env_file": env_file,
        "query": " ".join(query).strip(),
    }


def _normalize_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def _required_artifact_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{label} must contain a non-empty string query"
            if label == "route query artifact"
            else f"{label} must be a non-empty string"
        )
    return value


if __name__ == "__main__":  # pragma: no cover
    main()
