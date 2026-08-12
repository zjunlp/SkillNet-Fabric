"""Rich terminal presentation for public CLI commands."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text


def print_command_result(command: str, payload: dict[str, Any], *, json_mode: bool) -> None:
    """Print stable JSON or a compact command-specific summary."""

    if json_mode:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    renderers = {
        "build": _print_build,
        "route": _print_route,
        "plan": _print_plan,
        "doctor-state": _print_doctor_state,
        "run-state": _print_run_state,
    }
    try:
        renderer = renderers[command]
    except KeyError as exc:
        raise ValueError(f"unsupported terminal result: {command}") from exc
    renderer(_console(), payload)


@contextmanager
def command_status(message: str, *, enabled: bool) -> Iterator[None]:
    """Show an interactive spinner without polluting redirected output."""

    if not enabled or not sys.stderr.isatty():
        yield
        return
    with Console(stderr=True, highlight=False).status(message, spinner="dots"):
        yield


def _console() -> Console:
    return Console(file=sys.stdout, highlight=False)


def _print_build(console: Console, payload: dict[str, Any]) -> None:
    graph = payload["graph"]
    wiki = payload["artifacts"]["wiki"]
    _title(console, "Build complete")
    _fields(
        console,
        (
            ("Workspace", payload["workspace"]),
            ("Build", payload["build_id"]),
            ("Graph", f"{payload['skill_count']} skills | {graph['edge_count']} edges"),
            ("Wiki", f"{wiki['pages_written']} pages"),
        ),
    )
    _heading(console, "Artifacts")
    _fields(
        console,
        (
            ("Full Wiki", wiki["index"]),
            ("Graph", payload["artifacts"]["graph"]),
            ("Status", payload["artifacts"]["status"]),
        ),
    )


def _print_route(console: Console, payload: dict[str, Any]) -> None:
    selected = payload["selected_skills"]
    _title(console, "Route complete")
    if selected:
        table = Table(
            title="Selected skills",
            box=None,
            pad_edge=False,
            header_style="bold cyan",
        )
        table.add_column("Skill", style="bold")
        table.add_column("Why")
        for skill in selected:
            table.add_row(Text(str(skill["name"])), Text(str(skill["reason"])))
        console.print(table)
    else:
        _field(console, "Selected skills", "None")
    _field(console, "Rationale", payload["rationale"])
    _field(
        console,
        "Evidence",
        f"{len(payload['relation_evidence'])} relations | "
        f"{len(payload['wiki_pages_read'])} wiki pages",
    )
    if payload["coverage_gaps"]:
        _heading(console, "Coverage gaps")
        for gap in payload["coverage_gaps"]:
            console.print(Text(f"- {gap}"), soft_wrap=True)


def _print_plan(console: Console, payload: dict[str, Any]) -> None:
    _title(console, "Execution plan ready")
    _fields(
        console,
        (
            ("Package", payload["root"]),
            ("Prompt", payload["prompt_path"]),
            ("Validation", payload["planner_validation_path"]),
            ("Context", f"{payload['estimated_prompt_tokens']} tokens"),
        ),
    )


def _print_doctor_state(console: Console, payload: dict[str, Any]) -> None:
    _title(console, "SkillFabric status")
    api = "ready" if payload["api_configured"] else "missing configuration"
    workspace = "ready" if payload["workspace_ready"] else "not ready"
    _fields(
        console,
        (
            ("API", api),
            ("Workspace", f"{workspace}: {payload['workspace']}"),
            ("Skills", payload["skill_count"]),
            ("Build", payload["build_id"] or "not built"),
            ("Next", payload["next_action"]),
        ),
    )
    if payload["missing_configuration"]:
        _field(console, "Missing", ", ".join(payload["missing_configuration"]))


def _print_run_state(console: Console, payload: dict[str, Any]) -> None:
    _title(console, "SkillFabric run state")
    fields: list[tuple[str, object]] = [
        ("Action", payload["action"]),
        ("Workspace", payload["workspace"]),
    ]
    for key, label in (
        ("task", "Task"),
        ("prompt_path", "Prompt"),
        ("package_root", "Package"),
    ):
        if payload.get(key):
            fields.append((label, payload[key]))
    _fields(console, fields)


def _title(console: Console, title: str) -> None:
    console.print(Text(title, style="bold green"))


def _heading(console: Console, title: str) -> None:
    console.print(Text(f"\n{title}", style="bold cyan"))


def _fields(
    console: Console,
    fields: Iterable[tuple[str, object]],
) -> None:
    for label, value in fields:
        _field(console, label, value)


def _field(console: Console, label: str, value: object) -> None:
    line = Text()
    line.append(f"{label:<12}", style="bold")
    line.append(str(value))
    console.print(line, soft_wrap=True)


__all__ = ["command_status", "print_command_result"]
