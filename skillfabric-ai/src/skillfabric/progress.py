"""Progress event reporting for public CLI workflows."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TextIO


@dataclass(slots=True)
class ProgressReporter:
    """Emit human or JSONL progress to stderr."""

    enabled: bool = False
    json_mode: bool = False
    quiet: bool = False
    stream: TextIO | None = None

    @contextmanager
    def phase(self, phase: str, *, total: int | None = None) -> Iterator[None]:
        started = time.monotonic()
        self.emit("start", phase, total=total)
        try:
            yield
        except BaseException as exc:
            self.emit(
                "abort",
                phase,
                elapsed_ms=_elapsed_ms(started),
                reason=f"{type(exc).__name__}: {exc}",
            )
            raise
        self.emit("finish", phase, elapsed_ms=_elapsed_ms(started), done=total, total=total)

    def tick(
        self,
        phase: str,
        *,
        done: int | None = None,
        total: int | None = None,
        note: str = "",
    ) -> None:
        self.emit("tick", phase, done=done, total=total, note=note)

    def emit(self, event: str, phase: str, **fields: object) -> None:
        if self.quiet or not self.enabled:
            return
        stream = self.stream or sys.stderr
        payload = {
            "event": event,
            "phase": phase,
            "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            **{key: value for key, value in fields.items() if value is not None and value != ""},
        }
        if self.json_mode:
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        else:
            message = f"[{phase}] {event}"
            note = payload.get("note")
            if note:
                message = f"{message}: {note}"
            stream.write(message + "\n")
        stream.flush()


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1000))
