"""Incremental, fail-closed JSON object cache checkpoints."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from skillfabric.storage.workspace import atomic_write_text

_CHECKPOINT_NAME = re.compile(r"^(?P<sequence>[0-9]{8})\.json$")
_CHECKPOINT_TEMP_NAME = re.compile(r"^[0-9]{8}\.json\.tmp$")


class CheckpointCacheError(RuntimeError):
    """Raised when a canonical cache or checkpoint shard is invalid."""


class JsonObjectCheckpointCache:
    """Persist JSON object entries in bounded shards before final compaction."""

    def __init__(self, path: str | Path | None, *, interval: int) -> None:
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
            raise ValueError("checkpoint interval must be a positive integer")
        self.path = None if path is None else Path(path)
        self.interval = interval
        self._entries: dict[str, dict[str, Any]] = {}
        self._pending: dict[str, dict[str, Any]] = {}
        self._next_sequence = 1

    @property
    def checkpoint_dir(self) -> Path | None:
        if self.path is None:
            return None
        return self.path.with_name(f"{self.path.stem}.checkpoints")

    def load(self) -> dict[str, dict[str, Any]]:
        """Load and merge the canonical cache and all committed checkpoint shards."""

        self._entries = {}
        self._pending = {}
        self._next_sequence = 1
        if self.path is not None and self.path.exists():
            self._merge(_read_json_object(self.path, label="canonical cache"), self.path)

        checkpoint_dir = self.checkpoint_dir
        try:
            if checkpoint_dir is None or not checkpoint_dir.exists():
                return dict(self._entries)
            if not checkpoint_dir.is_dir():
                raise CheckpointCacheError(f"checkpoint path must be a directory: {checkpoint_dir}")

            shards: list[tuple[int, Path]] = []
            for candidate in checkpoint_dir.iterdir():
                match = _CHECKPOINT_NAME.fullmatch(candidate.name)
                if match is not None and candidate.is_file():
                    shards.append((int(match.group("sequence")), candidate))
                    continue
                if _CHECKPOINT_TEMP_NAME.fullmatch(candidate.name) and candidate.is_file():
                    continue
                raise CheckpointCacheError(f"unexpected checkpoint artifact: {candidate}")
        except OSError as exc:
            raise CheckpointCacheError(
                f"failed to inspect checkpoint directory {checkpoint_dir}: {exc}"
            ) from exc

        for expected_sequence, (sequence, shard_path) in enumerate(
            sorted(shards),
            start=1,
        ):
            if sequence != expected_sequence:
                raise CheckpointCacheError(
                    f"checkpoint sequence must be contiguous from 00000001; found {shard_path.name}"
                )
            additions = _read_json_object(shard_path, label="checkpoint shard")
            if len(additions) > self.interval:
                raise CheckpointCacheError(
                    f"checkpoint shard exceeds interval {self.interval}: {shard_path}"
                )
            self._merge(additions, shard_path)
            self._next_sequence = sequence + 1
        return dict(self._entries)

    def record(self, key: str, value: dict[str, Any]) -> None:
        """Record one validated entry and checkpoint when the interval is reached."""

        normalized_key = str(key)
        existing = self._entries.get(normalized_key)
        if existing is not None:
            if existing != value:
                raise CheckpointCacheError(f"conflicting cache entry for key {normalized_key}")
            return
        self._entries[normalized_key] = value
        self._pending[normalized_key] = value
        if len(self._pending) >= self.interval:
            self.flush()

    def retain(self, keys: set[str]) -> None:
        """Drop entries that no longer belong to the current build input."""

        self._entries = {key: value for key, value in self._entries.items() if key in keys}
        self._pending = {key: value for key, value in self._pending.items() if key in keys}

    def flush(self) -> None:
        """Atomically write the current partial shard, if any."""

        if not self._pending or self.path is None:
            return
        checkpoint_dir = self.checkpoint_dir
        assert checkpoint_dir is not None
        shard_path = checkpoint_dir / f"{self._next_sequence:08d}.json"
        try:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_text(shard_path, _json_text(self._pending))
        except OSError as exc:
            raise CheckpointCacheError(
                f"failed to write checkpoint shard {shard_path}: {exc}"
            ) from exc
        self._pending = {}
        self._next_sequence += 1

    def compact(self) -> None:
        """Publish one canonical cache and remove superseded checkpoint shards."""

        if self.path is None:
            return
        self.flush()
        try:
            atomic_write_text(self.path, _json_text(self._entries))
        except OSError as exc:
            raise CheckpointCacheError(
                f"failed to write canonical cache {self.path}: {exc}"
            ) from exc
        checkpoint_dir = self.checkpoint_dir
        if checkpoint_dir is None or not checkpoint_dir.exists():
            return
        try:
            for candidate in checkpoint_dir.iterdir():
                if not (
                    _CHECKPOINT_NAME.fullmatch(candidate.name)
                    or _CHECKPOINT_TEMP_NAME.fullmatch(candidate.name)
                ):
                    raise CheckpointCacheError(f"unexpected checkpoint artifact: {candidate}")
                candidate.unlink()
            checkpoint_dir.rmdir()
        except OSError as exc:
            raise CheckpointCacheError(
                f"failed to clean checkpoint directory {checkpoint_dir}: {exc}"
            ) from exc

    def _merge(self, additions: dict[str, dict[str, Any]], source: Path) -> None:
        for key, value in additions.items():
            existing = self._entries.get(key)
            if existing is not None and existing != value:
                raise CheckpointCacheError(f"conflicting cache entry for key {key} in {source}")
            self._entries[key] = value


def _read_json_object(path: Path, *, label: str) -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointCacheError(f"failed to read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict) or any(
        not isinstance(value, dict) for value in payload.values()
    ):
        raise CheckpointCacheError(f"{label} must map keys to JSON objects: {path}")
    return {str(key): value for key, value in payload.items()}


def _json_text(payload: dict[str, dict[str, Any]]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
