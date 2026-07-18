from __future__ import annotations

import json

import pytest

from skillfabric.storage.checkpoint_cache import (
    CheckpointCacheError,
    JsonObjectCheckpointCache,
)


def test_checkpoint_cache_rejects_missing_shard_sequence(tmp_path) -> None:
    cache_path = tmp_path / "contracts.json"
    checkpoint_dir = tmp_path / "contracts.checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "00000002.json").write_text(
        json.dumps({"second": {"value": 2}}),
        encoding="utf-8",
    )

    with pytest.raises(CheckpointCacheError, match="checkpoint sequence"):
        JsonObjectCheckpointCache(cache_path, interval=100).load()


def test_checkpoint_cache_rejects_shard_larger_than_interval(tmp_path) -> None:
    cache_path = tmp_path / "contracts.json"
    checkpoint_dir = tmp_path / "contracts.checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "00000001.json").write_text(
        json.dumps(
            {
                "first": {"value": 1},
                "second": {"value": 2},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(CheckpointCacheError, match="exceeds interval"):
        JsonObjectCheckpointCache(cache_path, interval=1).load()


def test_checkpoint_cache_rejects_conflicting_canonical_and_shard_entries(
    tmp_path,
) -> None:
    cache_path = tmp_path / "contracts.json"
    cache_path.write_text(json.dumps({"same": {"value": 1}}), encoding="utf-8")
    checkpoint_dir = tmp_path / "contracts.checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "00000001.json").write_text(
        json.dumps({"same": {"value": 2}}),
        encoding="utf-8",
    )

    with pytest.raises(CheckpointCacheError, match="conflicting cache entry"):
        JsonObjectCheckpointCache(cache_path, interval=100).load()


def test_checkpoint_cache_compacts_committed_shards_and_ignores_temp_file(
    tmp_path,
) -> None:
    cache_path = tmp_path / "contracts.json"
    cache = JsonObjectCheckpointCache(cache_path, interval=2)
    assert cache.load() == {}

    cache.record("first", {"value": 1})
    cache.record("second", {"value": 2})
    checkpoint_dir = tmp_path / "contracts.checkpoints"
    (checkpoint_dir / "00000002.json.tmp").write_text("partial", encoding="utf-8")
    cache.record("third", {"value": 3})
    cache.compact()

    assert json.loads(cache_path.read_text(encoding="utf-8")) == {
        "first": {"value": 1},
        "second": {"value": 2},
        "third": {"value": 3},
    }
    assert not checkpoint_dir.exists()
