from __future__ import annotations

import contextlib
import io
import json
import stat
from unittest.mock import patch

from skillfabric.cli import main as cli_main
from skillfabric.router.bundle import RouterBundleConfig, build_router_bundle
from skillfabric.wiki.query_wiki import materialize_query_wiki
from tests.unit.fake_embeddings import FakeEmbeddingProvider
from tests.unit.wiki_helpers import build_fixture_workspace


def test_init_check_reports_presence_without_secret_values(tmp_path) -> None:
    env_file = tmp_path / ".env.test"
    env_file.write_text(
        "API_KEY=private-test-value\n"
        "BASE_URL=https://example.test/v1\n"
        "MODEL=openai/test-model\n"
        "EMBEDDING_MODEL=openai/test-embedding\n",
        encoding="utf-8",
    )
    output = io.StringIO()

    with contextlib.redirect_stdout(output):
        cli_main(["init", "--check", "--json", "--env-file", str(env_file)])

    text = output.getvalue()
    payload = json.loads(text)
    assert payload["configured"] is True
    assert all(payload["present"].values())
    assert "private-test-value" not in text
    assert "openai/test-model" not in text


def test_init_writes_private_env_file_without_echoing_values(tmp_path) -> None:
    env_file = tmp_path / ".env.test"
    output = io.StringIO()

    with (
        patch("getpass.getpass", return_value="private-test-value"),
        patch(
            "builtins.input",
            side_effect=[
                "https://example.test/v1",
                "openai/test-model",
                "openai/test-embedding",
            ],
        ),
        contextlib.redirect_stdout(output),
    ):
        cli_main(["init", "--env-file", str(env_file)])

    assert env_file.exists()
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
    assert "private-test-value" not in output.getvalue()


def test_query_wiki_card_command_never_prints_full_source(tmp_path) -> None:
    workspace = tmp_path / ".skillfabric"
    build_fixture_workspace(workspace)
    bundle = build_router_bundle(
        RouterBundleConfig(
            workspace=workspace,
            query="parse PDF tables",
            seed_limit=2,
            expanded_limit=8,
        ),
        embedding_provider=FakeEmbeddingProvider(),
    )
    query_wiki = materialize_query_wiki(
        workspace,
        bundle,
        trace_dir=workspace / "runs" / "card-command",
    )
    output = io.StringIO()

    with contextlib.redirect_stdout(output):
        cli_main(
            [
                "query-wiki",
                "card",
                str(query_wiki.root),
                "skill:pdf-table-parser",
            ]
        )

    text = output.getvalue()
    assert "normalized_csv_table" in text
    assert "untrusted_skill_source" not in text
