from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CLI_ROOT = ROOT / "cli"


def test_removed_legacy_cli_entrypoints_and_submit_panel() -> None:
    assert not (CLI_ROOT / "src" / "components" / "index.ts").exists()
    assert not (CLI_ROOT / "src" / "components" / "sandbox" / "SubmitPanel.tsx").exists()


def test_cli_index_does_not_expose_dead_stream_flag() -> None:
    index_source = (CLI_ROOT / "src" / "index.tsx").read_text(encoding="utf-8")
    assert "--stream" not in index_source
    assert "stream:" not in index_source


def test_cli_api_surface_is_narrowed_to_client_singleton() -> None:
    api_source = (CLI_ROOT / "src" / "utils" / "api.ts").read_text(encoding="utf-8")
    assert "export class PaperBotClient" not in api_source
    assert "async request<" not in api_source
    assert "export const client = new PaperBotClient();" in api_source


def test_cli_uses_eslint_flat_config() -> None:
    assert (CLI_ROOT / "eslint.config.js").exists()
