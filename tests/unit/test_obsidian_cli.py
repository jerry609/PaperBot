from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from paperbot.presentation.cli import main as cli_main


class _FakeResearchStore:
    def get_track(self, *, user_id: str, track_id: int):
        if user_id == "default" and track_id == 7:
            return {
                "id": 7,
                "user_id": "default",
                "name": "ICL Compression",
                "description": "Track compression methods.",
                "keywords": ["ICL", "Compression"],
                "methods": ["retrieval"],
                "venues": ["ICLR"],
                "is_active": True,
            }
        return None

    def list_tracks(self, *, user_id: str, include_archived: bool, limit: int):
        return [
            {
                "id": 7,
                "user_id": user_id,
                "name": "ICL Compression",
                "description": "Track compression methods.",
                "keywords": ["ICL", "Compression"],
                "methods": ["retrieval"],
                "venues": ["ICLR"],
                "is_active": True,
            }
        ]

    def list_saved_papers(self, *, user_id: str, track_id: int | None, limit: int):
        assert user_id == "default"
        assert track_id == 7
        assert limit == 5
        return [
            {
                "saved_at": "2026-03-11T11:00:00+00:00",
                "paper": {
                    "id": 1,
                    "title": "UniICL",
                    "authors": ["Alice Smith"],
                    "abstract": "Compresses in-context examples.",
                    "year": 2026,
                    "venue": "ICLR",
                    "doi": "10.1000/uniicl",
                    "arxiv_id": "2601.12345",
                    "citation_count": 12,
                    "keywords": ["ICL"],
                    "fields_of_study": ["Natural Language Processing"],
                    "url": "https://example.com/uniicl",
                },
            }
        ]

    def close(self) -> None:
        return None


class _FakeExporter:
    def export_library_snapshot(self, *, vault_path, saved_items, track=None, root_dir="PaperBot"):
        assert Path(vault_path) == Path("/tmp/my-vault")
        assert len(saved_items) == 1
        assert track is not None
        assert track["name"] == "ICL Compression"
        assert root_dir == "PaperBot"
        return {
            "vault_path": str(vault_path),
            "root_dir": root_dir,
            "paper_count": 1,
            "paper_notes": ["/tmp/my-vault/PaperBot/Papers/2026-uniicl-2601-12345.md"],
            "track_note": "/tmp/my-vault/PaperBot/Tracks/icl-compression.md",
            "moc_note": "/tmp/my-vault/PaperBot/MOC.md",
        }


def test_cli_obsidian_export_parser_flags():
    parser = cli_main.create_parser()
    args = parser.parse_args(
        [
            "export",
            "obsidian",
            "--vault",
            "/tmp/my-vault",
            "--track-name",
            "ICL Compression",
            "--limit",
            "5",
            "--json",
        ]
    )

    assert args.command == "export"
    assert args.export_target == "obsidian"
    assert args.vault == "/tmp/my-vault"
    assert args.track_name == "ICL Compression"
    assert args.limit == 5
    assert args.json is True


def test_cli_obsidian_export_json_output(monkeypatch, capsys):
    import paperbot.infrastructure.exporters as exporters_pkg
    import paperbot.infrastructure.exporters.obsidian_exporter as exporter_module
    import paperbot.infrastructure.stores.research_store as research_store_module

    monkeypatch.setattr(research_store_module, "SqlAlchemyResearchStore", _FakeResearchStore)
    monkeypatch.setattr(exporters_pkg, "ObsidianFilesystemExporter", _FakeExporter)
    monkeypatch.setattr(exporter_module, "ObsidianFilesystemExporter", _FakeExporter)

    exit_code = cli_main.run_cli(
        [
            "export",
            "obsidian",
            "--vault",
            "/tmp/my-vault",
            "--track-name",
            "ICL Compression",
            "--limit",
            "5",
            "--json",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["paper_count"] == 1
    assert payload["track_note"].endswith("icl-compression.md")
    assert payload["moc_note"].endswith("MOC.md")


def test_cli_obsidian_export_uses_settings_defaults(monkeypatch, capsys):
    import paperbot.infrastructure.exporters as exporters_pkg
    import paperbot.infrastructure.exporters.obsidian_exporter as exporter_module
    import paperbot.infrastructure.stores.research_store as research_store_module

    monkeypatch.setattr(research_store_module, "SqlAlchemyResearchStore", _FakeResearchStore)
    monkeypatch.setattr(exporters_pkg, "ObsidianFilesystemExporter", _FakeExporter)
    monkeypatch.setattr(exporter_module, "ObsidianFilesystemExporter", _FakeExporter)
    monkeypatch.setattr(
        cli_main,
        "create_settings",
        lambda: SimpleNamespace(
            obsidian=SimpleNamespace(
                vault_path="/tmp/my-vault",
                root_dir="PaperBot",
                paper_template_path=None,
            )
        ),
    )

    exit_code = cli_main.run_cli(
        [
            "export",
            "obsidian",
            "--track-name",
            "ICL Compression",
            "--limit",
            "5",
            "--json",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["vault_path"] == "/tmp/my-vault"
