from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from paperbot.infrastructure.exporters import obsidian_sync


class _FakeResearchStore:
    def __init__(self) -> None:
        self.closed = False

    def get_track(self, *, user_id: str, track_id: int):
        assert user_id == "default"
        assert track_id == 7
        return {"id": 7, "user_id": "default", "name": "ICL Compression"}

    def list_saved_papers(self, *, user_id: str, track_id: int, limit: int):
        assert user_id == "default"
        assert track_id == 7
        assert limit == 25
        return [{"paper": {"id": 1, "title": "UniICL"}}]

    def close(self) -> None:
        self.closed = True


def test_export_track_snapshot_uses_obsidian_settings(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()

    class _FakeExporter:
        def __init__(self, *, paper_template_path=None):
            captured["template_path"] = paper_template_path

        def export_library_snapshot(self, *, vault_path, saved_items, track, root_dir):
            captured["vault_path"] = Path(vault_path)
            captured["saved_items"] = saved_items
            captured["track"] = track
            captured["root_dir"] = root_dir
            return {"paper_count": len(saved_items)}

    monkeypatch.setattr(
        obsidian_sync,
        "create_settings",
        lambda: SimpleNamespace(
            obsidian=SimpleNamespace(
                enabled=True,
                vault_path=str(vault_dir),
                root_dir="PaperBot Notes",
                paper_template_path=str(tmp_path / "paper.md.j2"),
                export_limit=25,
                auto_export_on_save=True,
                auto_sync_tracks=True,
            )
        ),
    )
    monkeypatch.setattr(obsidian_sync, "SqlAlchemyResearchStore", _FakeResearchStore)
    monkeypatch.setattr(obsidian_sync, "ObsidianFilesystemExporter", _FakeExporter)

    result = obsidian_sync.export_track_snapshot(user_id="default", track_id=7)

    assert result == {"paper_count": 1}
    assert captured["vault_path"] == vault_dir
    assert captured["root_dir"] == "PaperBot Notes"
    assert captured["track"] == {"id": 7, "user_id": "default", "name": "ICL Compression"}
    assert captured["saved_items"] == [{"paper": {"id": 1, "title": "UniICL"}}]
    assert captured["template_path"] == (tmp_path / "paper.md.j2")


def test_obsidian_auto_export_enabled_requires_vault_path(monkeypatch):
    monkeypatch.setattr(
        obsidian_sync,
        "create_settings",
        lambda: SimpleNamespace(
            obsidian=SimpleNamespace(
                enabled=True,
                vault_path="",
                root_dir="PaperBot",
                paper_template_path=None,
                export_limit=10,
                auto_export_on_save=True,
                auto_sync_tracks=True,
            )
        ),
    )

    assert obsidian_sync.obsidian_auto_export_enabled() is False
    assert obsidian_sync.obsidian_auto_export_enabled(for_tracks=True) is False
