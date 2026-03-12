from __future__ import annotations

from pathlib import Path

from paperbot.infrastructure.obsidian.watcher import DebouncedPathQueue, ObsidianVaultWatcher


def test_debounced_path_queue_coalesces_repeated_updates() -> None:
    now = [0.0]
    queue = DebouncedPathQueue(debounce_seconds=1.0, time_fn=lambda: now[0])

    queue.push(Path("/tmp/note.md"))
    now[0] = 0.25
    queue.push(Path("/tmp/note.md"))
    now[0] = 1.0
    assert queue.flush_ready() == []

    now[0] = 1.3
    assert queue.flush_ready() == [Path("/tmp/note.md")]


def test_obsidian_vault_watcher_flush_pending_uses_markdown_paths_only(tmp_path: Path) -> None:
    received = []
    watcher = ObsidianVaultWatcher(
        root_path=tmp_path,
        on_paths_changed=lambda paths: received.append(paths),
        debounce_seconds=0.1,
    )

    watcher.record_event(tmp_path / "paper.md")
    watcher.record_event(tmp_path / "paper.md")
    watcher.record_event(tmp_path / "paper.txt")

    flushed = watcher.flush_pending()
    assert flushed == [(tmp_path / "paper.md").resolve()]
    assert received == [[(tmp_path / "paper.md").resolve()]]
