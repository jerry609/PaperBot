from __future__ import annotations

from pathlib import Path

from paperbot.infrastructure.event_log.memory_event_log import InMemoryEventLog
from paperbot.infrastructure.obsidian import (
    ObsidianBidirectionalSync,
    PAPER_MANAGED_HEADINGS,
    hash_markdown,
)


class _FakeMemoryStore:
    def __init__(self) -> None:
        self.memories = []

    def add_memories(self, *, user_id: str, memories: list, **_: object):
        self.memories.extend((user_id, memory) for memory in memories)
        return len(memories), 0, []


def test_obsidian_sync_scan_captures_user_tags_links_notes_and_conflicts(tmp_path: Path) -> None:
    user_id = "obsidian-user"
    vault = tmp_path / "vault"
    root = vault / "PaperBot"
    papers_dir = root / "Papers"
    papers_dir.mkdir(parents=True)

    managed_body = """# UniICL

## Summary
Baseline summary.

## Metadata
- Venue: ICLR

## Tracks
- [[PaperBot/Tracks/icl-compression/_MOC|ICL Compression]]
"""
    note_path = papers_dir / "2026-uniicl-paper-1.md"
    note_path.write_text(
        (
            "---\n"
            "paperbot_type: paper\n"
            "paperbot_id: paper-1\n"
            f"user_id: {user_id}\n"
            "title: UniICL\n"
            "tags:\n"
            "  - icl\n"
            "paperbot_managed_tags:\n"
            "  - icl\n"
            "paperbot_managed_links:\n"
            "  - PaperBot/Tracks/icl-compression/_MOC\n"
            f"paperbot_managed_hash: {hash_markdown(managed_body)}\n"
            "---\n"
            f"{managed_body}"
        ),
        encoding="utf-8",
    )

    memory_store = _FakeMemoryStore()
    event_log = InMemoryEventLog()
    sync_service = ObsidianBidirectionalSync(
        vault_path=vault,
        root_dir="PaperBot",
        memory_store=memory_store,
        event_log=event_log,
    )

    baseline = sync_service.scan()
    assert baseline.changed_notes == 1
    assert baseline.memories_created == 0

    user_managed_body = """# UniICL

## Summary
User edited the managed summary.

## Metadata
- Venue: ICLR

## Tracks
- [[PaperBot/Tracks/icl-compression/_MOC|ICL Compression]]

## Personal Notes
Need to try this for #important retrieval runs.

## Open Questions
Relates to [[Custom/Prompt Compression]].
"""
    note_path.write_text(
        (
            "---\n"
            "paperbot_type: paper\n"
            "paperbot_id: paper-1\n"
            f"user_id: {user_id}\n"
            "title: UniICL\n"
            "tags:\n"
            "  - icl\n"
            "paperbot_managed_tags:\n"
            "  - icl\n"
            "paperbot_managed_links:\n"
            "  - PaperBot/Tracks/icl-compression/_MOC\n"
            f"paperbot_managed_hash: {hash_markdown(managed_body)}\n"
            "---\n"
            f"{user_managed_body}"
        ),
        encoding="utf-8",
    )

    result = sync_service.scan()
    assert result.changed_notes == 1
    assert result.memories_created == 4
    assert result.tag_updates == 1
    assert result.wikilink_updates == 1
    assert result.note_updates == 1
    assert result.conflicts_detected == 1

    memory_contents = [memory.content for _, memory in memory_store.memories]
    assert any('Obsidian tag for "UniICL": important' == content for content in memory_contents)
    assert any(
        'Obsidian wiki-link from "UniICL" to "Custom/Prompt Compression"' == content
        for content in memory_contents
    )
    assert any(
        'Obsidian personal note for "UniICL" (Personal Notes)' in content
        for content in memory_contents
    )

    status = sync_service.get_status()
    assert status.tracked_note_count == 1
    assert status.conflict_count == 1
    assert status.pending_count == 0

    event_types = [event["type"] for event in event_log.events]
    assert "obsidian.note.conflict_detected" in event_types
