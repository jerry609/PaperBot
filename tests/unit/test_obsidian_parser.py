from __future__ import annotations

from paperbot.infrastructure.obsidian import (
    PAPER_MANAGED_HEADINGS,
    merge_user_sections,
    parse_note_text,
)


def test_parse_note_text_extracts_user_sections_tags_and_wikilinks() -> None:
    text = """---
paperbot_type: paper
tags:
  - graph
paperbot_managed_tags:
  - graph
paperbot_managed_links:
  - PaperBot/Tracks/icl-compression/_MOC
---
# UniICL

## Summary
Managed summary.

## Metadata
- Venue: ICLR

## Personal Notes
Need to revisit this with #important and [[Custom/Idea|Idea]].

## Open Questions
Compare against [[PaperBot/Papers/follow-up-paper|Follow-up Paper]].
"""

    parsed = parse_note_text(text, managed_headings=PAPER_MANAGED_HEADINGS)

    assert parsed.title == "UniICL"
    assert [section.heading for section in parsed.user_sections] == [
        "Personal Notes",
        "Open Questions",
    ]
    assert parsed.user_tags == ["important"]
    assert parsed.user_wikilinks == [
        "Custom/Idea",
        "PaperBot/Papers/follow-up-paper",
    ]

    merged_body = merge_user_sections(parsed.managed_body, parsed.user_sections)
    assert "## Personal Notes" in merged_body
    assert "## Open Questions" in merged_body
