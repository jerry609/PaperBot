from __future__ import annotations

from pathlib import Path

from paperbot.infrastructure.exporters import ObsidianFilesystemExporter


def test_export_library_snapshot_writes_paper_track_and_moc_notes(tmp_path: Path):
    exporter = ObsidianFilesystemExporter()
    vault = tmp_path / "vault"
    vault.mkdir()

    saved_items = [
        {
            "saved_at": "2026-03-11T11:00:00+00:00",
            "paper": {
                "id": 1,
                "title": "UniICL",
                "authors": ["Alice Smith", "Bob Doe"],
                "abstract": "Compresses in-context examples.",
                "year": 2026,
                "venue": "ICLR",
                "doi": "10.1000/uniicl",
                "arxiv_id": "2601.12345",
                "citation_count": 12,
                "keywords": ["ICL", "Compression"],
                "fields_of_study": ["Natural Language Processing"],
                "url": "https://example.com/uniicl",
            },
        }
    ]
    track = {
        "id": 7,
        "user_id": "default",
        "name": "ICL Compression",
        "description": "Track compression methods for in-context learning.",
        "keywords": ["ICL", "Compression"],
        "methods": ["retrieval"],
        "venues": ["ICLR"],
        "is_active": True,
    }

    result = exporter.export_library_snapshot(
        vault_path=vault,
        saved_items=saved_items,
        track=track,
    )

    paper_note = Path(result["paper_notes"][0])
    track_note = Path(result["track_note"])
    moc_note = Path(result["moc_note"])

    assert paper_note.exists()
    assert track_note.exists()
    assert moc_note.exists()

    paper_body = paper_note.read_text(encoding="utf-8")
    assert "paperbot_type: paper" in paper_body
    assert "# UniICL" in paper_body
    assert "[[PaperBot/Tracks/icl-compression|ICL Compression]]" in paper_body
    assert "[DOI](https://doi.org/10.1000/uniicl)" in paper_body

    track_body = track_note.read_text(encoding="utf-8")
    assert "paperbot_type: track" in track_body
    assert "# ICL Compression" in track_body
    assert "[[PaperBot/Papers/2026-uniicl-2601-12345|UniICL]]" in track_body

    moc_body = moc_note.read_text(encoding="utf-8")
    assert "# PaperBot MOC" in moc_body
    assert "[[PaperBot/Tracks/icl-compression|ICL Compression]]" in moc_body
    assert "[[PaperBot/Papers/2026-uniicl-2601-12345|UniICL]]" in moc_body


def test_export_library_snapshot_supports_custom_template_and_related_links(tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    template_path = tmp_path / "paper_note.md.j2"
    template_path.write_text(
        (
            "{{ frontmatter }}\n"
            "# {{ title }}\n"
            "Track Link: {{ track_link }}\n"
            "{% for link in related_links %}- {{ link }}\n{% endfor %}"
        ),
        encoding="utf-8",
    )

    exporter = ObsidianFilesystemExporter(paper_template_path=template_path)
    result = exporter.export_library_snapshot(
        vault_path=vault,
        saved_items=[
            {
                "saved_at": "2026-03-11T11:00:00+00:00",
                "paper": {
                    "id": 1,
                    "title": "UniICL",
                    "authors": ["Alice Smith"],
                    "abstract": "Compresses in-context examples.",
                    "year": 2026,
                    "venue": "ICLR",
                    "semantic_scholar_id": "S2-UNIICL",
                    "citation_count": 12,
                    "related_papers": [
                        {"title": "Prompt Compression Survey", "year": 2025},
                        "Context Distillation for LLMs",
                    ],
                },
            }
        ],
        track={
            "id": 7,
            "user_id": "default",
            "name": "ICL Compression",
            "keywords": ["ICL"],
            "methods": [],
            "venues": [],
            "is_active": True,
        },
    )

    paper_note = Path(result["paper_notes"][0]).read_text(encoding="utf-8")
    assert "related_papers:" in paper_note
    assert "Prompt Compression Survey" in paper_note
    assert "[[PaperBot/Papers/2025-prompt-compression-survey|Prompt Compression Survey]]" in paper_note
    assert "[[PaperBot/Papers/context-distillation-for-llms|Context Distillation for LLMs]]" in paper_note
    assert "[[PaperBot/Tracks/icl-compression|ICL Compression]]" in paper_note


def test_export_library_snapshot_requires_existing_vault_directory(tmp_path: Path):
    exporter = ObsidianFilesystemExporter()
    missing_vault = tmp_path / "missing-vault"

    try:
        exporter.export_library_snapshot(
            vault_path=missing_vault,
            saved_items=[],
        )
    except ValueError as exc:
        assert "vault_path must be an existing directory" in str(exc)
    else:
        raise AssertionError("expected exporter to reject a missing vault directory")
