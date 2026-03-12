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
                "references": [
                    {
                        "paperId": "prior",
                        "title": "Prior Compression Work",
                        "year": 2025,
                    }
                ],
                "citations": [
                    {
                        "paperId": "future",
                        "title": "Future Compression Follow-up",
                        "year": 2027,
                    }
                ],
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
        "tasks": [{"title": "Benchmark prompt compression", "status": "doing"}],
        "milestones": [{"name": "Submit workshop paper", "status": "todo"}],
        "scholars": [{"name": "Alice Smith", "affiliation": "PaperBot Lab"}],
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
    assert "[[PaperBot/Tracks/icl-compression/_MOC|ICL Compression]]" in paper_body
    assert "[DOI](https://doi.org/10.1000/uniicl)" in paper_body
    assert "cites:" in paper_body
    assert "cited_by:" in paper_body
    assert "## References" in paper_body
    assert (
        "[[PaperBot/Papers/2025-prior-compression-work-prior|Prior Compression Work]]" in paper_body
    )
    assert "## Cited By" in paper_body
    assert (
        "[[PaperBot/Papers/2027-future-compression-follow-up-future|Future Compression Follow-up]]"
        in paper_body
    )

    track_body = track_note.read_text(encoding="utf-8")
    assert "paperbot_type: track" in track_body
    assert "# ICL Compression" in track_body
    assert "## Research Tasks" in track_body
    assert "- [doing] Benchmark prompt compression" in track_body
    assert "## Milestones" in track_body
    assert "- [todo] Submit workshop paper" in track_body
    assert "## Tracked Scholars" in track_body
    assert "- Alice Smith (PaperBot Lab)" in track_body
    assert "[[PaperBot/Papers/2026-uniicl-2601-12345|UniICL]]" in track_body

    moc_body = moc_note.read_text(encoding="utf-8")
    assert "# PaperBot MOC" in moc_body
    assert "[[PaperBot/Tracks/icl-compression/_MOC|ICL Compression]]" in moc_body
    assert "[[PaperBot/Papers/2026-uniicl-2601-12345|UniICL]]" in moc_body
    assert track_note == vault / "PaperBot" / "Tracks" / "icl-compression" / "_MOC.md"


def test_export_library_snapshot_supports_custom_template_and_related_links(tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    template_path = tmp_path / "paper_note.md.j2"
    template_path.write_text(
        (
            "{{ frontmatter }}\n"
            "# {{ title }}\n"
            "Summary: {{ abstract }}\n"
            "Track Link: {{ track_link }}\n"
            "{% for link in related_links %}- {{ link }}\n{% endfor %}"
        ),
        encoding="utf-8",
    )

    exporter = ObsidianFilesystemExporter()
    result = exporter.export_library_snapshot(
        vault_path=vault,
        saved_items=[
            {
                "saved_at": "2026-03-11T11:00:00+00:00",
                "paper": {
                    "id": 1,
                    "title": "UniICL",
                    "authors": ["Alice Smith"],
                    "abstract": "Contains <script>alert(1)</script> & evidence.",
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
        paper_template_path=template_path,
    )

    paper_note = Path(result["paper_notes"][0]).read_text(encoding="utf-8")
    assert "related_papers:" in paper_note
    assert "Prompt Compression Survey" in paper_note
    assert "Contains &lt;script&gt;alert(1)&lt;/script&gt; &amp; evidence." in paper_note
    assert (
        "[[PaperBot/Papers/2025-prompt-compression-survey|Prompt Compression Survey]]" in paper_note
    )
    assert (
        "[[PaperBot/Papers/context-distillation-for-llms|Context Distillation for LLMs]]"
        in paper_note
    )
    assert "[[PaperBot/Tracks/icl-compression/_MOC|ICL Compression]]" in paper_note
    assert paper_note.count("paperbot_type: paper") == 1


def test_export_library_snapshot_backfills_existing_cited_by_links(tmp_path: Path):
    exporter = ObsidianFilesystemExporter()
    vault = tmp_path / "vault"
    vault.mkdir()

    exporter.export_library_snapshot(
        vault_path=vault,
        saved_items=[
            {
                "paper": {
                    "id": "prior",
                    "title": "Prior Compression Work",
                    "year": 2025,
                }
            }
        ],
    )

    exporter.export_library_snapshot(
        vault_path=vault,
        saved_items=[
            {
                "paper": {
                    "id": 1,
                    "title": "UniICL",
                    "year": 2026,
                    "references": [
                        {
                            "paperId": "prior",
                            "title": "Prior Compression Work",
                            "year": 2025,
                        }
                    ],
                }
            }
        ],
    )

    prior_note = (vault / "PaperBot" / "Papers" / "2025-prior-compression-work-prior.md").read_text(
        encoding="utf-8"
    )
    assert "cited_by:" in prior_note
    assert "## Cited By" in prior_note
    assert "[[PaperBot/Papers/2026-uniicl-1|UniICL]]" in prior_note


def test_update_note_link_index_uses_explicit_root_path(tmp_path: Path, monkeypatch):
    exporter = ObsidianFilesystemExporter()
    root_path = tmp_path / "vault" / "PaperBot"
    note_path = root_path / "Papers" / "nested" / "prior.md"
    note_path.parent.mkdir(parents=True)
    note_path.write_text(
        (
            "---\n"
            "paperbot_type: paper\n"
            "paperbot_managed_links: []\n"
            "cited_by: []\n"
            "---\n"
            "# Prior\n\n"
            "## Cited By\n"
            "- [[PaperBot/Papers/existing|Existing]]\n"
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def _capture_write(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(exporter, "_write_managed_note", _capture_write)

    exporter._update_note_link_index(
        note_path=note_path,
        root_path=root_path,
        frontmatter_key="cited_by",
        heading="Cited By",
        link="[[PaperBot/Papers/current|Current]]",
    )

    assert captured["root_path"] == root_path
    assert "[[PaperBot/Papers/current|Current]]" in captured["frontmatter_payload"]["cited_by"]


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


def test_export_library_snapshot_preserves_existing_personal_notes(tmp_path: Path):
    exporter = ObsidianFilesystemExporter()
    vault = tmp_path / "vault"
    vault.mkdir()

    result = exporter.export_library_snapshot(
        vault_path=vault,
        saved_items=[
            {
                "paper": {
                    "id": "paper-1",
                    "title": "UniICL",
                    "abstract": "Initial summary.",
                    "year": 2026,
                }
            }
        ],
    )
    note_path = Path(result["paper_notes"][0])
    note_path.write_text(
        note_path.read_text(encoding="utf-8").rstrip()
        + "\n\n## Personal Notes\nKeep this observation.\n",
        encoding="utf-8",
    )

    exporter.export_library_snapshot(
        vault_path=vault,
        saved_items=[
            {
                "paper": {
                    "id": "paper-1",
                    "title": "UniICL",
                    "abstract": "Updated generated summary.",
                    "year": 2026,
                }
            }
        ],
    )

    body = note_path.read_text(encoding="utf-8")
    assert "Updated generated summary." in body
    assert "## Personal Notes" in body
    assert "Keep this observation." in body
    assert "paperbot_managed_hash:" in body
    assert "paperbot_exported_at:" in body


def test_export_library_snapshot_writes_pending_note_when_user_edits_managed_section(
    tmp_path: Path,
):
    exporter = ObsidianFilesystemExporter()
    vault = tmp_path / "vault"
    vault.mkdir()

    result = exporter.export_library_snapshot(
        vault_path=vault,
        saved_items=[
            {
                "paper": {
                    "id": "paper-1",
                    "title": "UniICL",
                    "abstract": "Initial summary.",
                    "year": 2026,
                }
            }
        ],
    )
    note_path = Path(result["paper_notes"][0])
    original_body = note_path.read_text(encoding="utf-8")
    note_path.write_text(
        original_body.replace("Initial summary.", "User-edited managed summary."),
        encoding="utf-8",
    )

    exporter.export_library_snapshot(
        vault_path=vault,
        saved_items=[
            {
                "paper": {
                    "id": "paper-1",
                    "title": "UniICL",
                    "abstract": "New PaperBot summary.",
                    "year": 2026,
                }
            }
        ],
    )

    current_body = note_path.read_text(encoding="utf-8")
    pending_path = vault / "PaperBot" / ".paperbot-pending" / "Papers" / note_path.name
    pending_body = pending_path.read_text(encoding="utf-8")

    assert "User-edited managed summary." in current_body
    assert "New PaperBot summary." not in current_body
    assert pending_path.exists()
    assert "paperbot_status: pending" in pending_body
    assert "New PaperBot summary." in pending_body
