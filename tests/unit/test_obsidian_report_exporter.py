from __future__ import annotations

from pathlib import Path

from paperbot.infrastructure.exporters import ObsidianReportExporter


def test_export_report_note_writes_obsidian_longform_note(tmp_path: Path):
    exporter = ObsidianReportExporter()
    vault = tmp_path / "vault"
    vault.mkdir()

    result = exporter.export_report_note(
        vault_path=vault,
        root_dir="PaperBot",
        report={
            "title": "ICL Compression Landscape",
            "track_name": "ICL Compression",
            "workflow_type": "research",
            "summary": "Compression methods are converging around retrieval-aware token pruning.",
            "key_insight": "Retrieval-aware compression preserves accuracy at lower token budgets.",
            "sections": [
                {
                    "title": "Key Findings",
                    "content": "Prompt compression is shifting from heuristics to trainable selectors.",
                    "cited_papers": [
                        {
                            "title": "UniICL",
                            "year": 2026,
                            "authors": ["Alice Smith"],
                            "semantic_scholar_id": "S2-UNIICL",
                            "relevant_finding": "Introduces a unified compression benchmark.",
                        }
                    ],
                }
            ],
            "methods": [
                {
                    "name": "Retrieval Compression",
                    "paper": "UniICL",
                    "pros": "High recall",
                    "cons": "Extra index latency",
                }
            ],
            "trends": "Benchmarks are becoming more retrieval-centric.",
            "future_directions": "Evaluate long-context compression under agent workflows.",
            "references": [
                {
                    "title": "UniICL",
                    "year": 2026,
                    "authors": ["Alice Smith"],
                    "semantic_scholar_id": "S2-UNIICL",
                }
            ],
            "tags": ["NLP", "Compression"],
        },
    )

    note_path = Path(result["note_path"])
    body = note_path.read_text(encoding="utf-8")

    assert note_path == vault / "PaperBot" / "Reports" / "icl-compression-landscape.md"
    assert "type: research-report" in body
    assert "[[PaperBot/Tracks/icl-compression/_MOC|ICL Compression]]" in body
    assert "> [!abstract] 研究概述" in body
    assert "> [!tip] 核心观点" in body
    assert "> [!quote] [[PaperBot/Papers/2026-uniicl-s2-uniicl|UniICL]]" in body
    assert "| 方法 | 论文 | 优势 | 局限 |" in body
    assert "| Retrieval Compression | [[PaperBot/Papers/uniicl\\|UniICL]] | High recall | Extra index latency |" in body
    assert "> [!info] 趋势分析" in body
    assert "## 引用论文" in body
    assert "- [[PaperBot/Papers/2026-uniicl-s2-uniicl|UniICL]] — Alice Smith, 2026" in body
