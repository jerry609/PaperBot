from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_daily_push_reports import _load_rows, _write_manual_csv, _write_markdown_report


def test_audit_helpers_extract_and_write_outputs(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "2026-03-03-digest.json").write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "normalized_query": "kv cache",
                        "top_items": [
                            {
                                "title": "FlashKV",
                                "url": "https://arxiv.org/abs/2601.00001",
                                "digest_card": {
                                    "highlight": "fast",
                                    "method": "compression",
                                    "finding": "2x",
                                    "tags": ["kv"],
                                },
                                "main_figure": {"url": "https://example.com/fig.png"},
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = _load_rows(reports_dir)
    assert len(rows) == 1
    assert rows[0].has_main_figure is True

    md_path = tmp_path / "audit.md"
    csv_path = tmp_path / "audit.csv"
    _write_markdown_report(rows, md_path)
    _write_manual_csv(rows, csv_path)

    assert "highlight coverage: 1/1" in md_path.read_text(encoding="utf-8")
    assert "human_highlight_ok" in csv_path.read_text(encoding="utf-8")
