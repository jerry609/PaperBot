#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class SampleRow:
    report_file: str
    query: str
    title: str
    url: str
    has_highlight: bool
    has_method: bool
    has_finding: bool
    has_tags: bool
    has_main_figure: bool


def _iter_items(report: Dict[str, Any], report_name: str) -> List[SampleRow]:
    rows: List[SampleRow] = []
    for query in report.get("queries") or []:
        query_name = str(query.get("normalized_query") or query.get("raw_query") or "").strip()
        for item in query.get("top_items") or []:
            digest_card = item.get("digest_card") if isinstance(item, dict) else {}
            if not isinstance(digest_card, dict):
                digest_card = {}
            tags = digest_card.get("tags") or []
            rows.append(
                SampleRow(
                    report_file=report_name,
                    query=query_name,
                    title=str(item.get("title") or "").strip(),
                    url=str(item.get("url") or item.get("external_url") or "").strip(),
                    has_highlight=bool(str(digest_card.get("highlight") or "").strip()),
                    has_method=bool(str(digest_card.get("method") or "").strip()),
                    has_finding=bool(str(digest_card.get("finding") or "").strip()),
                    has_tags=bool(tags),
                    has_main_figure=bool(item.get("main_figure")),
                )
            )
    return rows


def _load_rows(reports_dir: Path) -> List[SampleRow]:
    rows: List[SampleRow] = []
    for file in sorted(reports_dir.glob("*.json"), reverse=True):
        try:
            payload = json.loads(file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        rows.extend(_iter_items(payload, file.name))
    return rows


def _write_markdown_report(rows: List[SampleRow], output_path: Path) -> None:
    total = len(rows)
    if total <= 0:
        output_path.write_text("# Daily Push Audit\n\nNo sampled items.\n", encoding="utf-8")
        return

    def _ratio(count: int) -> str:
        pct = (count / total) * 100
        return f"{count}/{total} ({pct:.1f}%)"

    highlight = sum(1 for r in rows if r.has_highlight)
    method = sum(1 for r in rows if r.has_method)
    finding = sum(1 for r in rows if r.has_finding)
    tags = sum(1 for r in rows if r.has_tags)
    figure = sum(1 for r in rows if r.has_main_figure)

    lines = [
        "# Daily Push Audit",
        "",
        "## Coverage",
        "",
        f"- highlight coverage: {_ratio(highlight)}",
        f"- method coverage: {_ratio(method)}",
        f"- finding coverage: {_ratio(finding)}",
        f"- tags coverage: {_ratio(tags)}",
        f"- main figure coverage: {_ratio(figure)}",
        "",
        "## Manual Review Checklist",
        "",
        "- Validate highlight/method/finding semantic correctness (binary pass/fail).",
        "- Check whether main figure matches architecture/framework intent.",
        "- Mark repeated failure patterns for prompt/rule tuning.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manual_csv(rows: List[SampleRow], output_path: Path) -> None:
    fieldnames = [
        "report_file",
        "query",
        "title",
        "url",
        "has_highlight",
        "has_method",
        "has_finding",
        "has_tags",
        "has_main_figure",
        "human_highlight_ok",
        "human_method_ok",
        "human_finding_ok",
        "human_figure_ok",
        "notes",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "report_file": row.report_file,
                    "query": row.query,
                    "title": row.title,
                    "url": row.url,
                    "has_highlight": int(row.has_highlight),
                    "has_method": int(row.has_method),
                    "has_finding": int(row.has_finding),
                    "has_tags": int(row.has_tags),
                    "has_main_figure": int(row.has_main_figure),
                    "human_highlight_ok": "",
                    "human_method_ok": "",
                    "human_finding_ok": "",
                    "human_figure_ok": "",
                    "notes": "",
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Daily Push report quality coverage.")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports/dailypaper"),
        help="directory with daily report JSON files",
    )
    parser.add_argument("--sample-size", type=int, default=20, help="sample item count")
    parser.add_argument("--seed", type=int, default=20260303, help="random seed for deterministic sampling")
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("reports/dailypaper_audit.md"),
        help="markdown summary output path",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("reports/dailypaper_manual_review.csv"),
        help="manual review CSV output path",
    )
    args = parser.parse_args()

    rows = _load_rows(args.reports_dir)
    if rows:
        random.Random(args.seed).shuffle(rows)
    sampled = rows[: max(0, int(args.sample_size))]

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown_report(sampled, args.output_md)
    _write_manual_csv(sampled, args.output_csv)

    print(f"sampled={len(sampled)} total={len(rows)}")
    print(f"markdown={args.output_md}")
    print(f"csv={args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
