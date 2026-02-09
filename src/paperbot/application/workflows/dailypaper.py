from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from paperbot.application.services.llm_service import LLMService, get_llm_service
from paperbot.application.workflows.analysis.paper_judge import PaperJudge


SUPPORTED_LLM_FEATURES = ("summary", "trends", "insight", "relevance")


def build_daily_paper_report(
    *,
    search_result: Dict[str, Any],
    title: str = "DailyPaper Digest",
    top_n: int = 10,
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    query_rows: List[Dict[str, Any]] = []

    for query in search_result.get("queries") or []:
        items = list(query.get("items") or [])[: max(0, int(top_n))]
        query_rows.append(
            {
                "raw_query": query.get("raw_query") or query.get("normalized_query") or "",
                "normalized_query": query.get("normalized_query") or "",
                "total_hits": int(query.get("total_hits") or 0),
                "top_items": items,
            }
        )

    global_top = list(search_result.get("items") or [])[: max(0, int(top_n))]
    summary = search_result.get("summary") or {}

    return {
        "title": title,
        "date": now.date().isoformat(),
        "generated_at": now.isoformat(),
        "source": search_result.get("source") or "papers.cool",
        "sources": search_result.get("sources") or ["papers_cool"],
        "stats": {
            "unique_items": int(summary.get("unique_items") or 0),
            "total_query_hits": int(summary.get("total_query_hits") or 0),
            "query_count": len(query_rows),
        },
        "queries": query_rows,
        "global_top": global_top,
    }


def enrich_daily_paper_report(
    report: Dict[str, Any],
    *,
    llm_service: Optional[LLMService] = None,
    llm_features: Sequence[str] = ("summary",),
    max_items_per_query: int = 3,
) -> Dict[str, Any]:
    """Optionally enrich DailyPaper report with LLM outputs."""

    features = normalize_llm_features(llm_features)
    if not features:
        return copy.deepcopy(report)

    svc = llm_service or get_llm_service()
    enriched = copy.deepcopy(report)
    llm_block: Dict[str, Any] = {
        "enabled": True,
        "features": features,
        "query_trends": [],
        "daily_insight": "",
    }

    for query in enriched.get("queries") or []:
        query_name = query.get("normalized_query") or query.get("raw_query") or ""
        top_items = (query.get("top_items") or [])[: max(1, int(max_items_per_query))]

        if "summary" in features:
            for item in top_items:
                item["ai_summary"] = svc.summarize_paper(
                    title=item.get("title") or "",
                    abstract=item.get("snippet") or item.get("abstract") or "",
                )

        if "relevance" in features:
            for item in top_items:
                item["relevance"] = svc.assess_relevance(paper=item, query=query_name)

        if "trends" in features and top_items:
            trend_text = svc.analyze_trends(topic=query_name, papers=top_items)
            llm_block["query_trends"].append({"query": query_name, "analysis": trend_text})

    if "insight" in features:
        llm_block["daily_insight"] = svc.generate_daily_insight(enriched)

    enriched["llm_analysis"] = llm_block
    return enriched


def estimate_judge_tokens_for_item(item: Dict[str, Any], *, n_runs: int = 1) -> int:
    """Estimate judge token usage to support lightweight budget controls."""

    title = item.get("title") or ""
    abstract = item.get("snippet") or item.get("abstract") or ""
    keywords = ", ".join(item.get("keywords") or [])

    content_chars = len(f"{title}\n{abstract}\n{keywords}")
    content_tokens = max(120, content_chars // 4)

    # Prompt/rubric text is relatively long; keep a conservative fixed overhead.
    base_prompt_tokens = 650
    expected_output_tokens = 280
    per_run = base_prompt_tokens + content_tokens + expected_output_tokens
    return per_run * max(1, int(n_runs))


def select_judge_candidates(
    report: Dict[str, Any],
    *,
    max_items_per_query: int = 5,
    n_runs: int = 1,
    token_budget: int = 0,
) -> Dict[str, Any]:
    """Pick candidate papers for judge scoring under per-query and token-budget limits."""

    capped_per_query = max(1, int(max_items_per_query))
    runs = max(1, int(n_runs))
    budget = max(0, int(token_budget))

    candidates: List[Dict[str, Any]] = []
    for query_index, query in enumerate(report.get("queries") or []):
        top_items = list(query.get("top_items") or [])
        capped = top_items[:capped_per_query]
        for item_index, item in enumerate(capped):
            try:
                base_score = float(item.get("score") or 0)
            except Exception:
                base_score = 0.0
            candidates.append(
                {
                    "query_index": query_index,
                    "item_index": item_index,
                    "estimated_tokens": estimate_judge_tokens_for_item(item, n_runs=runs),
                    "base_score": base_score,
                }
            )

    ranked = sorted(candidates, key=lambda row: row["base_score"], reverse=True)

    selected: List[Dict[str, Any]] = []
    consumed = 0
    for row in ranked:
        if budget > 0 and consumed + int(row["estimated_tokens"]) > budget:
            continue
        consumed += int(row["estimated_tokens"])
        selected.append(row)

    if budget <= 0:
        selected = ranked
        consumed = sum(int(row["estimated_tokens"]) for row in selected)

    selected_by_query: Dict[int, List[int]] = {}
    for row in selected:
        selected_by_query.setdefault(int(row["query_index"]), []).append(int(row["item_index"]))

    for key in list(selected_by_query.keys()):
        selected_by_query[key] = sorted(set(selected_by_query[key]))

    return {
        "selected": selected,
        "selected_by_query": selected_by_query,
        "budget": {
            "token_budget": budget,
            "estimated_tokens": int(consumed),
            "candidate_items": len(candidates),
            "judged_items": len(selected),
            "skipped_due_budget": max(0, len(candidates) - len(selected)),
        },
    }


def apply_judge_scores_to_report(
    report: Dict[str, Any],
    *,
    llm_service: Optional[LLMService] = None,
    max_items_per_query: int = 5,
    n_runs: int = 1,
    judge_token_budget: int = 0,
) -> Dict[str, Any]:
    """Evaluate papers with LLM-as-Judge and attach per-paper judgment metadata."""

    judged = copy.deepcopy(report)
    svc = llm_service or get_llm_service()
    judge = PaperJudge(llm_service=svc)

    recommendation_count = {
        "must_read": 0,
        "worth_reading": 0,
        "skim": 0,
        "skip": 0,
    }

    selection = select_judge_candidates(
        judged,
        max_items_per_query=max_items_per_query,
        n_runs=n_runs,
        token_budget=judge_token_budget,
    )
    selected_by_query = selection["selected_by_query"]

    for query_index, query in enumerate(judged.get("queries") or []):
        query_name = query.get("normalized_query") or query.get("raw_query") or ""
        top_items = list(query.get("top_items") or [])
        if not top_items:
            continue

        chosen_indices = selected_by_query.get(query_index, [])
        chosen_items = [top_items[idx] for idx in chosen_indices if 0 <= idx < len(top_items)]
        if not chosen_items:
            continue

        judgments = judge.judge_batch(
            papers=chosen_items, query=query_name, n_runs=max(1, int(n_runs))
        )
        for item_index, judgment in zip(chosen_indices, judgments):
            item = top_items[item_index]
            j_payload = judgment.to_dict()
            item["judge"] = j_payload
            rec = j_payload.get("recommendation")
            if rec in recommendation_count:
                recommendation_count[rec] += 1

        capped_count = min(len(top_items), max(1, int(max_items_per_query)))
        capped = top_items[:capped_count]
        capped.sort(
            key=lambda it: float((it.get("judge") or {}).get("overall") or -1), reverse=True
        )
        query["top_items"] = capped + top_items[capped_count:]

    judged["judge"] = {
        "enabled": True,
        "max_items_per_query": int(max_items_per_query),
        "n_runs": int(max(1, int(n_runs))),
        "recommendation_count": recommendation_count,
        "budget": selection["budget"],
    }
    return judged


def render_daily_paper_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"# {report.get('title') or 'DailyPaper Digest'}")
    lines.append("")
    lines.append(f"- Date: {report.get('date')}")
    lines.append(f"- Generated At (UTC): {report.get('generated_at')}")
    lines.append(f"- Source: {report.get('source')}")
    lines.append(f"- Sources: {', '.join(report.get('sources') or [])}")
    stats = report.get("stats") or {}
    lines.append(f"- Unique Items: {stats.get('unique_items', 0)}")
    lines.append(f"- Total Query Hits: {stats.get('total_query_hits', 0)}")
    lines.append("")

    lines.append("## Query Highlights")
    lines.append("")
    for query in report.get("queries") or []:
        normalized = query.get("normalized_query") or ""
        total_hits = query.get("total_hits") or 0
        lines.append(f"### {normalized} ({total_hits} hits)")
        top_items = query.get("top_items") or []
        if not top_items:
            lines.append("- No hits")
            lines.append("")
            continue
        for item in top_items[:5]:
            title = item.get("title") or "Untitled"
            url = item.get("url") or item.get("external_url") or ""
            score = item.get("score")
            if url:
                lines.append(f"- [{title}]({url}) | score={score}")
            else:
                lines.append(f"- {title} | score={score}")

            ai_summary = (item.get("ai_summary") or "").strip()
            if ai_summary:
                lines.append(f"  - AI Summary: {ai_summary}")

            relevance = item.get("relevance")
            if isinstance(relevance, dict):
                rel_score = relevance.get("score")
                rel_reason = relevance.get("reason") or ""
                lines.append(f"  - Relevance: score={rel_score} reason={rel_reason}")

            judge = item.get("judge")
            if isinstance(judge, dict):
                overall = judge.get("overall")
                rec = judge.get("recommendation")
                lines.append(f"  - Judge: overall={overall} recommendation={rec}")
                one_line = judge.get("one_line_summary") or ""
                if one_line:
                    lines.append(f"  - Judge Summary: {one_line}")
        lines.append("")

    lines.append("## Global Top")
    lines.append("")
    for idx, item in enumerate(report.get("global_top") or [], start=1):
        title = item.get("title") or "Untitled"
        url = item.get("url") or item.get("external_url") or ""
        score = item.get("score")
        matched_queries = ", ".join(item.get("matched_queries") or [])
        if url:
            lines.append(f"{idx}. [{title}]({url}) | score={score} | queries={matched_queries}")
        else:
            lines.append(f"{idx}. {title} | score={score} | queries={matched_queries}")

    if not (report.get("global_top") or []):
        lines.append("- No items")

    llm_analysis = report.get("llm_analysis") or {}
    if llm_analysis:
        lines.append("")
        lines.append("## LLM Insights")
        lines.append("")

        features = ", ".join(llm_analysis.get("features") or [])
        if features:
            lines.append(f"- Enabled Features: {features}")

        daily_insight = (llm_analysis.get("daily_insight") or "").strip()
        if daily_insight:
            lines.append(f"- Daily Insight: {daily_insight}")

        trends = llm_analysis.get("query_trends") or []
        if trends:
            lines.append("")
            lines.append("### Query Trends")
            for trend in trends:
                topic = trend.get("query") or ""
                text = trend.get("analysis") or ""
                lines.append(f"- {topic}: {text}")

    judge_block = report.get("judge") or {}
    if judge_block:
        lines.append("")
        lines.append("## Judge Summary")
        lines.append("")
        recommendation_count = judge_block.get("recommendation_count") or {}
        for key in ("must_read", "worth_reading", "skim", "skip"):
            lines.append(f"- {key}: {recommendation_count.get(key, 0)}")

    lines.append("")
    return "\n".join(lines)


@dataclass
class DailyPaperArtifacts:
    report: Dict[str, Any]
    markdown: str
    markdown_path: Optional[str] = None
    json_path: Optional[str] = None


class DailyPaperReporter:
    def __init__(self, output_dir: str = "./reports/dailypaper"):
        self.output_dir = Path(output_dir)

    def write(
        self,
        *,
        report: Dict[str, Any],
        markdown: str,
        formats: Sequence[str] = ("markdown", "json"),
        slug: Optional[str] = None,
    ) -> DailyPaperArtifacts:
        formats_set = {fmt.lower().strip() for fmt in formats if fmt.strip()}
        if not formats_set:
            formats_set = {"markdown", "json"}

        day = report.get("date") or datetime.now(timezone.utc).date().isoformat()
        safe_slug = _safe_slug(slug or report.get("title") or "dailypaper")
        stem = f"{day}-{safe_slug}"

        self.output_dir.mkdir(parents=True, exist_ok=True)

        md_path: Optional[Path] = None
        json_path: Optional[Path] = None

        if "markdown" in formats_set:
            md_path = self.output_dir / f"{stem}.md"
            md_path.write_text(markdown, encoding="utf-8")

        if "json" in formats_set:
            json_path = self.output_dir / f"{stem}.json"
            json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        return DailyPaperArtifacts(
            report=report,
            markdown=markdown,
            markdown_path=str(md_path) if md_path else None,
            json_path=str(json_path) if json_path else None,
        )


def _safe_slug(text: str) -> str:
    lowered = (text or "").strip().lower()
    lowered = re.sub(r"\s+", "-", lowered)
    lowered = re.sub(r"[^a-z0-9\-_]+", "-", lowered)
    lowered = re.sub(r"-+", "-", lowered).strip("-")
    return lowered or "daily"


def normalize_output_formats(formats: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for fmt in formats:
        key = (fmt or "").strip().lower()
        if key == "both":
            for item in ("markdown", "json"):
                if item not in seen:
                    seen.add(item)
                    normalized.append(item)
            continue
        if key in {"markdown", "json"} and key not in seen:
            seen.add(key)
            normalized.append(key)
    return normalized or ["markdown", "json"]


def normalize_llm_features(features: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for feature in features:
        key = (feature or "").strip().lower()
        if key in SUPPORTED_LLM_FEATURES and key not in seen:
            seen.add(key)
            normalized.append(key)
    return normalized
