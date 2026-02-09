"""
CLI 入口点

提供命令行接口，委托给 main.py 中的逻辑。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Optional

from paperbot.application.workflows.dailypaper import (
    DailyPaperReporter,
    build_daily_paper_report,
    enrich_daily_paper_report,
    normalize_llm_features,
    normalize_output_formats,
    render_daily_paper_markdown,
)


def create_parser() -> argparse.ArgumentParser:
    """创建 CLI 参数解析器"""
    parser = argparse.ArgumentParser(
        prog="paperbot",
        description="PaperBot - 学者追踪与论文分析系统",
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # track 命令
    track_parser = subparsers.add_parser("track", help="追踪学者论文")
    track_parser.add_argument("--scholar", "-s", help="学者名称")
    track_parser.add_argument("--scholar-id", help="Semantic Scholar ID")
    track_parser.add_argument("--config", "-c", help="配置文件路径")
    track_parser.add_argument("--output", "-o", help="输出目录")

    # analyze 命令
    analyze_parser = subparsers.add_parser("analyze", help="分析单篇论文")
    analyze_parser.add_argument("paper_id", help="论文 ID 或 DOI")
    analyze_parser.add_argument("--output", "-o", help="输出目录")

    # score 命令
    score_parser = subparsers.add_parser("score", help="快速计算影响力评分")
    score_parser.add_argument("paper_id", help="论文 ID 或 DOI")

    # topic-search 命令
    topic_search_parser = subparsers.add_parser(
        "topic-search", help="按主题检索 papers.cool 并输出聚合结果"
    )
    topic_search_parser.add_argument(
        "--query",
        "-q",
        action="append",
        dest="queries",
        help="检索主题，可重复指定多次",
    )
    topic_search_parser.add_argument(
        "--branch",
        action="append",
        dest="branches",
        choices=["arxiv", "venue"],
        help="检索分支，可重复指定；默认 arxiv+venue",
    )
    topic_search_parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        default=None,
        help="数据源名称，可重复指定；默认 papers_cool",
    )
    topic_search_parser.add_argument("--top-k", type=int, default=5, help="每个主题保留的结果数")
    topic_search_parser.add_argument(
        "--show", type=int, default=25, help="每个分支请求的候选结果数量"
    )
    topic_search_parser.add_argument("--json", action="store_true", help="输出完整 JSON")

    # daily-paper 命令
    daily_parser = subparsers.add_parser(
        "daily-paper", help="生成 DailyPaper 风格日报（markdown/json）"
    )
    daily_parser.add_argument(
        "--query",
        "-q",
        action="append",
        dest="queries",
        help="检索主题，可重复指定多次",
    )
    daily_parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        default=None,
        help="数据源名称，可重复指定；默认 papers_cool",
    )
    daily_parser.add_argument(
        "--branch",
        action="append",
        dest="branches",
        choices=["arxiv", "venue"],
        help="检索分支，可重复指定；默认 arxiv+venue",
    )
    daily_parser.add_argument("--top-k", type=int, default=5, help="每个主题保留的结果数")
    daily_parser.add_argument("--show", type=int, default=25, help="每个分支请求的候选结果数量")
    daily_parser.add_argument("--top-n", type=int, default=10, help="日报中保留的 top 项数量")
    daily_parser.add_argument("--title", default="DailyPaper Digest", help="日报标题")
    daily_parser.add_argument(
        "--format",
        action="append",
        dest="formats",
        default=None,
        help="输出格式：markdown/json/both（可重复指定）",
    )
    daily_parser.add_argument("--output-dir", default="./reports/dailypaper", help="输出目录")
    daily_parser.add_argument("--save", action="store_true", help="将日报写入文件")
    daily_parser.add_argument("--json", action="store_true", help="打印 JSON 报告")
    daily_parser.add_argument("--with-llm", action="store_true", help="启用 LLM 增强分析")
    daily_parser.add_argument(
        "--llm-feature",
        action="append",
        dest="llm_features",
        default=None,
        help="LLM 功能：summary/trends/insight/relevance（可重复指定）",
    )

    # version
    parser.add_argument("--version", "-v", action="store_true", help="显示版本")

    return parser


def run_cli(args: Optional[list] = None) -> int:
    """
    运行 CLI

    Args:
        args: 命令行参数（默认使用 sys.argv）

    Returns:
        退出码
    """
    parser = create_parser()
    parsed = parser.parse_args(args)

    if parsed.version:
        print("PaperBot v1.0.0")
        return 0

    if not parsed.command:
        parser.print_help()
        return 0

    # 委托给主模块处理
    try:
        # 延迟导入以避免循环依赖
        if parsed.command == "track":
            from main import main as run_main

            # 构造兼容的参数
            sys.argv = ["main.py", "--mode", "scholar"]
            if parsed.scholar:
                sys.argv.extend(["--scholar", parsed.scholar])
            if parsed.config:
                sys.argv.extend(["--config", parsed.config])
            if parsed.output:
                sys.argv.extend(["--output", parsed.output])
            run_main()

        elif parsed.command == "analyze":
            from main import main as run_main

            sys.argv = ["main.py", "--mode", "analyze", "--paper-id", parsed.paper_id]
            if parsed.output:
                sys.argv.extend(["--output", parsed.output])
            run_main()

        elif parsed.command == "score":
            asyncio.run(_quick_score(parsed.paper_id))

        elif parsed.command == "topic-search":
            return _run_topic_search(parsed)

        elif parsed.command == "daily-paper":
            return _run_daily_paper(parsed)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


async def _quick_score(paper_id: str):
    """快速计算论文评分"""
    try:
        from paperbot.workflows import ScholarTrackingWorkflow
    except ImportError:
        print("Error: ScholarTrackingWorkflow not available")
        return

    print(f"Fetching paper: {paper_id}...")

    # TODO: 实现 Semantic Scholar 客户端调用
    print(f"Quick score for {paper_id} is not yet implemented.")
    print("Please use the main.py entry point instead.")


def _create_topic_search_workflow():
    from paperbot.application.workflows.paperscool_topic_search import PapersCoolTopicSearchWorkflow

    return PapersCoolTopicSearchWorkflow()


def _run_topic_search(parsed: argparse.Namespace) -> int:
    queries = list(parsed.queries or [])
    if not queries:
        queries = ["ICL压缩", "ICL隐式偏置", "KV Cache加速"]

    branches = parsed.branches or ["arxiv", "venue"]
    sources = parsed.sources or ["papers_cool"]

    workflow = _create_topic_search_workflow()
    result = workflow.run(
        queries=queries,
        sources=sources,
        branches=branches,
        top_k_per_query=max(1, int(parsed.top_k)),
        show_per_branch=max(1, int(parsed.show)),
    )

    if parsed.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print(f"source: {result['source']}")
    print(f"fetched_at: {result['fetched_at']}")
    print(f"unique_items: {result['summary']['unique_items']}")
    for row in result.get("summary", {}).get("query_highlights", []):
        print(
            f"- {row['normalized_query']}: {row['hit_count']} hits"
            + (f" | top: {row['top_title']}" if row.get("top_title") else "")
        )
    return 0


def _run_daily_paper(parsed: argparse.Namespace) -> int:
    queries = list(parsed.queries or [])
    if not queries:
        queries = ["ICL压缩", "ICL隐式偏置", "KV Cache加速"]

    branches = parsed.branches or ["arxiv", "venue"]
    sources = parsed.sources or ["papers_cool"]

    workflow = _create_topic_search_workflow()
    search_result = workflow.run(
        queries=queries,
        sources=sources,
        branches=branches,
        top_k_per_query=max(1, int(parsed.top_k)),
        show_per_branch=max(1, int(parsed.show)),
    )

    report = build_daily_paper_report(
        search_result=search_result,
        title=parsed.title,
        top_n=max(1, int(parsed.top_n)),
    )
    llm_enabled = bool(parsed.with_llm)
    llm_features = normalize_llm_features(parsed.llm_features or ["summary"])
    if llm_enabled:
        report = enrich_daily_paper_report(
            report,
            llm_features=llm_features or ["summary"],
        )
    markdown = render_daily_paper_markdown(report)

    markdown_path = None
    json_path = None
    if parsed.save:
        reporter = DailyPaperReporter(output_dir=parsed.output_dir)
        artifacts = reporter.write(
            report=report,
            markdown=markdown,
            formats=normalize_output_formats(parsed.formats or ["both"]),
            slug=parsed.title,
        )
        markdown_path = artifacts.markdown_path
        json_path = artifacts.json_path

    if parsed.json:
        payload = {
            "report": report,
            "markdown": markdown,
            "markdown_path": markdown_path,
            "json_path": json_path,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"daily title: {report['title']}")
    print(f"date: {report['date']}")
    print(f"unique_items: {report['stats']['unique_items']}")
    print(f"query_count: {report['stats']['query_count']}")
    if llm_enabled:
        print(f"llm_analysis: enabled ({', '.join(llm_features or ['summary'])})")
    if markdown_path or json_path:
        print(f"saved markdown: {markdown_path}")
        print(f"saved json: {json_path}")
    print("\n" + markdown)
    return 0


if __name__ == "__main__":
    sys.exit(run_cli())
