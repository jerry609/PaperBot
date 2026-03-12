"""
CLI 入口点

提供命令行接口，委托给 main.py 中的逻辑。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import find_dotenv, load_dotenv

from config.settings import create_settings
from paperbot.application.workflows.dailypaper import (
    DailyPaperReporter,
    apply_judge_scores_to_report,
    build_daily_paper_report,
    enrich_daily_paper_report,
    extract_figures_for_report,
    ingest_daily_report_to_registry,
    normalize_llm_features,
    persist_judge_scores_to_registry,
    normalize_output_formats,
    render_daily_paper_markdown,
)
from paperbot.application.services.daily_push_service import DailyPushService
from paperbot.application.workflows.unified_topic_search import (
    make_default_search_service,
    run_unified_topic_search,
)
from paperbot.infrastructure.stores.paper_store import PaperStore
from paperbot.infrastructure.stores.pipeline_session_store import PipelineSessionStore

# Load local .env automatically for CLI workflows using LLM providers.
load_dotenv(find_dotenv(usecwd=True), override=False)


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
        help="数据源名称，可重复指定；默认 papers_cool（可选 arxiv_api / hf_daily）",
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
        help="数据源名称，可重复指定；默认 papers_cool（可选 arxiv_api / hf_daily）",
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
    daily_parser.add_argument("--with-judge", action="store_true", help="启用 LLM-as-Judge 评分")
    daily_parser.add_argument(
        "--judge-runs", type=int, default=1, help="Judge 重复评分次数（取中位）"
    )
    daily_parser.add_argument(
        "--judge-max-items",
        type=int,
        default=5,
        help="每个 query 最多执行 Judge 的论文数",
    )
    daily_parser.add_argument(
        "--judge-token-budget",
        type=int,
        default=0,
        help="Judge 的 token 预算上限（0 表示不限制）",
    )
    daily_parser.add_argument(
        "--with-figures",
        action="store_true",
        help="启用 MinerU 图表提取（需要 --mineru-api-key 或 MINERU_API_KEY 环境变量）",
    )
    daily_parser.add_argument(
        "--mineru-api-key",
        default=None,
        help="MinerU Cloud API Key（也可通过 MINERU_API_KEY 环境变量设置）",
    )
    daily_parser.add_argument(
        "--mineru-api-base-url",
        default=None,
        help="MinerU API Base URL（默认 https://mineru.net/api/v4，也可通过 MINERU_API_BASE_URL 设置）",
    )
    daily_parser.add_argument(
        "--mineru-model-version",
        default=None,
        help="MinerU model_version（默认 vlm，也可通过 MINERU_MODEL_VERSION 设置）",
    )
    daily_parser.add_argument(
        "--mineru-max-wait-seconds",
        type=float,
        default=None,
        help="MinerU 任务轮询最长等待秒数（默认 180，也可通过 MINERU_MAX_WAIT_SECONDS 设置）",
    )
    daily_parser.add_argument(
        "--figures-max-items",
        type=int,
        default=5,
        help="每次图表提取最多处理的论文数",
    )
    daily_parser.add_argument("--notify", action="store_true", help="生成后推送日报通知")
    daily_parser.add_argument(
        "--notify-channel",
        action="append",
        dest="notify_channels",
        default=None,
        help="推送渠道：email/slack/dingding（可重复指定）",
    )
    daily_parser.add_argument(
        "--session-id",
        default=None,
        help="会话 ID（用于断点恢复；不传则自动生成）",
    )
    daily_parser.add_argument(
        "--resume",
        action="store_true",
        help="从最近 checkpoint 恢复执行",
    )

    export_parser = subparsers.add_parser("export", help="导出 PaperBot 数据")
    export_subparsers = export_parser.add_subparsers(dest="export_target", help="导出目标")

    obsidian_parser = export_subparsers.add_parser(
        "obsidian", help="导出已保存论文到 Obsidian vault"
    )
    obsidian_parser.add_argument(
        "--vault",
        default=None,
        help="Obsidian vault 目录（默认读取 obsidian.vault_path）",
    )
    obsidian_scope = obsidian_parser.add_mutually_exclusive_group()
    obsidian_scope.add_argument("--track-id", type=int, default=None, help="按 track ID 导出")
    obsidian_scope.add_argument(
        "--track-name",
        default=None,
        help="按 track 名称导出（大小写不敏感）",
    )
    obsidian_parser.add_argument("--user-id", default="default", help="用户 ID")
    obsidian_parser.add_argument("--limit", type=int, default=200, help="最多导出多少篇论文")
    obsidian_parser.add_argument(
        "--root-dir",
        default=None,
        help="vault 内的输出根目录（默认读取 obsidian.root_dir）",
    )
    obsidian_parser.add_argument(
        "--paper-template",
        default=None,
        help="自定义论文笔记 Jinja2 模板路径（默认读取 obsidian.paper_template_path）",
    )
    obsidian_parser.add_argument("--json", action="store_true", help="输出 JSON 摘要")

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

        elif parsed.command == "export":
            if parsed.export_target == "obsidian":
                return _run_obsidian_export(parsed)
            print("Error: export target is required", file=sys.stderr)
            return 1

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


_paper_search_service = None


def _get_paper_search_service():
    global _paper_search_service
    if _paper_search_service is None:
        _paper_search_service = make_default_search_service(registry=PaperStore())
    return _paper_search_service


def _run_topic_search(parsed: argparse.Namespace) -> int:
    queries = list(parsed.queries or [])
    if not queries:
        queries = ["ICL压缩", "ICL隐式偏置", "KV Cache加速"]

    branches = parsed.branches or ["arxiv", "venue"]
    sources = parsed.sources or ["papers_cool"]

    result = asyncio.run(
        run_unified_topic_search(
            queries=queries,
            sources=sources,
            branches=branches,
            top_k_per_query=max(1, int(parsed.top_k)),
            show_per_branch=max(1, int(parsed.show)),
            search_service=_get_paper_search_service(),
            persist=False,
        )
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

    session_store = PipelineSessionStore()
    session = session_store.start_session(
        workflow="cli_daily_paper",
        payload={k: getattr(parsed, k) for k in vars(parsed)},
        session_id=getattr(parsed, "session_id", None),
        resume=bool(getattr(parsed, "resume", False)),
    )
    session_id = str(session.get("session_id") or "")
    state: dict = session.get("state") if getattr(parsed, "resume", False) else {}

    if getattr(parsed, "resume", False) and session.get("status") == "completed":
        done = session.get("result") if isinstance(session.get("result"), dict) else {}
        if done:
            if parsed.json:
                print(json.dumps(done, ensure_ascii=False, indent=2))
            else:
                print(f"resumed session: {session_id}")
                print("session already completed; returning cached result")
            return 0

    if isinstance(state.get("report"), dict):
        report = dict(state.get("report") or {})
        search_result = (
            state.get("search_result") if isinstance(state.get("search_result"), dict) else {}
        )
    else:
        effective_top_k = max(1, int(parsed.top_k), int(parsed.top_n))
        search_result = asyncio.run(
            run_unified_topic_search(
                queries=queries,
                sources=sources,
                branches=branches,
                top_k_per_query=effective_top_k,
                show_per_branch=max(1, int(parsed.show)),
                search_service=_get_paper_search_service(),
                persist=False,
            )
        )

        report = build_daily_paper_report(
            search_result=search_result,
            title=parsed.title,
            top_n=max(1, int(parsed.top_n)),
        )
        session_store.save_checkpoint(
            session_id=session_id,
            checkpoint="report_built",
            state={"search_result": search_result, "report": report},
        )
    llm_enabled = bool(parsed.with_llm)
    llm_features = normalize_llm_features(parsed.llm_features or ["summary"])
    if llm_enabled:
        report = enrich_daily_paper_report(
            report,
            llm_features=llm_features or ["summary"],
        )

    judge_enabled = bool(parsed.with_judge)
    if judge_enabled:
        report = apply_judge_scores_to_report(
            report,
            max_items_per_query=max(1, int(parsed.judge_max_items)),
            n_runs=max(1, int(parsed.judge_runs)),
            judge_token_budget=max(0, int(parsed.judge_token_budget)),
        )

    figures_enabled = bool(parsed.with_figures)
    if figures_enabled:
        mineru_key = parsed.mineru_api_key or os.getenv("MINERU_API_KEY", "")
        if mineru_key:
            mineru_base_url = parsed.mineru_api_base_url or os.getenv("MINERU_API_BASE_URL", "")
            mineru_model_version = parsed.mineru_model_version or os.getenv(
                "MINERU_MODEL_VERSION", "vlm"
            )
            try:
                mineru_max_wait_seconds = (
                    float(parsed.mineru_max_wait_seconds)
                    if parsed.mineru_max_wait_seconds is not None
                    else float(os.getenv("MINERU_MAX_WAIT_SECONDS", "180"))
                )
            except (TypeError, ValueError):
                mineru_max_wait_seconds = 180.0
            report = extract_figures_for_report(
                report,
                api_key=mineru_key,
                max_items=max(1, int(parsed.figures_max_items)),
                base_url=mineru_base_url,
                model_version=mineru_model_version,
                max_wait_seconds=max(5.0, mineru_max_wait_seconds),
            )
        else:
            print("warning: --with-figures requires --mineru-api-key or MINERU_API_KEY env var",
                  file=sys.stderr)

    try:
        report["registry_ingest"] = ingest_daily_report_to_registry(report)
    except Exception as exc:
        report["registry_ingest"] = {"error": str(exc)}

    if judge_enabled:
        try:
            report["judge_registry_ingest"] = persist_judge_scores_to_registry(report)
        except Exception as exc:
            report["judge_registry_ingest"] = {"error": str(exc)}

    markdown = render_daily_paper_markdown(report)

    markdown_path = None
    json_path = None
    notify_result = None
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

    if parsed.notify:
        notify_service = DailyPushService.from_env()
        notify_result = notify_service.push_dailypaper(
            report=report,
            markdown=markdown,
            markdown_path=markdown_path,
            json_path=json_path,
            channels_override=parsed.notify_channels,
        )

    session_store.save_result(
        session_id=session_id,
        status="completed",
        result={
            "session_id": session_id,
            "report": report,
            "markdown": markdown,
            "markdown_path": markdown_path,
            "json_path": json_path,
            "notify": notify_result,
        },
    )

    if parsed.json:
        payload = {
            "session_id": session_id,
            "report": report,
            "markdown": markdown,
            "markdown_path": markdown_path,
            "json_path": json_path,
            "notify": notify_result,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"daily title: {report['title']}")
    print(f"date: {report['date']}")
    print(f"unique_items: {report['stats']['unique_items']}")
    print(f"query_count: {report['stats']['query_count']}")
    if llm_enabled:
        print(f"llm_analysis: enabled ({', '.join(llm_features or ['summary'])})")
    if judge_enabled:
        print(
            f"judge: enabled (runs={max(1, int(parsed.judge_runs))}, "
            f"max_items={max(1, int(parsed.judge_max_items))}, "
            f"token_budget={max(0, int(parsed.judge_token_budget))})"
        )
    if figures_enabled:
        fig_count = sum(
            1 for q in (report.get("queries") or [])
            for item in (q.get("top_items") or [])
            if item.get("main_figure")
        )
        print(f"figures: extracted main_figure for {fig_count} papers")
    if markdown_path or json_path:
        print(f"saved markdown: {markdown_path}")
        print(f"saved json: {json_path}")
    if parsed.notify:
        print(f"notify: {json.dumps(notify_result, ensure_ascii=False)}")
    print("\n" + markdown)
    return 0


def _find_track_by_name(
    store,
    *,
    user_id: str,
    track_name: str,
) -> Optional[dict]:
    target = str(track_name or "").strip().casefold()
    if not target:
        return None

    for track in store.list_tracks(user_id=user_id, include_archived=True, limit=500):
        if str(track.get("name") or "").strip().casefold() == target:
            return track
    return None


def _run_obsidian_export(parsed: argparse.Namespace) -> int:
    from paperbot.infrastructure.exporters import ObsidianFilesystemExporter
    from paperbot.infrastructure.stores.research_store import SqlAlchemyResearchStore

    settings = create_settings()
    obsidian_config = settings.obsidian
    vault_value = parsed.vault or obsidian_config.vault_path
    if not str(vault_value or "").strip():
        print(
            "Error: vault path is required. Pass --vault or configure obsidian.vault_path.",
            file=sys.stderr,
        )
        return 1
    root_dir = parsed.root_dir or obsidian_config.root_dir or "PaperBot"
    paper_template_path = parsed.paper_template or obsidian_config.paper_template_path

    store = SqlAlchemyResearchStore()
    try:
        track = None
        track_id = None
        if parsed.track_id is not None:
            track = store.get_track(user_id=parsed.user_id, track_id=int(parsed.track_id))
            if track is None:
                print(f"Error: track not found: {parsed.track_id}", file=sys.stderr)
                return 1
            track_id = int(track["id"])
        elif parsed.track_name:
            track = _find_track_by_name(store, user_id=parsed.user_id, track_name=parsed.track_name)
            if track is None:
                print(f"Error: track not found: {parsed.track_name}", file=sys.stderr)
                return 1
            track_id = int(track["id"])

        saved_items = store.list_saved_papers(
            user_id=parsed.user_id,
            track_id=track_id,
            limit=max(1, int(parsed.limit)),
        )
        if not saved_items:
            scope = f" for track {track['name']}" if track else ""
            print(f"Error: no saved papers found{scope}", file=sys.stderr)
            return 1

        exporter = ObsidianFilesystemExporter()
        template_path = (
            Path(paper_template_path).expanduser() if paper_template_path else None
        )
        result = exporter.export_library_snapshot(
            vault_path=Path(vault_value),
            saved_items=saved_items,
            track=track,
            root_dir=root_dir,
            paper_template_path=template_path,
            track_moc_filename=getattr(obsidian_config, "track_moc_filename", "_MOC.md"),
            group_tracks_in_folders=getattr(obsidian_config, "group_tracks_in_folders", True),
        )

        if parsed.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0

        print(f"vault: {result['vault_path']}")
        print(f"root_dir: {result['root_dir']}")
        print(f"paper_count: {result['paper_count']}")
        if track is not None:
            print(f"track: {track['name']}")
        print(f"moc_note: {result['moc_note']}")
        if result.get("track_note"):
            print(f"track_note: {result['track_note']}")
        return 0
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if hasattr(store, "close"):
            store.close()


if __name__ == "__main__":
    sys.exit(run_cli())
