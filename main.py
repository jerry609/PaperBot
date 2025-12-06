#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SecuriPaperBot - 主启动脚本
一个简化的启动入口，避免复杂的包导入问题
"""

import sys
import os
import argparse
import asyncio
import time
from pathlib import Path
from typing import Optional
import logging

# 解决 Windows 上 curl_cffi 的兼容性问题
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 添加当前目录到Python路径，解决导入问题
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        print(f"❌ 错误: 需要Python 3.8+，当前版本: {sys.version}")
            # 异步执行下载（与其他会议相同的逻辑）
        return False
    return True

def check_dependencies():
    """检查必要的依赖"""
    required_packages = [
        'requests', 'lxml', 'urllib3', 'aiohttp', 'bs4'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 缺少必要依赖: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    return True



async def _sequential_download(downloader, valid_papers):
    """顺序下载论文并显示进度"""
    download_results = []
    success_count = 0
    cached_count = 0
    start_time = time.time()

    print(f"📄 开始逐个下载 {len(valid_papers)} 篇论文")

    for idx, paper in enumerate(valid_papers):
        paper_start_time = time.time()
        try:
            result = await downloader.download_paper(
                paper['url'],
                paper.get('title', f'paper_{idx}'),
                paper_index=idx,
                total_papers=len(valid_papers)
            )
            paper_time = time.time() - paper_start_time

            if result and result.get('success'):
                if result.get('cached'):
                    print(f"📋 缓存命中 (耗时: {paper_time:.1f}s)")
                    cached_count += 1
                else:
                    size_kb = result.get('size', 0) / 1024
                    print(f"✅ 下载成功 (耗时: {paper_time:.1f}s, 大小: {size_kb:.1f}KB)")
                success_count += 1
                download_results.append(result)
            else:
                error_msg = result.get('error', '未知错误') if result else '下载失败'
                print(f"❌ 下载失败: {error_msg}")
                download_results.append(result or {'success': False})

            if idx < len(valid_papers) - 1:
                await asyncio.sleep(2)  # 避免请求过于频繁

        except Exception as e:
            print(f"❌ 下载异常: {str(e)}")
            download_results.append({'success': False, 'error': str(e)})

    total_time = time.time() - start_time
    print("\n" + "🎉 下载完成统计:")
    print(f"✅ 成功下载: {success_count}/{len(download_results)} 篇论文")
    print(f"📋 缓存命中: {cached_count} 篇")
    print(f"⏱️  总耗时: {total_time:.1f} 秒")
    print(f"📁 文件存储在: {downloader.download_path}")

    if success_count > 0:
        avg_time = total_time / success_count
        print(f"📊 平均速度: {avg_time:.2f} 秒/篇")

    success_rate = (success_count / len(valid_papers)) if valid_papers else 0
    print(f"📈 成功率: {success_rate:.1%}")


def simple_paper_download(conference: str, year: str, url: Optional[str] = None, smart_mode: bool = False):
    """
    简化的论文下载功能，根据会议类型选择合适的下载器。
    - CCS会议使用专用的 `downloader_ccs`。
    - 其他会议使用通用的 `downloader`，并支持智能并发模式。
    """
    print(f"🚀 开始下载 {conference.upper()} {year} 年论文...")

    is_ccs = conference.lower() == 'ccs'
    
    # 根据会议类型选择下载器
    if is_ccs:
        from utils.downloader_ccs import PaperDownloader as DownloaderClass
        print("📚 目标会议: CCS (使用专用解析逻辑)")
        if smart_mode:
            print("ℹ️  CCS 下载目前不支持智能模式，将使用稳定顺序模式。")
            smart_mode = False  # 强制为顺序模式
    else:
    from agents.conference_research_agent import ConferenceResearchAgent

    async def _run_download():
        try:
            agent = ConferenceResearchAgent({"download_path": f'./papers/{conference}_{year}'})
            result = await agent.process(conference, year)
            papers = result.get("papers", [])
            print(f"✅ 找到 {len(papers)} 篇论文")
            with_pdf = [p for p in papers if p.get("local_path")]
            print(f"📝 下载成功 {len(with_pdf)} 篇；含代码链接 {sum(1 for p in papers if p.get('github_links'))}")
        except Exception as e:
            print(f"❌ 下载过程中出现严重错误: {e}")

    try:
        asyncio.run(_run_download())
        print("✅ 下载任务完成")
    except Exception as e:
        print(f"❌ 异步执行失败: {e}")
        print("💡 请确保您有网络访问权限和正确的会议信息。")
        
def main():
    """主函数"""
    # 设置控制台编码，防止中文乱码
    import sys
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    # 全局时区/seed/日志
    os.environ.setdefault("TZ", "UTC")
    try:
        time.tzset()
    except Exception:
        pass
    seed = int(os.getenv("PAPERBOT_SEED", "42"))
    try:
        import random
        random.seed(seed)
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    logging.basicConfig(
        level=os.getenv("PAPERBOT_LOG_LEVEL", "INFO"),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("=" * 60)
    print("🔐 SecuriPaperBot - 智能论文分析框架")
    print("=" * 60)
    
    # 检查环境
    if not check_python_version():
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description="SecuriPaperBot - 智能论文分析工具")
    
    # 添加子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 下载命令（默认行为）
    parser.add_argument('--conference', choices=['ccs', 'sp', 'ndss', 'usenix'], 
                       help='会议名称')
    parser.add_argument('--year', help='会议年份 (例如: 23 表示2023年)')
    parser.add_argument('--url', help='机构ACM访问URL')
    parser.add_argument('--check', action='store_true', help='检查环境配置')
    parser.add_argument('--demo', action='store_true', help='运行演示模式')
    parser.add_argument('--smart', action='store_true', help='明确启用智能并发下载模式')
    parser.add_argument('--no-smart', action='store_true', help='禁用智能模式，使用传统稳定模式')
    
    # 学者追踪命令
    track_parser = subparsers.add_parser('track', help='学者追踪功能')
    track_parser.add_argument('--config', type=str, 
                              default='config/scholar_subscriptions.yaml',
                              help='订阅配置文件路径')
    track_parser.add_argument('--scholar-id', type=str, 
                              help='仅追踪指定学者 (Semantic Scholar ID)')
    track_parser.add_argument('--force', action='store_true',
                              help='强制重新检测（清除缓存）')
    track_parser.add_argument('--dry-run', action='store_true',
                              help='仅检测新论文，不生成报告')
    track_parser.add_argument('--summary', action='store_true',
                              help='显示追踪状态摘要')
    track_parser.add_argument('--dataset-path', type=str,
                               help='本地数据集路径（覆盖 data_source.dataset_path）')
    track_parser.add_argument('--repro', action='store_true',
                               help='启用可复现性验证（需 Docker）')

    # 运行实验（ExperimentManager）
    exp_parser = subparsers.add_parser('run-exp', help='运行实验配置 (ExperimentManager)')
    exp_parser.add_argument('--config', required=True, help='实验配置文件路径 (YAML)')

    # 渲染报告（从 meta.json + 模板）
    render_parser = subparsers.add_parser('render-report', help='根据 meta.json 渲染报告')
    render_parser.add_argument('--meta', required=False, help='pipeline/实验生成的 meta.json 路径，缺省自动选最新')
    render_parser.add_argument('--template', default=None, help='报告模板名称，默认使用 meta 或 settings 中配置')
    render_parser.add_argument('--output', default=None, help='输出文件路径（可选，默认按论文ID命名写入默认目录）')
    
    parser.add_argument('--mode', choices=['production', 'academic'], default=os.getenv("PAPERBOT_MODE", "production"),
                       help='运行模式 (production/academic)')
    parser.add_argument('--report-template', dest='report_template', default=None,
                       help='覆盖报告模板名称 (如 paper_report.md.j2 / academic_report.md.j2)')
    parser.add_argument('--data-source', dest='data_source', default=None,
                       help='数据源类型覆盖 (api/local/hybrid)，local 时需配合 dataset_path/dataset_name')

    args = parser.parse_args()
    
    if args.check:
        print("🔍 检查环境配置...")
        deps_ok = check_dependencies()
        if deps_ok:
            print("✅ 环境检查通过")
        else:
            print("❌ 环境检查失败")
        return
    
    if args.demo:
        print("🎯 运行演示模式...")
        print("📋 支持的功能:")
        print("  - ACM CCS 论文下载")
        print("  - IEEE S&P 论文下载")
        print("  - NDSS 论文下载")
        print("  - USENIX Security 论文下载")
        print("  - 论文代码链接提取")
        print("  - 代码质量分析")
        print("  - 文档生成")
        print("💡 所有功能完全可用!")
        return
    
    if not check_dependencies():
        print("\n📦 安装依赖:")
        print("pip install requests lxml urllib3 aiohttp beautifulsoup4 pdfplumber")
        sys.exit(1)
    
    # 处理学者追踪命令
    if args.command == 'track':
        run_scholar_tracking(args)
        return

    # 处理实验命令
    if args.command == 'run-exp':
        run_experiment(args)
        return

    # 渲染报告
    if args.command == 'render-report':
        render_report(args)
        return
    
    if args.conference and args.year:
        # 所有会议使用相同的下载逻辑
        if args.no_smart:
            smart_mode = False  # 明确禁用智能模式
        elif args.smart:
            smart_mode = True   # 明确启用智能模式
        else:
            smart_mode = True   # 默认启用智能模式
        
        mode_desc = "智能加速模式" if smart_mode else "传统稳定模式"
        print(f"🚀 开始下载 {args.conference.upper()} {args.year} 论文 ({mode_desc})...")
        simple_paper_download(args.conference, args.year, args.url, smart_mode)
    else:
        print("📝 使用帮助:")
        print("  检查环境: python main.py --check")
        print("  查看演示: python main.py --demo")
        print("  下载论文 (默认智能模式):")
        print("    CCS:    python main.py --conference ccs --year 23")
        print("    S&P:    python main.py --conference sp --year 23")
        print("    NDSS:   python main.py --conference ndss --year 23")
        print("    USENIX: python main.py --conference usenix --year 23")
        print("  关闭智能模式 (传统稳定模式):")
        print("    python main.py --conference ndss --year 23 --no-smart")
        print("  学者追踪:")
        print("    python main.py track --summary")
        print("    python main.py track")
        print("    python main.py track --scholar-id 1741101")
        print("    python main.py track --force")
        print("  查看帮助: python main.py --help")
        print("💡 注意: 所有会议都使用相同的下载方式，支持不同年份")
        print("🤖 智能模式默认启用，提供更快的下载速度和进度显示")


def run_experiment(args):
    """运行 ExperimentManager 实验"""
    from ExperimentManager.runner import ExperimentRunner

    cfg_path = Path(args.config).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = current_dir / cfg_path
    cfg_path = cfg_path.resolve()
    if not cfg_path.exists():
        print(f"❌ 找不到实验配置文件: {cfg_path}")
        return

    print(f"🧪 运行实验: {cfg_path}")
    runner = ExperimentRunner(str(cfg_path))
    runner.run()
    print("✅ 实验完成，结果已写入 output/experiments")


def render_report(args):
    """根据 meta.json 渲染报告（paper/academic 模板兼容）"""
    import json
    from config.settings import settings
    from reports.writer import ReportWriter
    from scholar_tracking.models import PaperMeta
    from scholar_tracking.models.influence import InfluenceResult
    from pathlib import Path
    import glob

    meta_path = None
    if args.meta:
        candidate = Path(args.meta).expanduser()
        if not candidate.is_absolute():
            candidate = current_dir / candidate
        if candidate.exists():
            meta_path = candidate.resolve()
    else:
        # 自动发现 output/experiments 下最新 *_meta.json
        pattern = current_dir / "output" / "experiments" / "*_meta.json"
        metas = glob.glob(str(pattern))
        if metas:
            metas.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
            meta_path = Path(metas[0]).resolve()

    if not meta_path or not meta_path.exists():
        print("❌ 找不到 meta 文件，请使用 --meta 指定或确保 output/experiments 下存在 *_meta.json")
        return

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"❌ 读取 meta 失败: {e}")
        return

    # 兼容不同字段命名
    paper_dict = meta.get("paper") or meta.get("paper_meta") or {}
    influence_dict = meta.get("influence") or meta.get("influence_result") or meta.get("results_summary", [{}])[0].get("influence", {})
    research = meta.get("research") or meta.get("research_result") or {}
    code_analysis = meta.get("code_analysis") or meta.get("code_analysis_result") or {}
    quality = meta.get("quality") or meta.get("quality_result") or {}
    scholar_name = meta.get("scholar_name") or meta.get("scholar") or None

    try:
        paper = PaperMeta.from_dict(paper_dict)
    except Exception as e:
        print(f"❌ 构造 PaperMeta 失败: {e}")
        return

    if not influence_dict:
        # 最小兜底，避免模板崩溃
        influence_dict = {
            "total_score": 0.0,
            "academic_score": 0.0,
            "engineering_score": 0.0,
            "explanation": "No influence data provided.",
            "metrics_breakdown": {},
            "recommendation": "低优先级",
        }
    influence = InfluenceResult.from_dict(influence_dict)

    template_name = (
        args.template
        or meta.get("template")
        or settings.report.get("template", "paper_report.md.j2")
    )

    writer = ReportWriter(template_name=template_name)
    md = writer.render_template(
        paper=paper,
        influence=influence,
        research_result=research,
        code_analysis_result=code_analysis,
        quality_result=quality,
        scholar_name=scholar_name,
    )

    # 追加复现信息（若 meta 提供）
    reproducibility_lines = []
    if meta.get("git_commit") or meta.get("pip_freeze"):
        reproducibility_lines.append("\n\n---\n## 复现信息")
        if meta.get("git_commit"):
            reproducibility_lines.append(f"- Git Commit: `{meta['git_commit']}`")
        if meta.get("pip_freeze"):
            reproducibility_lines.append("- 环境依赖（截断展示）:")
            for line in meta["pip_freeze"][:20]:
                reproducibility_lines.append(f"  - {line}")
    if reproducibility_lines:
        md += "\n".join(reproducibility_lines)

    if args.output:
        out_path = Path(args.output).expanduser()
        if not out_path.is_absolute():
            out_path = current_dir / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        print(f"✅ 报告已写入: {out_path}")
    else:
        path = writer.write_report(md, paper, scholar_name=scholar_name)
        print(f"✅ 报告已写入: {path}")


def run_scholar_tracking(args):
    """运行学者追踪功能"""
    print("=" * 60)
    print("📚 PaperBot 学者追踪系统")
    print("=" * 60)

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = current_dir / config_path
    config_path = config_path.resolve()
    if not config_path.exists():
        print(f"❌ 找不到订阅配置文件: {config_path}")
        return

    async def _run_tracking():
        from scholar_tracking import PaperTrackerAgent, ScholarProfileAgent
        from scholar_tracking.models import PaperMeta
        from core.workflow_coordinator import ScholarWorkflowCoordinator
        from config.settings import settings
        from repro import ReproAgent
        import tempfile, shutil, git

        overrides = {"subscriptions_config_path": str(config_path)}
        mode = getattr(args, "mode", None) or getattr(settings, "mode", "production")
        overrides["mode"] = mode
        if args.report_template:
            overrides["report_template"] = args.report_template
        elif mode == "academic":
            overrides["report_template"] = "academic_report.md.j2"
        if args.data_source:
            overrides["data_source"] = {**settings.data_source, "type": args.data_source}
        if getattr(args, "dataset_path", None):
            ds = overrides.get("data_source", settings.data_source.copy())
            ds["dataset_path"] = args.dataset_path
            overrides["data_source"] = ds
        profile_agent = ScholarProfileAgent(overrides)

        # 显示摘要
        if args.summary:
            print("\n📊 追踪状态摘要:")
            print(profile_agent.summary())
            return

        settings = profile_agent.get_settings()
        reporting_cfg = settings.get("reporting", {})
        min_score = settings.get("min_influence_score", 0)

        tracker_agent = PaperTrackerAgent({**overrides, "api": settings.get("api", {}), "data_source": settings.get("data_source", {})})
        coordinator = ScholarWorkflowCoordinator(
            {
                "output_dir": str(profile_agent.get_output_dir()),
                "report_template": reporting_cfg.get("template", overrides.get("report_template", "paper_report.md.j2")),
                "use_documentation_agent": False, # 禁用 DocumentationAgent 以避免接口不匹配
                "mode": mode,
            }
        )

        # 强制模式
        if args.force and args.scholar_id:
            print(f"\n🔄 强制重新检测学者: {args.scholar_id}")
            profile_agent.clear_scholar_cache(args.scholar_id)
        elif args.force:
            print("\n🔄 清除所有缓存...")
            profile_agent.clear_all_cache()

        # 追踪学者
        if args.scholar_id:
            scholar = profile_agent.get_scholar_by_id(args.scholar_id)
            if not scholar:
                print(f"❌ 未找到学者: {args.scholar_id}")
                return
            result = await tracker_agent.track_scholar(scholar, dry_run=args.dry_run)
            results = [result]
            await tracker_agent.ss_agent.close()
        else:
            print("\n🔍 开始追踪所有订阅学者...")
            results = await tracker_agent.track_all_scholars(dry_run=args.dry_run)

        # 显示结果
        total_new = 0
        for result in results:
            scholar_name = result.get("scholar_name", "Unknown")
            new_count = result.get("new_papers_count", len(result.get("new_papers", [])))
            status = result.get("status", "unknown")

            if status == "success":
                print(f"  ✅ {scholar_name}: 发现 {new_count} 篇新论文")
                total_new += new_count
            elif status == "error":
                print(f"  ❌ {scholar_name}: {result.get('error', '未知错误')}")
            else:
                print(f"  ⚠️  {scholar_name}: {status}")

        print(f"\n📈 总计发现 {total_new} 篇新论文")

        if total_new == 0:
            print("\n✅ 学者追踪完成!")
            return

        persist_reports = not args.dry_run
        if args.dry_run:
            print("\n🧪 Dry-Run 模式：将运行分析但不写入 Markdown 文件。")
        else:
            print("\n📝 生成分析报告...")

        for result in results:
            if result.get("status") != "success":
                continue

            scholar_name = result.get("scholar_name")
            new_papers = result.get("new_papers", [])

            if not new_papers:
                continue

            print(f"\n  处理 {scholar_name} 的 {len(new_papers)} 篇论文...")
            papers = [PaperMeta.from_dict(p) for p in new_papers]
            processed_records = []

            for paper in papers:
                try:
                    report_path, influence, pipeline_data = await coordinator.run_paper_pipeline(
                        paper,
                        scholar_name,
                        persist_report=persist_reports,
                    )

                    pis = f"{influence.total_score:.1f}/100 ({influence.recommendation.value})"
                    if report_path:
                        print(f"    📄 {paper.title[:40]}... -> {report_path.name} | PIS {pis}")
                    else:
                        print(f"    📄 {paper.title[:40]}... -> (未持久化) | PIS {pis}")

                    # 可复现性验证（需 repo）
                    repro_result = {}
                    if args.repro and (paper.github_url or paper.has_code):
                        tmp_dir = Path(tempfile.mkdtemp(prefix="paperbot-repro-"))
                        try:
                            repo_url = paper.github_url
                            if repo_url:
                                print(f"    🔁 Repro: cloning {repo_url}")
                                git.Repo.clone_from(repo_url, tmp_dir)
                                repro_agent = ReproAgent({"repro": settings.repro})
                                repro_result = await repro_agent.run(tmp_dir)
                            else:
                                repro_result = {"status": "skipped", "reason": "no_repo"}
                        except Exception as e:
                            repro_result = {"status": "error", "error": str(e)}
                            print(f"    ⚠️ Repro 失败: {e}")
                        finally:
                            try:
                                shutil.rmtree(tmp_dir)
                            except Exception:
                                pass

                        # 重新渲染报告，写回
                        try:
                            md = coordinator.report_writer.render_template(
                                paper=paper,
                                influence=influence,
                                research_result=pipeline_data.get("stages", {}).get("research", {}).get("result", {}),
                                code_analysis_result=pipeline_data.get("stages", {}).get("code_analysis", {}).get("result", {}),
                                quality_result=pipeline_data.get("stages", {}).get("quality", {}).get("result", {}),
                                scholar_name=scholar_name,
                                repro_result=repro_result,
                                meta=None,
                            )
                            path = coordinator.report_writer.write_report(md, paper, scholar_name)
                            pipeline_data["report_path"] = str(path)
                        except Exception as e:
                            print(f"    ⚠️ 重渲染报告失败: {e}")

                    processed_records.append(
                        {
                            "paper_id": paper.paper_id,
                            "title": paper.title,
                            "report_path": str(report_path) if report_path else None,
                            "pis": round(influence.total_score, 2),
                            "recommendation": influence.recommendation.value,
                            "status": pipeline_data.get("status", "success"),
                        }
                    )
                except Exception as e:
                    print(f"    ❌ 处理失败: {paper.title[:40]}... - {e}")
                    processed_records.append(
                        {
                            "paper_id": paper.paper_id,
                            "title": paper.title,
                            "status": f"failed: {e}",
                        }
                    )

            scholar_id = result.get("scholar_id")
            if (
                processed_records
                and scholar_id
                and reporting_cfg.get("persist_history", True)
            ):
                profile_agent.record_processed_papers(
                    scholar_id,
                    processed_records,
                )

        print("\n✅ 学者追踪完成!")

    try:
        asyncio.run(_run_tracking())
    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 追踪过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()