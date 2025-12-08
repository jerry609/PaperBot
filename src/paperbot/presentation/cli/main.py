"""
CLI 入口点

提供命令行接口，委托给 main.py 中的逻辑。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Optional


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


if __name__ == "__main__":
    sys.exit(run_cli())

