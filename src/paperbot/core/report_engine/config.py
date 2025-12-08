"""
Report Engine 配置与数据模型（精简版）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict


@dataclass
class ReportEngineConfig:
    """报告引擎配置。"""
    enabled: bool = False
    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    base_url: Optional[str] = None
    temperature: float = 0.2
    top_p: float = 0.9
    output_dir: Path = field(default_factory=lambda: Path("output/reports"))
    template_dir: Path = field(default_factory=lambda: Path("core/report_engine/templates"))
    pdf_enabled: bool = True
    max_words: int = 6000
    scenario: str = "default"  # default / review / verification / compare / scholar
    model_tiers: Optional[Dict[str, str]] = None  # task_type -> model name


@dataclass
class ReportResult:
    """报告生成结果。"""
    html_path: Optional[Path] = None
    pdf_path: Optional[Path] = None
    ir_path: Optional[Path] = None
    summary_path: Optional[Path] = None
    html_content: Optional[str] = None
    title: str = ""

