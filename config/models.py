"""
Pydantic 配置模型，提供类型安全的配置与兼容加载。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal, Any, Dict

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: Literal["anthropic", "openai", "custom"] = "anthropic"
    model: str = "claude-3-5-sonnet-20241022"
    api_key: Optional[str] = None
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int = Field(default=4096, gt=0)


class SemanticScholarConfig(BaseModel):
    api_key: Optional[str] = None
    rate_limit_per_min: int = 100


class ReproConfig(BaseModel):
    docker_image: str = "python:3.10-slim"
    timeout_sec: int = 300
    cpu_shares: int = 1024
    mem_limit: str = "512m"
    network: bool = False
    # Executor selection: "docker", "e2b", or "auto"
    executor: Literal["docker", "e2b", "auto"] = "auto"
    # E2B configuration
    e2b_api_key: Optional[str] = None
    e2b_template: str = "Python3"
    e2b_timeout: int = 300


class PipelineConfig(BaseModel):
    mode: Literal["production", "academic"] = "production"
    report_template: str = "paper_report.md.j2"
    enable_code_analysis: bool = True
    enable_repro: bool = False


class AppConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    semantic_scholar: SemanticScholarConfig = SemanticScholarConfig()
    repro: ReproConfig = ReproConfig()
    pipeline: PipelineConfig = PipelineConfig()
    raw: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        file_path = Path(path).expanduser()
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        data = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
        # 保留原始数据便于兼容
        return cls(**data, raw=data)

    @classmethod
    def from_settings(cls, settings: Dict[str, Any]) -> "AppConfig":
        """兼容已有 settings dict。"""
        return cls(**settings, raw=settings)

