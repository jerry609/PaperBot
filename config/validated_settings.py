"""
基于 pydantic 的配置校验与对象化加载。

提供 SettingsModel（可忽略多余字段），并转换为现有 dataclass Settings。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field, Extra
import yaml

from .settings import (
    DatabaseConfig,
    DownloadConfig,
    AnalysisConfig,
    SecurityConfig,
    OutputConfig,
    LoggingConfig,
    APIConfig,
    ConferenceConfig,
    ReportEngineConf,
    CollabHostConfig,
    Settings,
)


class DatabaseConfigModel(BaseModel):
    url: str = ""
    timeout: int = 30
    pool_size: int = 10

    class Config:
        extra = Extra.ignore


class DownloadConfigModel(BaseModel):
    path: str = "./papers"
    max_retries: int = 3
    retry_delay: int = 5
    max_concurrent_downloads: int = 5
    cleanup_days: int = 30
    user_agent: str = "SecuriPaperBot/1.0"
    timeout: int = 30

    class Config:
        extra = Extra.ignore


class AnalysisConfigModel(BaseModel):
    depth: str = "detailed"
    parallel_processing: bool = True
    cache_results: bool = True
    quality_threshold: float = 0.8
    ignore_patterns: list[str] = Field(default_factory=lambda: ["*/test/*", "*/docs/*", "*/examples/*"])
    file_types: list[str] = Field(default_factory=lambda: [".py", ".js", ".java", ".cpp", ".go"])

    class Config:
        extra = Extra.ignore


class SecurityConfigModel(BaseModel):
    verify_ssl: bool = True
    rate_limit: int = 60
    timeout: int = 30
    allowed_domains: list[str] = Field(
        default_factory=lambda: [
            "dl.acm.org",
            "ieeexplore.ieee.org",
            "www.ndss-symposium.org",
            "www.usenix.org",
        ]
    )

    class Config:
        extra = Extra.ignore


class OutputConfigModel(BaseModel):
    format: str = "markdown"
    path: str = "./output"
    compress: bool = False
    structure: list[str] = Field(
        default_factory=lambda: ["summary", "analysis", "quality_report", "security_report", "recommendations"]
    )

    class Config:
        extra = Extra.ignore


class LoggingConfigModel(BaseModel):
    level: str = "INFO"
    file: str = "logs/securipaperbot.log"
    max_size: int = 10485760
    backup_count: int = 5
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    class Config:
        extra = Extra.ignore


class ConferenceConfigModel(BaseModel):
    name: str = ""
    base_url: str = ""
    parser: str = ""

    class Config:
        extra = Extra.ignore


class APIConfigModel(BaseModel):
    github_token: Optional[str] = None
    openai_api_key: Optional[str] = None
    github_base_url: str = "https://api.github.com"
    openai_model: str = "gpt-4"
    openai_temperature: float = 0.3
    openai_max_tokens: int = 2000

    class Config:
        extra = Extra.ignore


class ReportEngineConfModel(BaseModel):
    enabled: bool = False
    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    base_url: Optional[str] = None
    output_dir: str = "output/reports"
    template_dir: str = "core/report_engine/templates"
    pdf_enabled: bool = True
    max_words: int = 6000
    scenario: str = "default"
    model_tiers: Dict[str, str] = Field(default_factory=dict)

    class Config:
        extra = Extra.ignore


class CollabHostConfigModel(BaseModel):
    enabled: bool = False
    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    base_url: Optional[str] = None
    temperature: float = 0.3
    top_p: float = 0.9

    class Config:
        extra = Extra.ignore


class SettingsModel(BaseModel):
    download: DownloadConfigModel = DownloadConfigModel()
    analysis: AnalysisConfigModel = AnalysisConfigModel()
    security: SecurityConfigModel = SecurityConfigModel()
    output: OutputConfigModel = OutputConfigModel()
    logging: LoggingConfigModel = LoggingConfigModel()
    api: APIConfigModel = APIConfigModel()
    mode: str = "production"
    data_source: Dict[str, Any] = Field(default_factory=lambda: {"type": "api", "dataset_name": None, "dataset_path": None})
    report: Dict[str, Any] = Field(default_factory=lambda: {"template": "paper_report.md.j2"})
    repro: Dict[str, Any] = Field(default_factory=lambda: {"docker_image": "python:3.10-slim"})
    offline: bool = False
    conferences: Dict[str, ConferenceConfigModel] = Field(default_factory=dict)
    report_engine: ReportEngineConfModel = ReportEngineConfModel()
    collab: Dict[str, Any] = Field(default_factory=lambda: {"enabled": False, "host": CollabHostConfigModel().dict()})

    class Config:
        extra = Extra.ignore

    def to_dataclass(self) -> Settings:
        s = Settings()
        s.download = DownloadConfig(**self.download.dict())
        s.analysis = AnalysisConfig(**self.analysis.dict())
        s.security = SecurityConfig(**self.security.dict())
        s.output = OutputConfig(**self.output.dict())
        s.logging = LoggingConfig(**self.logging.dict())
        s.api = APIConfig(**self.api.dict())
        s.mode = self.mode
        s.data_source = dict(self.data_source)
        s.report = dict(self.report)
        s.repro = dict(self.repro)
        s.offline = self.offline
        s.conferences = {k: ConferenceConfig(**v.dict()) for k, v in self.conferences.items()}
        s.report_engine = ReportEngineConf(**self.report_engine.dict())
        s.collab = self.collab
        return s


def load_validated_settings(config_path: Optional[str] = None) -> Settings:
    """使用 pydantic 校验后返回 Settings dataclass。"""
    cfg_file = Path(config_path) if config_path else Path(__file__).parent / "config.yaml"
    data = {}
    if cfg_file.exists():
        data = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
    model = SettingsModel(**data)
    return model.to_dataclass()

