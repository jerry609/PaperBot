# config/settings.py
"""
Unified Pydantic V2 configuration system.

Replaces the legacy dataclass-based settings with type-safe Pydantic BaseModel
that supports YAML loading and environment variable overrides.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

from config.models import AppConfig, LLMConfig, PipelineConfig, ReproConfig, SemanticScholarConfig


class DatabaseConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    url: str = ""
    timeout: int = 30
    pool_size: int = 10


class DownloadConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    path: str = "./papers"
    max_retries: int = 3
    retry_delay: int = 5
    max_concurrent_downloads: int = 5
    cleanup_days: int = 30
    user_agent: str = "SecuriPaperBot/1.0"
    timeout: int = 30


class AnalysisConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    depth: str = "detailed"
    parallel_processing: bool = True
    cache_results: bool = True
    quality_threshold: float = 0.8
    ignore_patterns: List[str] = Field(
        default_factory=lambda: ["*/test/*", "*/docs/*", "*/examples/*"]
    )
    file_types: List[str] = Field(default_factory=lambda: [".py", ".js", ".java", ".cpp", ".go"])


class SecurityConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    verify_ssl: bool = True
    rate_limit: int = 60
    timeout: int = 30
    allowed_domains: List[str] = Field(
        default_factory=lambda: [
            "dl.acm.org",
            "ieeexplore.ieee.org",
            "www.ndss-symposium.org",
            "www.usenix.org",
        ]
    )


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    format: str = "markdown"
    path: str = "./output"
    compress: bool = False
    structure: List[str] = Field(
        default_factory=lambda: [
            "summary",
            "analysis",
            "quality_report",
            "security_report",
            "recommendations",
        ]
    )


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    level: str = "INFO"
    file: str = "logs/securipaperbot.log"
    max_size: int = 10485760  # 10MB
    backup_count: int = 5
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class ConferenceConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = ""
    base_url: str = ""
    parser: str = ""


class APIConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    github_token: Optional[str] = None
    openai_api_key: Optional[str] = None
    github_base_url: str = "https://api.github.com"
    openai_model: str = "gpt-4"
    openai_temperature: float = 0.3
    openai_max_tokens: int = 2000


class ReportEngineConf(BaseModel):
    model_config = ConfigDict(extra="ignore")

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


class CollabHostConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    base_url: Optional[str] = None
    temperature: float = 0.3
    top_p: float = 0.9


class Settings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    download: DownloadConfig = Field(default_factory=DownloadConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    report_engine: ReportEngineConf = Field(default_factory=ReportEngineConf)
    collab: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": False,
            "host": CollabHostConfig().model_dump(),
        }
    )
    mode: str = "production"
    data_source: Dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "api",
            "dataset_name": None,
            "dataset_path": None,
        }
    )
    report: Dict[str, Any] = Field(default_factory=lambda: {"template": "paper_report.md.j2"})
    repro: Dict[str, Any] = Field(
        default_factory=lambda: {
            "docker_image": "python:3.10-slim",
            "cpu_shares": 1,
            "mem_limit": "1g",
            "timeout_sec": 300,
            "network": False,
            "executor": "auto",
            "e2b_api_key": None,
            "e2b_template": "Python3",
            "e2b_timeout": 300,
        }
    )
    offline: bool = False
    conferences: Dict[str, ConferenceConfig] = Field(default_factory=dict)

    @classmethod
    def load_from_file(cls, config_path: Optional[str] = None) -> Settings:
        if config_path is None:
            config_path_obj = Path(__file__).parent / "config.yaml"
        else:
            config_path_obj = Path(config_path)

        if not config_path_obj.exists():
            return cls()

        with open(config_path_obj, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        return cls.from_dict(config_data or {})

    @classmethod
    def from_dict(cls, config_data: Dict[str, Any]) -> Settings:
        defaults = cls()
        mapped: Dict[str, Any] = {}
        for key in (
            "download",
            "analysis",
            "security",
            "output",
            "logging",
            "report_engine",
            "mode",
            "offline",
        ):
            if key in config_data:
                mapped[key] = config_data[key]

        if "apis" in config_data:
            mapped["api"] = config_data["apis"]
        elif "api" in config_data:
            mapped["api"] = config_data["api"]

        if "collab" in config_data:
            collab_cfg = config_data["collab"]
            host_cfg = collab_cfg.get("host", {})
            collab_data = dict(defaults.collab)
            collab_data.update({k: v for k, v in collab_cfg.items() if k != "host"})
            host_data = dict(collab_data.get("host") or CollabHostConfig().model_dump())
            host_data.update(host_cfg)
            collab_data["host"] = host_data
            mapped["collab"] = collab_data

        if "data_source" in config_data:
            mapped["data_source"] = {**defaults.data_source, **config_data["data_source"]}

        if "report" in config_data:
            mapped["report"] = {**defaults.report, **config_data["report"]}

        if "repro" in config_data:
            mapped["repro"] = {**defaults.repro, **config_data["repro"]}

        if "conferences" in config_data:
            mapped["conferences"] = config_data["conferences"]

        return cls(**mapped)

    def load_environment_variables(self) -> None:
        """Apply environment variable overrides."""
        self.api.github_token = os.getenv("GITHUB_TOKEN", self.api.github_token)
        self.api.openai_api_key = os.getenv("OPENAI_API_KEY", self.api.openai_api_key)

        env_mode = os.getenv("PAPERBOT_MODE")
        if env_mode:
            self.mode = env_mode
        env_template = os.getenv("PAPERBOT_REPORT_TEMPLATE")
        if env_template:
            self.report["template"] = env_template
        env_ds_type = os.getenv("PAPERBOT_DATA_SOURCE")
        if env_ds_type:
            self.data_source["type"] = env_ds_type
        env_ds_path = os.getenv("PAPERBOT_DATASET_PATH")
        if env_ds_path:
            self.data_source["dataset_path"] = env_ds_path
        env_repro_image = os.getenv("PAPERBOT_REPRO_IMAGE")
        if env_repro_image:
            self.repro["docker_image"] = env_repro_image

        # E2B configuration
        env_e2b_key = os.getenv("E2B_API_KEY")
        if env_e2b_key:
            self.repro["e2b_api_key"] = env_e2b_key
        env_e2b_template = os.getenv("PAPERBOT_E2B_TEMPLATE")
        if env_e2b_template:
            self.repro["e2b_template"] = env_e2b_template
        env_e2b_timeout = os.getenv("PAPERBOT_E2B_TIMEOUT")
        if env_e2b_timeout:
            try:
                self.repro["e2b_timeout"] = int(env_e2b_timeout)
            except ValueError:
                pass
        env_executor = os.getenv("PAPERBOT_EXECUTOR")
        if env_executor and env_executor in ("docker", "e2b", "auto"):
            self.repro["executor"] = env_executor

        env_offline = os.getenv("PAPERBOT_OFFLINE")
        if env_offline is not None:
            self.offline = env_offline.lower() in ("1", "true", "yes", "on")

        # Report Engine
        re_enabled = os.getenv("PAPERBOT_RE_ENABLED")
        re_api = os.getenv("PAPERBOT_RE_API_KEY")
        re_model = os.getenv("PAPERBOT_RE_MODEL")
        re_base = os.getenv("PAPERBOT_RE_BASE_URL")
        re_out = os.getenv("PAPERBOT_RE_OUTPUT_DIR")
        re_tpl = os.getenv("PAPERBOT_RE_TEMPLATE_DIR")
        re_pdf = os.getenv("PAPERBOT_RE_PDF_ENABLED")
        re_max = os.getenv("PAPERBOT_RE_MAX_WORDS")
        re_scenario = os.getenv("PAPERBOT_RE_SCENARIO")
        re_tiers = os.getenv("PAPERBOT_RE_MODEL_TIERS")
        if re_enabled is not None:
            self.report_engine.enabled = re_enabled.lower() in ("1", "true", "yes", "on")
        if re_api:
            self.report_engine.api_key = re_api
        if re_model:
            self.report_engine.model = re_model
        if re_base:
            self.report_engine.base_url = re_base
        if re_out:
            self.report_engine.output_dir = re_out
        if re_tpl:
            self.report_engine.template_dir = re_tpl
        if re_pdf is not None:
            self.report_engine.pdf_enabled = re_pdf.lower() in ("1", "true", "yes", "on")
        if re_max:
            try:
                self.report_engine.max_words = int(re_max)
            except ValueError:
                pass
        if re_scenario:
            self.report_engine.scenario = re_scenario
        if re_tiers:
            tiers = {}
            for pair in re_tiers.split(","):
                if ":" in pair:
                    k, v = pair.split(":", 1)
                    tiers[k.strip()] = v.strip()
            self.report_engine.model_tiers = tiers

        # Collab host LLM
        host_api = os.getenv("PAPERBOT_HOST_API_KEY")
        host_model = os.getenv("PAPERBOT_HOST_MODEL")
        host_base = os.getenv("PAPERBOT_HOST_BASE_URL")
        host_enabled = os.getenv("PAPERBOT_HOST_ENABLED")
        if host_api:
            self.collab["host"]["api_key"] = host_api
        if host_model:
            self.collab["host"]["model"] = host_model
        if host_base:
            self.collab["host"]["base_url"] = host_base
        if host_enabled is not None:
            self.collab["host"]["enabled"] = host_enabled.lower() in ("1", "true", "yes", "on")

        acm_url = os.getenv("ACM_LIBRARY_URL")
        if acm_url and "ccs" in self.conferences:
            self.conferences["ccs"].base_url = acm_url

    def to_dict(self) -> Dict[str, Any]:
        return {
            "download": self.download.model_dump(),
            "analysis": self.analysis.model_dump(),
            "security": self.security.model_dump(),
            "output": self.output.model_dump(),
            "logging": self.logging.model_dump(),
            "api": self.api.model_dump(),
            "report_engine": self.report_engine.model_dump(),
            "collab": self.collab,
            "conferences": {name: conf.model_dump() for name, conf in self.conferences.items()},
        }


def create_settings(config_path: Optional[str] = None) -> Settings:
    settings = Settings.load_from_file(config_path)
    settings.load_environment_variables()
    return settings


def to_app_config(settings: Settings) -> AppConfig:
    """Map Settings to the newer AppConfig (backwards compatible)."""
    llm_cfg = LLMConfig(
        provider="anthropic",
        model=settings.api.openai_model,
        api_key=settings.api.openai_api_key,
        temperature=settings.api.openai_temperature,
        max_tokens=settings.api.openai_max_tokens,
    )
    s2_cfg = SemanticScholarConfig(api_key=None)
    repro_cfg = ReproConfig(
        docker_image=settings.repro.get("docker_image", "python:3.10-slim"),
        timeout_sec=int(settings.repro.get("timeout_sec", 300)),
        cpu_shares=int(settings.repro.get("cpu_shares", 1)),
        mem_limit=settings.repro.get("mem_limit", "1g"),
        network=bool(settings.repro.get("network", False)),
        executor=settings.repro.get("executor", "auto"),
        e2b_api_key=settings.repro.get("e2b_api_key"),
        e2b_template=settings.repro.get("e2b_template", "Python3"),
        e2b_timeout=int(settings.repro.get("e2b_timeout", 300)),
    )
    pipeline_cfg = PipelineConfig(
        mode=settings.mode,
        report_template=settings.report.get("template", "paper_report.md.j2"),
        enable_code_analysis=True,
        enable_repro=settings.repro.get("enable_repro", False),
    )
    return AppConfig(
        llm=llm_cfg,
        semantic_scholar=s2_cfg,
        repro=repro_cfg,
        pipeline=pipeline_cfg,
        raw=settings.to_dict(),
    )


# Global settings instance
settings = create_settings()
