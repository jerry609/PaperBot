# securipaperbot/config/settings.py

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from config.models import AppConfig, LLMConfig, SemanticScholarConfig, ReproConfig, PipelineConfig


@dataclass
class DatabaseConfig:
    """数据库配置"""
    url: str = ""
    timeout: int = 30
    pool_size: int = 10


@dataclass
class DownloadConfig:
    """下载配置"""
    path: str = "./papers"
    max_retries: int = 3
    retry_delay: int = 5
    max_concurrent_downloads: int = 5
    cleanup_days: int = 30
    user_agent: str = "SecuriPaperBot/1.0"
    timeout: int = 30


@dataclass
class AnalysisConfig:
    """分析配置"""
    depth: str = "detailed"
    parallel_processing: bool = True
    cache_results: bool = True
    quality_threshold: float = 0.8
    ignore_patterns: list = field(default_factory=lambda: [
        "*/test/*", "*/docs/*", "*/examples/*"
    ])
    file_types: list = field(default_factory=lambda: [
        ".py", ".js", ".java", ".cpp", ".go"
    ])


@dataclass
class SecurityConfig:
    """安全配置"""
    verify_ssl: bool = True
    rate_limit: int = 60
    timeout: int = 30
    allowed_domains: list = field(default_factory=lambda: [
        "dl.acm.org",
        "ieeexplore.ieee.org", 
        "www.ndss-symposium.org",
        "www.usenix.org"
    ])


@dataclass
class OutputConfig:
    """输出配置"""
    format: str = "markdown"
    path: str = "./output"
    compress: bool = False
    structure: list = field(default_factory=lambda: [
        "summary", "analysis", "quality_report", "security_report", "recommendations"
    ])


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    file: str = "logs/securipaperbot.log"
    max_size: int = 10485760  # 10MB
    backup_count: int = 5
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class ConferenceConfig:
    """会议配置"""
    name: str = ""
    base_url: str = ""
    parser: str = ""


@dataclass
class APIConfig:
    """API配置"""
    github_token: Optional[str] = None
    openai_api_key: Optional[str] = None
    github_base_url: str = "https://api.github.com"
    openai_model: str = "gpt-4"
    openai_temperature: float = 0.3
    openai_max_tokens: int = 2000


@dataclass
class ReportEngineConf:
    """报告引擎配置"""
    enabled: bool = False
    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    base_url: Optional[str] = None
    output_dir: str = "output/reports"
    template_dir: str = "core/report_engine/templates"
    pdf_enabled: bool = True
    max_words: int = 6000
    scenario: str = "default"
    model_tiers: Dict[str, str] = field(default_factory=dict)


@dataclass
class CollabHostConfig:
    """主持人 LLM 配置"""
    enabled: bool = False
    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    base_url: Optional[str] = None
    temperature: float = 0.3
    top_p: float = 0.9


@dataclass
class Settings:
    """主配置类"""
    download: DownloadConfig = field(default_factory=DownloadConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    report_engine: ReportEngineConf = field(default_factory=ReportEngineConf)
    collab: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "host": CollabHostConfig().__dict__,
    })
    # 模式与数据源
    mode: str = "production"  # production / academic
    data_source: Dict[str, Any] = field(default_factory=lambda: {
        "type": "api",          # api / local / hybrid
        "dataset_name": None,   # 对应 datasets/processed/{name}.csv
        "dataset_path": None,   # 显式路径优先
    })
    report: Dict[str, Any] = field(default_factory=lambda: {
        "template": "paper_report.md.j2",  # academic_report.md.j2 for academic
    })
    repro: Dict[str, Any] = field(default_factory=lambda: {
        "docker_image": "python:3.10-slim",
        "cpu_shares": 1,
        "mem_limit": "1g",
        "timeout_sec": 300,
        "network": False,  # 禁用网络，True 表示允许
    })
    offline: bool = False  # 离线/私有模式开关
    conferences: Dict[str, ConferenceConfig] = field(default_factory=dict)

    @classmethod
    def load_from_file(cls, config_path: Optional[str] = None) -> 'Settings':
        """从文件加载配置"""
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        config_path = Path(config_path)
        
        if not config_path.exists():
            # 返回默认配置
            return cls()
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        return cls.from_dict(config_data)
    
    @classmethod
    def from_dict(cls, config_data: Dict[str, Any]) -> 'Settings':
        """从字典创建配置"""
        settings = cls()
        
        # 更新各个配置部分
        if 'download' in config_data:
            settings.download = DownloadConfig(**config_data['download'])
        
        if 'analysis' in config_data:
            settings.analysis = AnalysisConfig(**config_data['analysis'])
        
        if 'security' in config_data:
            settings.security = SecurityConfig(**config_data['security'])
        
        if 'output' in config_data:
            settings.output = OutputConfig(**config_data['output'])
        
        if 'logging' in config_data:
            settings.logging = LoggingConfig(**config_data['logging'])
        
        if 'apis' in config_data:
            settings.api = APIConfig(**config_data['apis'])

        if 'report_engine' in config_data:
            settings.report_engine = ReportEngineConf(**config_data['report_engine'])
        
        if 'collab' in config_data:
            collab_cfg = config_data['collab']
            host_cfg = collab_cfg.get("host", {})
            settings.collab = {
                **settings.collab,
                **{k: v for k, v in collab_cfg.items() if k != "host"},
            }
            settings.collab["host"] = {**settings.collab.get("host", {}), **host_cfg}
        
        if 'mode' in config_data:
            settings.mode = config_data['mode']
        
        if 'data_source' in config_data:
            settings.data_source = {**settings.data_source, **config_data['data_source']}
        
        if 'report' in config_data:
            settings.report = {**settings.report, **config_data['report']}
        
        if 'repro' in config_data:
            settings.repro = {**settings.repro, **config_data['repro']}
        
        if 'offline' in config_data:
            settings.offline = bool(config_data['offline'])
        
        if 'conferences' in config_data:
            settings.conferences = {
                name: ConferenceConfig(**conf_data)
                for name, conf_data in config_data['conferences'].items()
            }
        
        return settings
    
    def load_environment_variables(self):
        """加载环境变量"""
        # API配置
        self.api.github_token = os.getenv('GITHUB_TOKEN', self.api.github_token)
        self.api.openai_api_key = os.getenv('OPENAI_API_KEY', self.api.openai_api_key)
        env_mode = os.getenv('PAPERBOT_MODE')
        if env_mode:
            self.mode = env_mode
        env_template = os.getenv('PAPERBOT_REPORT_TEMPLATE')
        if env_template:
            self.report['template'] = env_template
        env_ds_type = os.getenv('PAPERBOT_DATA_SOURCE')
        if env_ds_type:
            self.data_source['type'] = env_ds_type
        env_ds_path = os.getenv('PAPERBOT_DATASET_PATH')
        if env_ds_path:
            self.data_source['dataset_path'] = env_ds_path
        env_repro_image = os.getenv('PAPERBOT_REPRO_IMAGE')
        if env_repro_image:
            self.repro['docker_image'] = env_repro_image
        env_offline = os.getenv('PAPERBOT_OFFLINE')
        if env_offline is not None:
            self.offline = env_offline.lower() in ("1", "true", "yes", "on")

        # Report Engine
        re_enabled = os.getenv('PAPERBOT_RE_ENABLED')
        re_api = os.getenv('PAPERBOT_RE_API_KEY')
        re_model = os.getenv('PAPERBOT_RE_MODEL')
        re_base = os.getenv('PAPERBOT_RE_BASE_URL')
        re_out = os.getenv('PAPERBOT_RE_OUTPUT_DIR')
        re_tpl = os.getenv('PAPERBOT_RE_TEMPLATE_DIR')
        re_pdf = os.getenv('PAPERBOT_RE_PDF_ENABLED')
        re_max = os.getenv('PAPERBOT_RE_MAX_WORDS')
        re_scenario = os.getenv('PAPERBOT_RE_SCENARIO')
        re_tiers = os.getenv('PAPERBOT_RE_MODEL_TIERS')  # 格式: task:model,task:model
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

        # 主持人 LLM
        host_api = os.getenv('PAPERBOT_HOST_API_KEY')
        host_model = os.getenv('PAPERBOT_HOST_MODEL')
        host_base = os.getenv('PAPERBOT_HOST_BASE_URL')
        host_enabled = os.getenv('PAPERBOT_HOST_ENABLED')
        if host_api:
            self.collab["host"]["api_key"] = host_api
        if host_model:
            self.collab["host"]["model"] = host_model
        if host_base:
            self.collab["host"]["base_url"] = host_base
        if host_enabled is not None:
            self.collab["host"]["enabled"] = host_enabled.lower() in ("1", "true", "yes", "on")
        
        # 其他环境变量
        acm_url = os.getenv('ACM_LIBRARY_URL')
        if acm_url and 'ccs' in self.conferences:
            self.conferences['ccs'].base_url = acm_url
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'download': self.download.__dict__,
            'analysis': self.analysis.__dict__,
            'security': self.security.__dict__,
            'output': self.output.__dict__,
            'logging': self.logging.__dict__,
            'api': self.api.__dict__,
            'report_engine': self.report_engine.__dict__,
            'collab': self.collab,
            'conferences': {
                name: conf.__dict__ for name, conf in self.conferences.items()
            }
        }


# 创建全局设置实例
def create_settings(config_path: Optional[str] = None) -> Settings:
    """创建设置实例"""
    settings = Settings.load_from_file(config_path)
    settings.load_environment_variables()
    return settings


def to_app_config(settings: Settings) -> AppConfig:
    """将旧版 Settings 映射为新的 AppConfig（向后兼容）。"""
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


# 全局设置实例
settings = create_settings()