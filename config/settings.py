# securipaperbot/config/settings.py

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


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
class Settings:
    """主配置类"""
    download: DownloadConfig = field(default_factory=DownloadConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    api: APIConfig = field(default_factory=APIConfig)
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


# 全局设置实例
settings = create_settings()