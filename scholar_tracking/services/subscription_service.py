# scholar_tracking/services/subscription_service.py
"""
订阅配置管理服务
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml

from ..models import Scholar

logger = logging.getLogger(__name__)


class SubscriptionService:
    """学者订阅配置管理服务"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化订阅服务
        
        Args:
            config_path: 配置文件路径，默认为 config/scholar_subscriptions.yaml
        """
        if config_path is None:
            # 默认配置文件路径
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "scholar_subscriptions.yaml"
        
        self.config_path = Path(config_path)
        self._config: Optional[Dict[str, Any]] = None
        self._scholars: Optional[List[Scholar]] = None
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if self._config is not None:
            return self._config
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Subscription config not found: {self.config_path}")
        
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f)
            
            self._validate_config()
            logger.info(f"Loaded subscription config from {self.config_path}")
            return self._config
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")
    
    def _validate_config(self):
        """校验配置文件结构"""
        if not self._config:
            raise ValueError("Config is empty")
        
        subscriptions = self._config.get("subscriptions", {})
        if not subscriptions:
            raise ValueError("Missing 'subscriptions' section in config")
        
        scholars = subscriptions.get("scholars", [])
        if not scholars:
            raise ValueError("No scholars configured for tracking")
        
        # 校验每个学者的必填字段
        for i, scholar in enumerate(scholars):
            if not scholar.get("name"):
                raise ValueError(f"Scholar at index {i} missing 'name'")
            if not scholar.get("semantic_scholar_id"):
                raise ValueError(f"Scholar '{scholar.get('name')}' missing 'semantic_scholar_id'")
        
        # 校验 settings
        settings = subscriptions.get("settings", {})
        required_settings = [
            "check_interval",
            "papers_per_scholar",
            "min_influence_score",
            "output_dir",
            "cache_dir",
            "api",
        ]
        missing = [key for key in required_settings if not settings.get(key)]
        if missing:
            raise ValueError(f"Missing subscription settings: {', '.join(missing)}")

        api_config = settings.get("api", {})
        semantic_api = api_config.get("semantic_scholar")
        if not semantic_api or not semantic_api.get("base_url"):
            raise ValueError("Semantic Scholar API config requires 'base_url'")
        if semantic_api.get("timeout") is None:
            raise ValueError("Semantic Scholar API config requires 'timeout'")
    
    def get_scholars(self) -> List[Scholar]:
        """获取所有订阅的学者"""
        if self._scholars is not None:
            return self._scholars
        
        config = self.load_config()
        scholars_config = config.get("subscriptions", {}).get("scholars", [])
        
        self._scholars = [
            Scholar.from_config(s) for s in scholars_config
        ]
        
        logger.info(f"Loaded {len(self._scholars)} scholars from config")
        return self._scholars
    
    def get_settings(self) -> Dict[str, Any]:
        """获取全局设置"""
        config = self.load_config()
        settings = config.get("subscriptions", {}).get("settings", {})
        
        # 返回带默认值的设置
        return {
            "check_interval": settings.get("check_interval", "weekly"),
            "papers_per_scholar": settings.get("papers_per_scholar", 20),
            "min_influence_score": settings.get("min_influence_score", 50),
            "output_dir": settings.get("output_dir", "output/reports"),
            "cache_dir": settings.get("cache_dir", "cache/scholar_papers"),
            "api": settings.get("api", {}),
            "logging": settings.get("logging", {}),
            "reporting": settings.get("reporting", {}),
        }

    def get_reporting_config(self) -> Dict[str, Any]:
        """获取报告输出配置"""
        settings = self.get_settings()
        return settings.get("reporting", {})
    
    def get_api_config(self, api_name: str) -> Dict[str, Any]:
        """获取特定 API 的配置"""
        settings = self.get_settings()
        return settings.get("api", {}).get(api_name, {})
    
    def get_output_dir(self) -> Path:
        """获取报告输出目录"""
        settings = self.get_settings()
        output_dir = settings.get("output_dir", "output/reports")
        
        # 转为绝对路径
        if not os.path.isabs(output_dir):
            project_root = Path(__file__).parent.parent.parent
            output_dir = project_root / output_dir
        
        return Path(output_dir)
    
    def get_cache_dir(self) -> Path:
        """获取缓存目录"""
        settings = self.get_settings()
        cache_dir = settings.get("cache_dir", "cache/scholar_papers")
        
        # 转为绝对路径
        if not os.path.isabs(cache_dir):
            project_root = Path(__file__).parent.parent.parent
            cache_dir = project_root / cache_dir
        
        return Path(cache_dir)
