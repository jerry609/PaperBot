# src/paperbot/agents/huggingface/agent.py
"""
HuggingFace Agent

从 HuggingFace Hub 获取论文关联模型的元数据。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from paperbot.agents.base import BaseAgent

logger = logging.getLogger(__name__)


@dataclass
class HuggingFaceResult:
    """HuggingFace 分析结果"""
    paper_title: str
    arxiv_id: Optional[str] = None
    models: List[Dict[str, Any]] = field(default_factory=list)
    total_downloads: int = 0
    total_likes: int = 0
    top_model: Optional[str] = None
    has_official_impl: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_title": self.paper_title,
            "arxiv_id": self.arxiv_id,
            "models": self.models,
            "total_downloads": self.total_downloads,
            "total_likes": self.total_likes,
            "top_model": self.top_model,
            "has_official_impl": self.has_official_impl,
            "model_count": len(self.models),
        }


class HuggingFaceAgent(BaseAgent):
    """
    HuggingFace 模型元数据获取 Agent
    
    功能:
    - 根据论文标题/arXiv ID 搜索关联模型
    - 获取模型下载量、点赞数、参数量
    - 识别官方实现
    
    输入:
    - paper_title: 论文标题
    - arxiv_id: arXiv ID (可选)
    
    输出:
    - HuggingFaceResult 包含模型列表和统计信息
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._client = None
    
    @property
    def hf_client(self):
        """延迟初始化 HuggingFace 客户端"""
        if self._client is None:
            try:
                from paperbot.infrastructure.api_clients.huggingface_client import HuggingFaceClient
                token = self.config.get("huggingface_token")
                self._client = HuggingFaceClient(token=token)
            except Exception as e:
                self.logger.warning(f"HuggingFace 客户端初始化失败: {e}")
                self._client = None
        return self._client
    
    def _validate_input(self, paper_title: str = None, **kwargs) -> Optional[Dict[str, Any]]:
        """验证输入"""
        if not paper_title:
            return {"status": "error", "error": "paper_title 是必需的"}
        return None
    
    async def _execute(
        self,
        paper_title: str,
        arxiv_id: Optional[str] = None,
        limit: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行 HuggingFace 模型搜索
        
        Args:
            paper_title: 论文标题
            arxiv_id: arXiv ID
            limit: 返回模型数量限制
            
        Returns:
            HuggingFaceResult 字典
        """
        if not self.hf_client:
            return {
                "status": "error",
                "error": "HuggingFace 客户端不可用",
            }
        
        self.logger.info(f"搜索论文关联模型: {paper_title[:50]}...")
        
        # 搜索模型
        models = self.hf_client.search_by_paper(
            paper_title=paper_title,
            arxiv_id=arxiv_id,
            limit=limit,
        )
        
        if not models:
            return HuggingFaceResult(
                paper_title=paper_title,
                arxiv_id=arxiv_id,
            ).to_dict()
        
        # 统计信息
        total_downloads = sum(m.downloads for m in models)
        total_likes = sum(m.likes for m in models)
        
        # 识别 top 模型
        top_model = max(models, key=lambda m: m.downloads)
        
        # 检测是否有官方实现 (启发式: model_id 包含论文作者或关键词)
        has_official = self._detect_official_impl(models, paper_title)
        
        result = HuggingFaceResult(
            paper_title=paper_title,
            arxiv_id=arxiv_id,
            models=[m.to_dict() for m in models],
            total_downloads=total_downloads,
            total_likes=total_likes,
            top_model=top_model.model_id,
            has_official_impl=has_official,
        )
        
        self.logger.info(
            f"找到 {len(models)} 个模型, "
            f"总下载量: {total_downloads}, 总点赞: {total_likes}"
        )
        
        return result.to_dict()
    
    def _detect_official_impl(
        self, 
        models: List, 
        paper_title: str
    ) -> bool:
        """
        启发式检测是否有官方实现
        
        检查:
        - model_id 中是否包含 "official"
        - 高下载量 + 权威作者 (meta, google, openai)
        """
        title_lower = paper_title.lower()
        official_authors = {"meta-llama", "google", "openai", "microsoft", "facebook"}
        
        for model in models:
            model_id = model.model_id.lower()
            
            # 显式标记
            if "official" in model_id:
                return True
            
            # 权威作者 + 高下载量
            author = model_id.split("/")[0] if "/" in model_id else ""
            if author in official_authors and model.downloads > 10000:
                return True
        
        return False
