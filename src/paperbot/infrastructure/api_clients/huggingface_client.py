# src/paperbot/infrastructure/api_clients/huggingface_client.py
"""
HuggingFace Hub API 客户端

封装 huggingface_hub 库，提供模型元数据获取功能。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from huggingface_hub import HfApi, ModelInfo
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("huggingface_hub 未安装: pip install huggingface_hub")


@dataclass
class HFModelMeta:
    """HuggingFace 模型元数据"""
    model_id: str
    downloads: int = 0
    likes: int = 0
    pipeline_tag: Optional[str] = None    # text-generation, image-classification
    library_name: Optional[str] = None    # transformers, diffusers
    tags: List[str] = field(default_factory=list)
    author: Optional[str] = None
    last_modified: Optional[str] = None
    private: bool = False
    
    # 可选字段 (需要额外 API 调用)
    parameter_count: Optional[int] = None
    model_card_content: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "downloads": self.downloads,
            "likes": self.likes,
            "pipeline_tag": self.pipeline_tag,
            "library_name": self.library_name,
            "tags": self.tags,
            "author": self.author,
            "parameter_count": self.parameter_count,
        }


class HuggingFaceClient:
    """
    HuggingFace Hub API 客户端
    
    功能:
    - 搜索模型
    - 获取模型详细信息
    - 解析 Model Card
    """
    
    def __init__(self, token: Optional[str] = None):
        """
        初始化客户端
        
        Args:
            token: HuggingFace API token (可选，用于私有仓库)
        """
        if not HF_AVAILABLE:
            raise RuntimeError("需要安装 huggingface_hub: pip install huggingface_hub")
        
        self.api = HfApi(token=token)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def search_models(
        self,
        query: str,
        limit: int = 10,
        filter_tags: Optional[List[str]] = None,
    ) -> List[HFModelMeta]:
        """
        搜索模型
        
        Args:
            query: 搜索关键词 (论文标题或模型名)
            limit: 返回数量限制
            filter_tags: 过滤标签 (如 ["pytorch", "text-generation"])
            
        Returns:
            模型元数据列表
        """
        try:
            models = self.api.list_models(
                search=query,
                limit=limit,
                filter=filter_tags,
                sort="downloads",
                direction=-1,  # 按下载量降序
            )
            
            results = []
            for model in models:
                meta = self._parse_model_info(model)
                results.append(meta)
            
            self.logger.info(f"搜索 '{query}' 找到 {len(results)} 个模型")
            return results
            
        except Exception as e:
            self.logger.error(f"搜索模型失败: {e}")
            return []
    
    def get_model_info(self, model_id: str) -> Optional[HFModelMeta]:
        """
        获取模型详细信息
        
        Args:
            model_id: 模型 ID (如 "meta-llama/Llama-2-7b")
            
        Returns:
            模型元数据或 None
        """
        try:
            info = self.api.model_info(model_id)
            return self._parse_model_info(info)
        except Exception as e:
            self.logger.error(f"获取模型信息失败 ({model_id}): {e}")
            return None
    
    def get_model_card(self, model_id: str) -> Optional[str]:
        """
        获取模型 README/Model Card 内容
        
        Args:
            model_id: 模型 ID
            
        Returns:
            Model Card 文本或 None
        """
        try:
            from huggingface_hub import hf_hub_download
            
            readme_path = hf_hub_download(
                repo_id=model_id,
                filename="README.md",
                repo_type="model",
            )
            
            with open(readme_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            self.logger.debug(f"获取 Model Card 失败 ({model_id}): {e}")
            return None
    
    def search_by_paper(
        self,
        paper_title: str,
        arxiv_id: Optional[str] = None,
        limit: int = 5,
    ) -> List[HFModelMeta]:
        """
        根据论文搜索关联模型
        
        Args:
            paper_title: 论文标题
            arxiv_id: arXiv ID (如 "2307.09288")
            limit: 返回数量限制
            
        Returns:
            模型元数据列表
        """
        results = []
        
        # 策略 1: 使用 arXiv ID 搜索
        if arxiv_id:
            arxiv_models = self.search_models(arxiv_id, limit=limit)
            results.extend(arxiv_models)
        
        # 策略 2: 使用论文标题关键词搜索
        # 提取标题关键词 (去除常见词)
        keywords = self._extract_keywords(paper_title)
        if keywords:
            title_models = self.search_models(keywords, limit=limit)
            # 去重
            existing_ids = {m.model_id for m in results}
            for m in title_models:
                if m.model_id not in existing_ids:
                    results.append(m)
        
        return results[:limit]
    
    def _parse_model_info(self, info) -> HFModelMeta:
        """解析 ModelInfo 到 HFModelMeta"""
        return HFModelMeta(
            model_id=info.id if hasattr(info, 'id') else str(info.modelId),
            downloads=getattr(info, 'downloads', 0) or 0,
            likes=getattr(info, 'likes', 0) or 0,
            pipeline_tag=getattr(info, 'pipeline_tag', None),
            library_name=getattr(info, 'library_name', None),
            tags=list(getattr(info, 'tags', []) or []),
            author=getattr(info, 'author', None),
            last_modified=str(getattr(info, 'lastModified', '')),
            private=getattr(info, 'private', False),
        )
    
    def _extract_keywords(self, title: str) -> str:
        """从标题提取关键词"""
        # 移除常见停用词
        stopwords = {
            'a', 'an', 'the', 'is', 'are', 'for', 'of', 'to', 'in', 'on', 
            'with', 'by', 'and', 'or', 'via', 'using', 'towards', 'toward'
        }
        words = title.lower().split()
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        return ' '.join(keywords[:5])  # 取前5个关键词
