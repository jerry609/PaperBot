# src/paperbot/agents/openreview/agent.py
"""
OpenReview Agent

从 OpenReview 获取论文审稿意见和决策信息。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from paperbot.agents.base import BaseAgent

logger = logging.getLogger(__name__)


@dataclass
class OpenReviewResult:
    """OpenReview 分析结果"""
    paper_title: str
    venue: Optional[str] = None
    paper_id: Optional[str] = None
    decision: Optional[str] = None
    avg_rating: float = 0.0
    avg_confidence: float = 0.0
    review_count: int = 0
    reviews: List[Dict[str, Any]] = field(default_factory=list)
    weaknesses_summary: str = ""
    strengths_summary: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_title": self.paper_title,
            "venue": self.venue,
            "paper_id": self.paper_id,
            "decision": self.decision,
            "avg_rating": self.avg_rating,
            "avg_confidence": self.avg_confidence,
            "review_count": self.review_count,
            "reviews": self.reviews,
            "weaknesses_summary": self.weaknesses_summary,
            "strengths_summary": self.strengths_summary,
        }


class OpenReviewAgent(BaseAgent):
    """
    OpenReview 审稿意见获取 Agent
    
    功能:
    - 搜索论文在 OpenReview 上的提交
    - 获取审稿评分和意见
    - 提取关键薄弱点 (使用 LLM 总结)
    
    输入:
    - paper_title: 论文标题
    - venue: 会议名称 (可选, 如 "iclr", "neurips")
    
    输出:
    - OpenReviewResult 包含评审信息和总结
    """
    
    # 支持的会议列表
    SUPPORTED_VENUES = ["iclr", "neurips", "icml", "aaai", "emnlp", "acl"]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._client = None
    
    @property
    def or_client(self):
        """延迟初始化 OpenReview 客户端"""
        if self._client is None:
            try:
                from paperbot.infrastructure.api_clients.openreview_client import OpenReviewClient
                username = self.config.get("openreview_username")
                password = self.config.get("openreview_password")
                self._client = OpenReviewClient(username=username, password=password)
            except Exception as e:
                self.logger.warning(f"OpenReview 客户端初始化失败: {e}")
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
        venue: Optional[str] = None,
        summarize_weaknesses: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行 OpenReview 查询
        
        Args:
            paper_title: 论文标题
            venue: 会议名称
            summarize_weaknesses: 是否用 LLM 总结薄弱点
            
        Returns:
            OpenReviewResult 字典
        """
        if not self.or_client:
            return {
                "status": "error",
                "error": "OpenReview 客户端不可用",
            }
        
        self.logger.info(f"查询 OpenReview: {paper_title[:50]}...")
        
        # 获取论文和评审
        paper = self.or_client.get_paper_with_reviews(
            title=paper_title,
            venue=venue,
        )
        
        if not paper:
            return OpenReviewResult(
                paper_title=paper_title,
                venue=venue,
            ).to_dict()
        
        # 构建结果
        result = OpenReviewResult(
            paper_title=paper_title,
            venue=paper.venue,
            paper_id=paper.paper_id,
            decision=paper.decision,
            avg_rating=round(paper.avg_rating, 2),
            avg_confidence=round(paper.avg_confidence, 2),
            review_count=len(paper.reviews),
            reviews=[r.to_dict() for r in paper.reviews],
        )
        
        # LLM 总结薄弱点
        if summarize_weaknesses and paper.reviews and self.client:
            result.weaknesses_summary = await self._summarize_weaknesses(paper.reviews)
            result.strengths_summary = await self._summarize_strengths(paper.reviews)
        else:
            # 简单拼接
            result.weaknesses_summary = self._simple_combine([r.weaknesses for r in paper.reviews])
            result.strengths_summary = self._simple_combine([r.strengths for r in paper.reviews])
        
        self.logger.info(
            f"找到 {result.review_count} 条评审, "
            f"平均评分: {result.avg_rating}, 决策: {result.decision}"
        )
        
        return result.to_dict()
    
    async def _summarize_weaknesses(self, reviews: List) -> str:
        """使用 LLM 总结所有评审的薄弱点"""
        weaknesses_text = "\n\n".join([
            f"Reviewer {i+1}: {r.weaknesses}"
            for i, r in enumerate(reviews)
            if r.weaknesses
        ])
        
        if not weaknesses_text:
            return ""
        
        prompt = f"""请简洁总结以下论文评审意见中提到的主要薄弱点和问题:

{weaknesses_text[:2000]}

请用中文列出 3-5 个关键问题，每个问题一行。"""
        
        try:
            return await self.ask_claude(prompt, max_tokens=500)
        except Exception as e:
            self.logger.warning(f"LLM 总结失败: {e}")
            return self._simple_combine([r.weaknesses for r in reviews])
    
    async def _summarize_strengths(self, reviews: List) -> str:
        """使用 LLM 总结所有评审的优点"""
        strengths_text = "\n\n".join([
            f"Reviewer {i+1}: {r.strengths}"
            for i, r in enumerate(reviews)
            if r.strengths
        ])
        
        if not strengths_text:
            return ""
        
        prompt = f"""请简洁总结以下论文评审意见中提到的主要优点:

{strengths_text[:2000]}

请用中文列出 3-5 个关键优点，每个优点一行。"""
        
        try:
            return await self.ask_claude(prompt, max_tokens=500)
        except Exception as e:
            self.logger.warning(f"LLM 总结失败: {e}")
            return self._simple_combine([r.strengths for r in reviews])
    
    def _simple_combine(self, texts: List[str], max_length: int = 500) -> str:
        """简单合并文本"""
        combined = " | ".join([t[:200] for t in texts if t])
        return combined[:max_length]
    
    @classmethod
    def is_supported_venue(cls, venue: str) -> bool:
        """检查是否支持该会议"""
        if not venue:
            return False
        venue_lower = venue.lower()
        return any(v in venue_lower for v in cls.SUPPORTED_VENUES)
