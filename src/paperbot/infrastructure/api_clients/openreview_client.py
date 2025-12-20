# src/paperbot/infrastructure/api_clients/openreview_client.py
"""
OpenReview API 客户端

封装 openreview-py 库，提供论文审稿意见获取功能。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import openreview
    OR_AVAILABLE = True
except ImportError:
    OR_AVAILABLE = False
    logger.warning("openreview-py 未安装: pip install openreview-py")


@dataclass
class ReviewInfo:
    """单条评审信息"""
    reviewer_id: str
    rating: Optional[float] = None
    confidence: Optional[float] = None
    summary: str = ""
    strengths: str = ""
    weaknesses: str = ""
    questions: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "reviewer_id": self.reviewer_id,
            "rating": self.rating,
            "confidence": self.confidence,
            "summary": self.summary[:500] if self.summary else "",
            "strengths": self.strengths[:500] if self.strengths else "",
            "weaknesses": self.weaknesses[:500] if self.weaknesses else "",
        }


@dataclass
class OpenReviewPaper:
    """OpenReview 论文信息"""
    paper_id: str
    title: str
    venue: str
    decision: Optional[str] = None
    abstract: str = ""
    authors: List[str] = field(default_factory=list)
    reviews: List[ReviewInfo] = field(default_factory=list)
    meta_review: str = ""
    
    @property
    def avg_rating(self) -> float:
        """计算平均评分"""
        ratings = [r.rating for r in self.reviews if r.rating is not None]
        return sum(ratings) / len(ratings) if ratings else 0.0
    
    @property
    def avg_confidence(self) -> float:
        """计算平均置信度"""
        confs = [r.confidence for r in self.reviews if r.confidence is not None]
        return sum(confs) / len(confs) if confs else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "venue": self.venue,
            "decision": self.decision,
            "avg_rating": round(self.avg_rating, 2),
            "avg_confidence": round(self.avg_confidence, 2),
            "review_count": len(self.reviews),
            "reviews": [r.to_dict() for r in self.reviews],
            "meta_review": self.meta_review[:500] if self.meta_review else "",
        }


class OpenReviewClient:
    """
    OpenReview API 客户端
    
    功能:
    - 搜索论文
    - 获取审稿意见
    - 提取决策和评分
    
    支持的会议:
    - ICLR, NeurIPS, ICML, AAAI 等
    """
    
    # 常见 ML 会议的 venue ID 映射
    VENUE_MAPPING = {
        "iclr": "ICLR.cc",
        "neurips": "NeurIPS.cc",
        "icml": "ICML.cc",
        "aaai": "AAAI.org",
    }
    
    def __init__(
        self, 
        username: Optional[str] = None, 
        password: Optional[str] = None
    ):
        """
        初始化客户端
        
        Args:
            username: OpenReview 用户名 (可选)
            password: OpenReview 密码 (可选)
        """
        if not OR_AVAILABLE:
            raise RuntimeError("需要安装 openreview-py: pip install openreview-py")
        
        try:
            # 尝试 API v2
            self.client = openreview.api.OpenReviewClient(
                baseurl='https://api2.openreview.net',
                username=username,
                password=password
            )
            self.api_version = 2
        except Exception:
            # 回退到 API v1
            self.client = openreview.Client(
                baseurl='https://openreview.net',
                username=username,
                password=password
            )
            self.api_version = 1
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"OpenReview 客户端初始化 (API v{self.api_version})")
    
    def search_paper(
        self,
        title: str,
        venue: Optional[str] = None,
    ) -> Optional[OpenReviewPaper]:
        """
        搜索论文
        
        Args:
            title: 论文标题
            venue: 会议名称 (如 "iclr", "neurips")
            
        Returns:
            OpenReviewPaper 或 None
        """
        try:
            # 构建搜索参数
            venue_id = self._resolve_venue(venue) if venue else None
            
            if self.api_version == 2:
                notes = self.client.search_notes(
                    term=title,
                    venue=venue_id,
                    limit=5,
                )
            else:
                notes = list(self.client.search_notes(
                    content={'title': title},
                    limit=5,
                ))
            
            if not notes:
                self.logger.info(f"未找到论文: {title[:50]}")
                return None
            
            # 取最匹配的
            note = notes[0]
            return self._parse_paper(note)
            
        except Exception as e:
            self.logger.error(f"搜索论文失败: {e}")
            return None
    
    def get_reviews(self, paper_id: str) -> List[ReviewInfo]:
        """
        获取论文的审稿意见
        
        Args:
            paper_id: 论文 ID (forum ID)
            
        Returns:
            ReviewInfo 列表
        """
        try:
            if self.api_version == 2:
                reviews = self.client.get_notes(
                    forum=paper_id,
                    signature='.*',
                )
            else:
                reviews = self.client.get_notes(forum=paper_id)
            
            # 过滤出评审意见
            review_notes = [
                n for n in reviews 
                if self._is_review_note(n)
            ]
            
            return [self._parse_review(r) for r in review_notes]
            
        except Exception as e:
            self.logger.error(f"获取评审失败: {e}")
            return []
    
    def get_paper_with_reviews(
        self,
        title: str,
        venue: Optional[str] = None,
    ) -> Optional[OpenReviewPaper]:
        """
        获取论文及其所有审稿意见
        
        Args:
            title: 论文标题
            venue: 会议名称
            
        Returns:
            包含评审的 OpenReviewPaper
        """
        paper = self.search_paper(title, venue)
        if not paper:
            return None
        
        paper.reviews = self.get_reviews(paper.paper_id)
        return paper
    
    def _resolve_venue(self, venue: str) -> Optional[str]:
        """解析会议名称到 venue ID"""
        venue_lower = venue.lower()
        for key, value in self.VENUE_MAPPING.items():
            if key in venue_lower:
                return value
        return venue
    
    def _is_review_note(self, note) -> bool:
        """判断是否为评审 note"""
        # 检查 invitation 或 content 特征
        invitation = getattr(note, 'invitation', '') or ''
        if 'Review' in invitation or 'review' in invitation:
            return True
        
        content = getattr(note, 'content', {}) or {}
        if 'rating' in content or 'recommendation' in content:
            return True
        
        return False
    
    def _parse_paper(self, note) -> OpenReviewPaper:
        """解析论文 note"""
        content = getattr(note, 'content', {}) or {}
        
        return OpenReviewPaper(
            paper_id=note.id if hasattr(note, 'id') else str(note.forum),
            title=self._get_content_value(content, 'title', ''),
            venue=getattr(note, 'venue', '') or '',
            decision=self._get_content_value(content, 'decision'),
            abstract=self._get_content_value(content, 'abstract', ''),
            authors=self._get_content_value(content, 'authors', []),
        )
    
    def _parse_review(self, note) -> ReviewInfo:
        """解析评审 note"""
        content = getattr(note, 'content', {}) or {}
        
        return ReviewInfo(
            reviewer_id=str(getattr(note, 'id', 'unknown')),
            rating=self._parse_rating(content),
            confidence=self._parse_confidence(content),
            summary=self._get_content_value(content, 'summary', ''),
            strengths=self._get_content_value(content, 'strengths', ''),
            weaknesses=self._get_content_value(content, 'weaknesses', ''),
            questions=self._get_content_value(content, 'questions', ''),
        )
    
    def _get_content_value(self, content: dict, key: str, default=None):
        """安全获取 content 值 (处理 v1/v2 差异)"""
        value = content.get(key, default)
        if isinstance(value, dict) and 'value' in value:
            return value['value']
        return value
    
    def _parse_rating(self, content: dict) -> Optional[float]:
        """解析评分"""
        for key in ['rating', 'recommendation', 'score']:
            value = self._get_content_value(content, key)
            if value is not None:
                try:
                    # 处理 "8: Accept" 格式
                    if isinstance(value, str) and ':' in value:
                        value = value.split(':')[0]
                    return float(value)
                except (ValueError, TypeError):
                    continue
        return None
    
    def _parse_confidence(self, content: dict) -> Optional[float]:
        """解析置信度"""
        value = self._get_content_value(content, 'confidence')
        if value is not None:
            try:
                if isinstance(value, str) and ':' in value:
                    value = value.split(':')[0]
                return float(value)
            except (ValueError, TypeError):
                pass
        return None
