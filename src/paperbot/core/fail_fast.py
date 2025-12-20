# src/paperbot/core/fail_fast.py
"""
Fail-Fast 评估器

P3 增强:
- 在管道早期识别低价值论文
- 节省计算资源
- 支持可配置阈值
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class FailFastConfig:
    """
    Fail-Fast 配置
    
    Attributes:
        min_research_score: 研究阶段最低分,低于此跳过深度分析
        min_code_health_score: 代码健康最低分,低于此跳过质量评估
        skip_quality_if_no_code: 无代码时是否跳过质量评估
        early_exit_threshold: 极低分触发提前退出
        enabled: 是否启用 Fail-Fast
    """
    min_research_score: float = 20.0
    min_code_health_score: float = 15.0
    skip_quality_if_no_code: bool = True
    early_exit_threshold: float = 10.0
    enabled: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailFastConfig":
        """从字典创建配置"""
        return cls(
            min_research_score=data.get("min_research_score", 20.0),
            min_code_health_score=data.get("min_code_health_score", 15.0),
            skip_quality_if_no_code=data.get("skip_quality_if_no_code", True),
            early_exit_threshold=data.get("early_exit_threshold", 10.0),
            enabled=data.get("enabled", True),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_research_score": self.min_research_score,
            "min_code_health_score": self.min_code_health_score,
            "skip_quality_if_no_code": self.skip_quality_if_no_code,
            "early_exit_threshold": self.early_exit_threshold,
            "enabled": self.enabled,
        }


@dataclass
class SkipDecision:
    """跳过决策结果"""
    should_skip: bool
    reason: str
    skipped_stages: list = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_skip": self.should_skip,
            "reason": self.reason,
            "skipped_stages": self.skipped_stages,
        }


class FailFastEvaluator:
    """
    Fail-Fast 评估器
    
    在管道早期识别低价值论文,节省资源。
    
    判断逻辑:
    1. 研究评分 < 阈值 → 跳过代码分析
    2. 代码健康分 < 阈值 → 跳过质量评估
    3. 无代码 → 可选跳过质量评估
    4. 极低分 → 提前终止整个管道
    """
    
    def __init__(self, config: Optional[FailFastConfig] = None):
        """
        初始化评估器
        
        Args:
            config: Fail-Fast 配置
        """
        self.config = config or FailFastConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def should_skip_stage(
        self,
        stage: str,
        score_bus,  # ScoreShareBus type hint avoided for circular import
        has_code: bool = True,
    ) -> SkipDecision:
        """
        判断是否应跳过某阶段
        
        Args:
            stage: 目标阶段名称
            score_bus: 评分共享总线
            has_code: 论文是否有代码
            
        Returns:
            SkipDecision 包含是否跳过及原因
        """
        if not self.config.enabled:
            return SkipDecision(should_skip=False, reason="Fail-Fast disabled")
        
        # 代码分析阶段判断
        if stage == "code":
            research_score = score_bus.get_score("research")
            if research_score and research_score.score < self.config.min_research_score:
                return SkipDecision(
                    should_skip=True,
                    reason=f"研究评分过低 ({research_score.score:.1f} < {self.config.min_research_score})",
                    skipped_stages=["code"]
                )
        
        # 质量评估阶段判断
        if stage == "quality":
            # 无代码跳过
            if not has_code and self.config.skip_quality_if_no_code:
                return SkipDecision(
                    should_skip=True,
                    reason="无代码,跳过质量评估",
                    skipped_stages=["quality"]
                )
            
            # 代码健康分过低
            code_score = score_bus.get_score("code")
            if code_score:
                is_empty = code_score.key_metrics.get("is_empty_repo", False)
                if is_empty or code_score.score < self.config.min_code_health_score:
                    return SkipDecision(
                        should_skip=True,
                        reason=f"代码仓库质量过低 ({code_score.score:.1f})",
                        skipped_stages=["quality"]
                    )
        
        return SkipDecision(should_skip=False, reason="通过检查")
    
    def evaluate_early_exit(
        self,
        score_bus,
    ) -> SkipDecision:
        """
        评估是否应提前终止整个管道
        
        Args:
            score_bus: 评分共享总线
            
        Returns:
            SkipDecision 包含是否提前退出及原因
        """
        if not self.config.enabled:
            return SkipDecision(should_skip=False, reason="Fail-Fast disabled")
        
        # 检查是否有极低分阶段
        lowest = score_bus.get_lowest_score()
        if lowest and lowest.score < self.config.early_exit_threshold:
            return SkipDecision(
                should_skip=True,
                reason=f"阶段 '{lowest.stage}' 评分过低 ({lowest.score:.1f}),提前终止",
                skipped_stages=["influence", "report"]
            )
        
        return SkipDecision(should_skip=False, reason="继续执行")
    
    def get_skip_stages(
        self,
        score_bus,
        has_code: bool = True,
    ) -> list:
        """
        获取所有应跳过的阶段列表
        
        Args:
            score_bus: 评分共享总线
            has_code: 论文是否有代码
            
        Returns:
            应跳过的阶段名称列表
        """
        skipped = []
        
        for stage in ["code", "quality", "influence", "report"]:
            decision = self.should_skip_stage(stage, score_bus, has_code)
            if decision.should_skip:
                skipped.extend(decision.skipped_stages)
                self.logger.info(f"Fail-Fast: 跳过 '{stage}' - {decision.reason}")
        
        return list(set(skipped))
    
    def log_decision(self, decision: SkipDecision, stage: str = "") -> None:
        """记录决策日志"""
        if decision.should_skip:
            self.logger.info(f"⚡ Fail-Fast [{stage}]: {decision.reason}")
        else:
            self.logger.debug(f"✓ Fail-Fast [{stage}]: {decision.reason}")
