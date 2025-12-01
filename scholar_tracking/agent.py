"""
Scholar Tracking Agent 主类

来源: BettaFish/QueryEngine/agent.py
适配: PaperBot 学者追踪系统

整合所有模块，实现完整的学者追踪流程：
1. 获取学者信息
2. 检测新论文
3. 分析代码仓库
4. 计算影响力
5. 生成报告
"""

import os
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from loguru import logger

from core.llm_client import LLMClient
from core.state import TrackingState, TrackingStage, ScholarState, PaperState, InfluenceState
from config.settings import Settings, get_settings


class ScholarTrackingAgent:
    """
    学者追踪 Agent 主类
    
    整合学者数据获取、论文追踪、代码分析、影响力计算等功能
    """
    
    def __init__(self, config: Optional[Settings] = None):
        """
        初始化 Scholar Tracking Agent
        
        Args:
            config: 配置对象，如果不提供则自动加载
        """
        self.config = config or get_settings()
        
        # 初始化LLM客户端（可选，用于智能分析）
        self.llm_client: Optional[LLMClient] = None
        if self.config.llm_api_key:
            try:
                self.llm_client = self._initialize_llm()
            except Exception as e:
                logger.warning(f"LLM 客户端初始化失败，将使用基础功能: {e}")
        
        # 初始化工具集
        self._initialize_tools()
        
        # 初始化节点
        self._initialize_nodes()
        
        # 状态管理
        self.state = TrackingState()
        
        # 确保输出目录存在
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.cache_dir, exist_ok=True)
        
        logger.info(f"Scholar Tracking Agent 已初始化")
        if self.llm_client:
            logger.info(f"使用 LLM: {self.llm_client.get_model_info()}")
        logger.info(f"输出目录: {self.config.output_dir}")
        logger.info(f"缓存目录: {self.config.cache_dir}")
    
    def _initialize_llm(self) -> LLMClient:
        """初始化 LLM 客户端"""
        return LLMClient(
            api_key=self.config.llm_api_key,
            model_name=self.config.llm_model_name,
            base_url=self.config.llm_base_url,
        )
    
    def _initialize_tools(self):
        """初始化工具集"""
        # 延迟导入，避免循环依赖
        from scholar_tracking.agents.semantic_scholar_agent import SemanticScholarAgent
        from scholar_tracking.agents.paper_tracker_agent import PaperTrackerAgent
        from scholar_tracking.agents.scholar_profile_agent import ScholarProfileAgent
        
        # Semantic Scholar API 代理
        self.semantic_scholar = SemanticScholarAgent(
            api_key=self.config.semantic_scholar_api_key,
            cache_dir=self.config.cache_dir
        )
        
        # 论文追踪代理
        self.paper_tracker = PaperTrackerAgent(
            semantic_scholar=self.semantic_scholar,
            cache_dir=self.config.cache_dir
        )
        
        # 学者档案代理
        self.scholar_profile = ScholarProfileAgent(
            semantic_scholar=self.semantic_scholar,
            cache_dir=self.config.cache_dir
        )
        
        logger.info("工具集初始化完成")
    
    def _initialize_nodes(self):
        """初始化处理节点"""
        # 节点将在需要时懒加载
        self._nodes_initialized = False
    
    def _ensure_nodes(self):
        """确保节点已初始化"""
        if self._nodes_initialized:
            return
        
        # 延迟导入节点
        from scholar_tracking.nodes import (
            ScholarFetchNode,
            PaperDetectionNode,
            InfluenceCalculationNode,
            ReportGenerationNode,
        )
        
        self.scholar_fetch_node = ScholarFetchNode(
            scholar_profile=self.scholar_profile,
            llm_client=self.llm_client
        )
        
        self.paper_detection_node = PaperDetectionNode(
            paper_tracker=self.paper_tracker,
            llm_client=self.llm_client
        )
        
        # 影响力计算节点
        from influence.calculator import InfluenceCalculator
        self.influence_calculator = InfluenceCalculator()
        self.influence_node = InfluenceCalculationNode(
            calculator=self.influence_calculator,
            llm_client=self.llm_client
        )
        
        # 报告生成节点
        from reports.report_writer import ReportWriter
        self.report_writer = ReportWriter(
            output_dir=self.config.output_dir,
            template_dir=os.path.join(os.path.dirname(__file__), "reports", "templates")
        )
        self.report_node = ReportGenerationNode(
            report_writer=self.report_writer,
            llm_client=self.llm_client
        )
        
        self._nodes_initialized = True
        logger.info("处理节点初始化完成")
    
    # ============ 主要 API ============
    
    def track(
        self,
        scholar_ids: Optional[List[str]] = None,
        save_report: bool = True,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict[str, Any]:
        """
        执行完整的学者追踪流程
        
        Args:
            scholar_ids: 要追踪的学者ID列表，如果不提供则使用配置中的订阅列表
            save_report: 是否保存报告到文件
            progress_callback: 进度回调函数 (stage_name, progress)
            
        Returns:
            追踪结果字典，包含新论文、影响力变化等信息
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"开始学者追踪")
        logger.info(f"{'='*60}")
        
        self._ensure_nodes()
        
        # 初始化状态
        self.state = TrackingState(
            task_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            task_name="scholar_tracking"
        )
        
        try:
            # Step 1: 获取学者信息
            self._update_progress("获取学者信息", 0.1, progress_callback)
            self._fetch_scholars(scholar_ids)
            
            # Step 2: 检测新论文
            self._update_progress("检测新论文", 0.3, progress_callback)
            self._detect_new_papers()
            
            # Step 3: 分析代码仓库（如果有）
            self._update_progress("分析代码仓库", 0.5, progress_callback)
            self._analyze_code_repos()
            
            # Step 4: 计算影响力
            self._update_progress("计算影响力", 0.7, progress_callback)
            self._calculate_influence()
            
            # Step 5: 生成报告
            self._update_progress("生成报告", 0.9, progress_callback)
            report_path = None
            if save_report:
                report_path = self._generate_report()
            
            # 完成
            self.state.mark_completed()
            self._update_progress("完成", 1.0, progress_callback)
            
            logger.info(f"\n{'='*60}")
            logger.info("学者追踪完成！")
            logger.info(f"{'='*60}")
            
            return self._build_result(report_path)
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(f"追踪过程中发生错误: {str(e)}\n错误堆栈: {error_traceback}")
            self.state.mark_failed(str(e))
            raise
    
    def track_single_scholar(
        self,
        scholar_id: str,
        save_report: bool = True
    ) -> Dict[str, Any]:
        """
        追踪单个学者
        
        Args:
            scholar_id: Semantic Scholar 学者ID
            save_report: 是否保存报告
            
        Returns:
            追踪结果
        """
        return self.track(scholar_ids=[scholar_id], save_report=save_report)
    
    def get_scholar_info(self, scholar_id: str) -> Optional[ScholarState]:
        """
        获取学者信息
        
        Args:
            scholar_id: Semantic Scholar 学者ID
            
        Returns:
            学者状态对象
        """
        try:
            author_data = self.semantic_scholar.get_author(scholar_id)
            if author_data:
                return ScholarState(
                    scholar_id=scholar_id,
                    name=author_data.get("name", ""),
                    semantic_scholar_id=author_data.get("authorId", scholar_id),
                    affiliations=author_data.get("affiliations", []),
                    h_index=author_data.get("hIndex", 0),
                    citation_count=author_data.get("citationCount", 0),
                    paper_count=author_data.get("paperCount", 0),
                )
        except Exception as e:
            logger.error(f"获取学者信息失败: {e}")
        return None
    
    def get_new_papers(self, scholar_id: str, since_days: int = 30) -> List[PaperState]:
        """
        获取学者的新论文
        
        Args:
            scholar_id: 学者ID
            since_days: 查询最近多少天的论文
            
        Returns:
            新论文列表
        """
        try:
            papers = self.paper_tracker.get_new_papers(scholar_id, since_days=since_days)
            return [
                PaperState(
                    paper_id=p.get("paperId", ""),
                    title=p.get("title", ""),
                    authors=[a.get("name", "") for a in p.get("authors", [])],
                    venue=p.get("venue", ""),
                    year=p.get("year", 0),
                    citation_count=p.get("citationCount", 0),
                    abstract=p.get("abstract", ""),
                    url=p.get("url", ""),
                    is_new=True,
                )
                for p in papers
            ]
        except Exception as e:
            logger.error(f"获取新论文失败: {e}")
            return []
    
    # ============ 内部方法 ============
    
    def _update_progress(
        self,
        stage_name: str,
        progress: float,
        callback: Optional[Callable[[str, float], None]]
    ):
        """更新进度"""
        self.state.progress = progress
        logger.info(f"[{progress*100:.0f}%] {stage_name}")
        if callback:
            callback(stage_name, progress)
    
    def _fetch_scholars(self, scholar_ids: Optional[List[str]]):
        """获取学者信息"""
        logger.info(f"\n[步骤 1] 获取学者信息...")
        self.state.update_stage(TrackingStage.FETCHING_SCHOLARS)
        
        # 如果没有指定学者ID，从配置加载
        if not scholar_ids:
            from config.scholar_subscriptions import load_scholar_subscriptions
            subscriptions = load_scholar_subscriptions()
            scholar_ids = [s.get("semantic_scholar_id") for s in subscriptions.get("scholars", [])]
        
        if not scholar_ids:
            logger.warning("没有找到要追踪的学者")
            return
        
        logger.info(f"将追踪 {len(scholar_ids)} 位学者")
        
        for scholar_id in scholar_ids:
            scholar_state = self.get_scholar_info(scholar_id)
            if scholar_state:
                self.state.add_scholar(scholar_state)
                logger.info(f"  ✓ {scholar_state.name} (h-index: {scholar_state.h_index})")
            else:
                logger.warning(f"  ✗ 无法获取学者信息: {scholar_id}")
    
    def _detect_new_papers(self):
        """检测新论文"""
        logger.info(f"\n[步骤 2] 检测新论文...")
        self.state.update_stage(TrackingStage.FETCHING_PAPERS)
        
        for scholar in self.state.scholars:
            logger.info(f"  检查 {scholar.name} 的新论文...")
            new_papers = self.get_new_papers(scholar.scholar_id)
            
            for paper in new_papers:
                self.state.add_paper(paper)
                logger.info(f"    ✓ 新论文: {paper.title[:50]}...")
            
            if not new_papers:
                logger.info(f"    - 没有发现新论文")
        
        logger.info(f"共发现 {len(self.state.new_papers)} 篇新论文")
    
    def _analyze_code_repos(self):
        """分析代码仓库"""
        logger.info(f"\n[步骤 3] 分析代码仓库...")
        self.state.update_stage(TrackingStage.ANALYZING_CODE)
        
        # 遍历新论文，查找代码仓库
        for paper in self.state.new_papers:
            if paper.github_url:
                logger.info(f"  分析: {paper.github_url}")
                # TODO: 集成代码分析功能
            else:
                # 尝试从论文中提取GitHub链接
                # TODO: 使用LLM或规则从摘要中提取
                pass
    
    def _calculate_influence(self):
        """计算影响力"""
        logger.info(f"\n[步骤 4] 计算影响力...")
        self.state.update_stage(TrackingStage.CALCULATING_INFLUENCE)
        
        for scholar in self.state.scholars:
            # 收集学者的论文数据
            papers = [p for p in self.state.papers if scholar.name in p.authors]
            
            # 计算影响力分数
            influence = self.influence_calculator.calculate(
                scholar_id=scholar.scholar_id,
                papers=papers,
                h_index=scholar.h_index,
                citation_count=scholar.citation_count,
            )
            
            influence_state = InfluenceState(
                scholar_id=scholar.scholar_id,
                academic_score=influence.get("academic_score", 0),
                engineering_score=influence.get("engineering_score", 0),
                total_score=influence.get("total_score", 0),
                tier1_papers=influence.get("tier1_papers", 0),
                tier2_papers=influence.get("tier2_papers", 0),
                github_stars=influence.get("github_stars", 0),
                github_forks=influence.get("github_forks", 0),
            )
            
            self.state.add_influence(influence_state)
            logger.info(f"  {scholar.name}: 总分 {influence_state.total_score:.2f}")
    
    def _generate_report(self) -> str:
        """生成报告"""
        logger.info(f"\n[步骤 5] 生成报告...")
        self.state.update_stage(TrackingStage.GENERATING_REPORT)
        
        # 准备报告数据
        report_data = {
            "task_id": self.state.task_id,
            "generated_at": datetime.now().isoformat(),
            "scholars": [s.to_dict() for s in self.state.scholars],
            "new_papers": [p.to_dict() for p in self.state.new_papers],
            "influence_results": [i.to_dict() for i in self.state.influence_results],
            "summary": self._generate_summary(),
        }
        
        # 生成报告文件
        report_path = self.report_writer.write(report_data)
        logger.info(f"报告已保存到: {report_path}")
        
        # 保存状态
        state_path = os.path.join(
            self.config.cache_dir,
            f"state_{self.state.task_id}.json"
        )
        self.state.save_to_file(state_path)
        logger.info(f"状态已保存到: {state_path}")
        
        return report_path
    
    def _generate_summary(self) -> Dict[str, Any]:
        """生成追踪摘要"""
        return {
            "total_scholars": len(self.state.scholars),
            "total_new_papers": len(self.state.new_papers),
            "papers_by_scholar": {
                s.name: len([p for p in self.state.new_papers if s.name in p.authors])
                for s in self.state.scholars
            },
            "top_influence": sorted(
                self.state.influence_results,
                key=lambda x: x.total_score,
                reverse=True
            )[:5] if self.state.influence_results else [],
        }
    
    def _build_result(self, report_path: Optional[str]) -> Dict[str, Any]:
        """构建返回结果"""
        return {
            "success": True,
            "task_id": self.state.task_id,
            "report_path": report_path,
            "summary": {
                "scholars_tracked": len(self.state.scholars),
                "new_papers_found": len(self.state.new_papers),
                "errors": len(self.state.errors),
            },
            "new_papers": [p.to_dict() for p in self.state.new_papers],
            "influence_results": [i.to_dict() for i in self.state.influence_results],
            "progress": self.state.get_progress_summary(),
        }
    
    # ============ 状态管理 ============
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """获取进度摘要"""
        return self.state.get_progress_summary()
    
    def load_state(self, filepath: str):
        """从文件加载状态"""
        self.state = TrackingState.load_from_file(filepath)
        logger.info(f"状态已从 {filepath} 加载")
    
    def save_state(self, filepath: str):
        """保存状态到文件"""
        self.state.save_to_file(filepath)
        logger.info(f"状态已保存到 {filepath}")


def create_agent(config: Optional[Settings] = None) -> ScholarTrackingAgent:
    """
    创建 Scholar Tracking Agent 实例的便捷函数
    
    Args:
        config: 可选的配置对象
        
    Returns:
        ScholarTrackingAgent 实例
    """
    return ScholarTrackingAgent(config)
