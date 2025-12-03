"""
Deep Research 学者追踪 Agent

实现 BettaFish InsightEngine 的 Deep Research 模式
通过迭代反思循环深入分析学者信息

参考: BettaFish/InsightEngine/insight_engine.py
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

from agents.base_agent import BaseAgent
from core.llm_client import LLMClient
from core.state import TrackingState, ScholarState, PaperState, InfluenceState
from scholar_tracking.nodes import (
    ScholarFetchNode,
    PaperDetectionNode,
    InfluenceCalculationNode,
    ReportGenerationNode,
    ReflectionSearchNode,
    ReflectionSummaryNode,
)
from prompts.scholar_prompts import (
    SYSTEM_PROMPT_ANALYZE_PAPER,
    SYSTEM_PROMPT_ASSESS_INFLUENCE,
)
from utils.json_parser import parse_json


@dataclass
class DeepResearchConfig:
    """Deep Research 配置"""
    max_reflections: int = 3  # 最大反思次数
    min_completeness_score: float = 0.8  # 最小信息完整度
    max_papers_per_search: int = 20  # 每次搜索最多获取论文数
    enable_code_analysis: bool = True  # 是否分析代码仓库
    enable_citation_analysis: bool = True  # 是否分析引用关系


@dataclass
class ResearchState:
    """Deep Research 状态"""
    scholar_info: Dict[str, Any] = field(default_factory=dict)
    collected_papers: List[Dict[str, Any]] = field(default_factory=list)
    current_summary: str = ""
    key_findings: List[str] = field(default_factory=list)
    completeness_score: float = 0.0
    reflection_count: int = 0
    search_history: List[Dict[str, Any]] = field(default_factory=list)
    influence_result: Optional[Dict[str, Any]] = None


class DeepResearchAgent(BaseAgent):
    """
    Deep Research 学者追踪 Agent
    
    实现迭代反思循环:
    1. 初始搜索 - 获取学者基本信息和论文
    2. 总结 - 整理已收集信息
    3. 反思 - 识别信息空白
    4. 补充搜索 - 填补空白
    5. 更新总结 - 整合新信息
    6. 重复 3-5 直到信息充分或达到最大次数
    7. 生成报告 - 输出最终分析结果
    """
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        semantic_scholar_agent: Any = None,
        scholar_profile_agent: Any = None,
        influence_calculator: Any = None,
        config: Optional[DeepResearchConfig] = None,
    ):
        """
        初始化 Deep Research Agent
        
        Args:
            llm_client: LLM 客户端
            semantic_scholar_agent: Semantic Scholar API Agent
            scholar_profile_agent: 学者档案管理 Agent
            influence_calculator: 影响力计算器
            config: 研究配置
        """
        super().__init__()
        
        self.llm_client = llm_client
        self.semantic_scholar_agent = semantic_scholar_agent
        self.scholar_profile_agent = scholar_profile_agent
        self.influence_calculator = influence_calculator
        self.config = config or DeepResearchConfig()
        
        # 初始化节点
        self._init_nodes()
    
    def _init_nodes(self):
        """初始化处理节点"""
        if self.llm_client:
            self.reflection_search_node = ReflectionSearchNode(
                llm_client=self.llm_client,
                max_reflections=self.config.max_reflections,
            )
            self.reflection_summary_node = ReflectionSummaryNode(
                llm_client=self.llm_client,
            )
        else:
            self.reflection_search_node = None
            self.reflection_summary_node = None
        
        if self.scholar_profile_agent:
            self.scholar_fetch_node = ScholarFetchNode(
                scholar_profile=self.scholar_profile_agent,
            )
        else:
            self.scholar_fetch_node = None
        
        if self.influence_calculator:
            self.influence_node = InfluenceCalculationNode(
                calculator=self.influence_calculator,
            )
        else:
            self.influence_node = None
    
    async def process(self, *args, **kwargs) -> Dict[str, Any]:
        """
        BaseAgent 要求的处理方法
        """
        action = kwargs.get("action", "research")
        
        if action == "research":
            scholar_id = kwargs.get("scholar_id")
            analysis_goal = kwargs.get("goal", "全面分析学者的学术影响力")
            return await self.deep_research(scholar_id, analysis_goal)
        
        return {"error": "Unknown action"}
    
    async def deep_research(
        self,
        scholar_id: str,
        analysis_goal: str = "全面分析学者的学术影响力",
    ) -> Dict[str, Any]:
        """
        执行深度研究
        
        Args:
            scholar_id: 学者 ID (Semantic Scholar ID)
            analysis_goal: 分析目标
            
        Returns:
            包含完整分析结果的字典
        """
        logger.info(f"🔬 开始深度研究: {scholar_id}")
        logger.info(f"📋 分析目标: {analysis_goal}")
        
        # 初始化研究状态
        state = ResearchState()
        
        try:
            # Step 1: 初始搜索
            logger.info("📡 Step 1: 初始搜索 - 获取学者信息和论文")
            await self._initial_search(state, scholar_id)
            
            if not state.scholar_info:
                logger.error(f"❌ 无法获取学者信息: {scholar_id}")
                return {"error": f"无法获取学者信息: {scholar_id}"}
            
            # Step 2: 初始总结
            logger.info("📝 Step 2: 初始总结")
            await self._initial_summary(state)
            
            # Step 3-5: 反思循环
            if self.reflection_search_node and self.reflection_summary_node:
                logger.info("🔄 Step 3-5: 反思循环")
                await self._reflection_loop(state, analysis_goal)
            
            # Step 6: 影响力评估
            logger.info("📊 Step 6: 影响力评估")
            await self._assess_influence(state)
            
            # Step 7: 生成报告
            logger.info("📑 Step 7: 生成最终报告")
            result = self._generate_final_result(state, analysis_goal)
            
            logger.success(f"✅ 深度研究完成: {state.scholar_info.get('name', scholar_id)}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 深度研究失败: {e}")
            return {
                "error": str(e),
                "partial_state": {
                    "scholar_info": state.scholar_info,
                    "papers_collected": len(state.collected_papers),
                    "completeness": state.completeness_score,
                },
            }
    
    async def _initial_search(self, state: ResearchState, scholar_id: str):
        """
        执行初始搜索
        
        Args:
            state: 研究状态
            scholar_id: 学者 ID
        """
        # 获取学者信息
        if self.semantic_scholar_agent:
            try:
                scholar_data = await self._fetch_scholar_info(scholar_id)
                state.scholar_info = scholar_data if scholar_data else {}
            except Exception as e:
                logger.warning(f"获取学者信息失败: {e}")
                state.scholar_info = {"authorId": scholar_id}
        
        # 获取学者论文
        if self.semantic_scholar_agent and state.scholar_info:
            try:
                papers = await self._fetch_scholar_papers(scholar_id)
                state.collected_papers = papers[:self.config.max_papers_per_search]
                logger.info(f"  → 获取到 {len(state.collected_papers)} 篇论文")
            except Exception as e:
                logger.warning(f"获取论文失败: {e}")
        
        # 记录搜索历史
        state.search_history.append({
            "type": "initial",
            "query": scholar_id,
            "timestamp": datetime.now().isoformat(),
            "results_count": len(state.collected_papers),
        })
    
    async def _fetch_scholar_info(self, scholar_id: str) -> Optional[Dict[str, Any]]:
        """获取学者信息"""
        if hasattr(self.semantic_scholar_agent, 'get_author'):
            return await self.semantic_scholar_agent.get_author(scholar_id)
        elif hasattr(self.semantic_scholar_agent, 'fetch_author_info'):
            return self.semantic_scholar_agent.fetch_author_info(scholar_id)
        return None
    
    async def _fetch_scholar_papers(self, scholar_id: str) -> List[Dict[str, Any]]:
        """获取学者论文"""
        if hasattr(self.semantic_scholar_agent, 'get_author_papers'):
            return await self.semantic_scholar_agent.get_author_papers(scholar_id)
        elif hasattr(self.semantic_scholar_agent, 'fetch_recent_papers'):
            return self.semantic_scholar_agent.fetch_recent_papers(scholar_id)
        return []
    
    async def _initial_summary(self, state: ResearchState):
        """
        生成初始总结
        
        Args:
            state: 研究状态
        """
        scholar = state.scholar_info
        papers = state.collected_papers
        
        # 生成基本总结
        summary_parts = []
        
        if scholar.get('name'):
            summary_parts.append(f"学者: {scholar['name']}")
        
        if scholar.get('affiliations'):
            affiliations = scholar['affiliations']
            if isinstance(affiliations, list):
                aff_str = ', '.join([a.get('name', str(a)) if isinstance(a, dict) else str(a) for a in affiliations[:3]])
            else:
                aff_str = str(affiliations)
            summary_parts.append(f"机构: {aff_str}")
        
        h_index = scholar.get('h_index') or scholar.get('hIndex', 0)
        citation_count = scholar.get('citation_count') or scholar.get('citationCount', 0)
        
        summary_parts.append(f"H-index: {h_index}, 总引用: {citation_count}")
        summary_parts.append(f"已获取 {len(papers)} 篇论文")
        
        if papers:
            recent_papers = sorted(papers, key=lambda p: p.get('year', 0), reverse=True)[:3]
            summary_parts.append("近期论文:")
            for p in recent_papers:
                summary_parts.append(f"  - {p.get('title', '未知标题')[:60]}... ({p.get('year', 'N/A')})")
        
        state.current_summary = "\n".join(summary_parts)
        state.completeness_score = 0.4  # 初始完整度
        
        logger.info(f"  → 初始完整度: {state.completeness_score:.1%}")
    
    async def _reflection_loop(self, state: ResearchState, analysis_goal: str):
        """
        执行反思循环
        
        Args:
            state: 研究状态
            analysis_goal: 分析目标
        """
        while state.reflection_count < self.config.max_reflections:
            logger.info(f"  🔄 反思 {state.reflection_count + 1}/{self.config.max_reflections}")
            
            # 执行反思搜索
            reflection_input = {
                "scholar_info": state.scholar_info,
                "collected_papers": state.collected_papers,
                "current_summary": state.current_summary,
                "reflection_count": state.reflection_count,
                "analysis_goal": analysis_goal,
            }
            
            reflection_result = self.reflection_search_node.run(reflection_input)
            
            if not reflection_result.get("should_continue"):
                logger.info(f"  → 反思循环结束: {reflection_result.get('reason', '信息已充分')}")
                break
            
            # 执行补充搜索
            search_query = reflection_result.get("search_query", "")
            search_type = reflection_result.get("search_type", "papers")
            
            logger.info(f"  → 补充搜索 ({search_type}): {search_query[:50]}...")
            
            new_results = await self._supplementary_search(search_query, search_type, state)
            
            # 更新总结
            if new_results:
                summary_input = {
                    "current_summary": state.current_summary,
                    "new_results": new_results,
                    "scholar_info": state.scholar_info,
                }
                
                summary_result = self.reflection_summary_node.run(summary_input)
                
                state.current_summary = summary_result.get("updated_summary", state.current_summary)
                state.key_findings.extend(summary_result.get("key_findings", []))
                state.completeness_score = summary_result.get("completeness_score", state.completeness_score)
                
                logger.info(f"  → 更新完整度: {state.completeness_score:.1%}")
            
            # 更新反思计数
            state.reflection_count = reflection_result.get("reflection_count", state.reflection_count + 1)
            
            # 记录搜索历史
            state.search_history.append({
                "type": search_type,
                "query": search_query,
                "timestamp": datetime.now().isoformat(),
                "results_count": len(new_results),
                "gaps": reflection_result.get("gaps_identified", []),
            })
            
            # 检查完整度
            if state.completeness_score >= self.config.min_completeness_score:
                logger.info(f"  → 达到目标完整度 ({self.config.min_completeness_score:.0%})，停止反思")
                break
    
    async def _supplementary_search(
        self,
        query: str,
        search_type: str,
        state: ResearchState,
    ) -> List[Dict[str, Any]]:
        """
        执行补充搜索
        
        Args:
            query: 搜索查询
            search_type: 搜索类型
            state: 研究状态
            
        Returns:
            新搜索结果
        """
        results = []
        
        try:
            if search_type == "papers" and self.semantic_scholar_agent:
                # 搜索更多论文
                if hasattr(self.semantic_scholar_agent, 'search_papers'):
                    new_papers = await self.semantic_scholar_agent.search_papers(query)
                    # 去重
                    existing_ids = {p.get('paperId', p.get('paper_id')) for p in state.collected_papers}
                    for paper in new_papers:
                        paper_id = paper.get('paperId', paper.get('paper_id'))
                        if paper_id and paper_id not in existing_ids:
                            state.collected_papers.append(paper)
                            results.append(paper)
                            existing_ids.add(paper_id)
            
            elif search_type == "code" and self.config.enable_code_analysis:
                # 搜索代码仓库 (未来扩展)
                logger.info("  → 代码搜索功能待实现")
            
            elif search_type == "citations" and self.config.enable_citation_analysis:
                # 搜索引用关系 (未来扩展)
                logger.info("  → 引用分析功能待实现")
            
            elif search_type == "collaborators":
                # 搜索合作者 (未来扩展)
                logger.info("  → 合作者分析功能待实现")
                
        except Exception as e:
            logger.warning(f"  ⚠ 补充搜索失败: {e}")
        
        return results
    
    async def _assess_influence(self, state: ResearchState):
        """
        评估影响力
        
        Args:
            state: 研究状态
        """
        if not self.influence_node:
            logger.warning("  ⚠ 影响力计算节点未配置")
            return
        
        try:
            influence_input = {
                "scholars": [state.scholar_info],
                "papers": state.collected_papers,
            }
            
            influence_result = self.influence_node.run(influence_input)
            results = influence_result.get("influence_results", [])
            
            if results:
                state.influence_result = results[0]
                logger.info(f"  → 影响力评分: {state.influence_result.get('total_score', 0):.2f}")
        
        except Exception as e:
            logger.warning(f"  ⚠ 影响力评估失败: {e}")
    
    def _generate_final_result(
        self,
        state: ResearchState,
        analysis_goal: str,
    ) -> Dict[str, Any]:
        """
        生成最终结果
        
        Args:
            state: 研究状态
            analysis_goal: 分析目标
            
        Returns:
            完整分析结果
        """
        scholar = state.scholar_info
        
        return {
            "success": True,
            "scholar": {
                "id": scholar.get('authorId', scholar.get('scholar_id')),
                "name": scholar.get('name', '未知'),
                "affiliations": scholar.get('affiliations', []),
                "h_index": scholar.get('hIndex', scholar.get('h_index', 0)),
                "citation_count": scholar.get('citationCount', scholar.get('citation_count', 0)),
            },
            "analysis": {
                "summary": state.current_summary,
                "key_findings": list(set(state.key_findings)),  # 去重
                "completeness_score": state.completeness_score,
            },
            "papers": {
                "total_collected": len(state.collected_papers),
                "sample": state.collected_papers[:5],  # 返回前5篇作为样本
            },
            "influence": state.influence_result or {},
            "metadata": {
                "analysis_goal": analysis_goal,
                "reflection_count": state.reflection_count,
                "search_history": state.search_history,
                "timestamp": datetime.now().isoformat(),
            },
        }
    
    def research_sync(
        self,
        scholar_id: str,
        analysis_goal: str = "全面分析学者的学术影响力",
    ) -> Dict[str, Any]:
        """
        同步版本的深度研究 (用于非异步环境)
        
        Args:
            scholar_id: 学者 ID
            analysis_goal: 分析目标
            
        Returns:
            分析结果
        """
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.deep_research(scholar_id, analysis_goal))


# 便捷函数
def create_deep_research_agent(
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    **kwargs,
) -> DeepResearchAgent:
    """
    创建 Deep Research Agent 的便捷函数
    
    Args:
        api_key: OpenAI API Key
        model: 模型名称
        **kwargs: 其他配置参数
        
    Returns:
        配置好的 DeepResearchAgent 实例
    """
    llm_client = None
    if api_key:
        from core.llm_client import LLMClient
        llm_client = LLMClient(api_key=api_key, model=model)
    
    config = DeepResearchConfig(
        max_reflections=kwargs.get('max_reflections', 3),
        min_completeness_score=kwargs.get('min_completeness_score', 0.8),
        max_papers_per_search=kwargs.get('max_papers_per_search', 20),
    )
    
    return DeepResearchAgent(
        llm_client=llm_client,
        config=config,
    )
