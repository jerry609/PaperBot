"""
反思搜索节点

实现 Deep Research 的反思循环模式
根据当前分析状态识别信息空白并生成补充搜索
"""

from typing import Any, Dict, List, Optional
import logging

from paperbot.repro.nodes.base_node import LLMNode
from paperbot.agents.prompts.scholar_prompts import (
    SYSTEM_PROMPT_REFLECTION_SEARCH,
    output_schema_reflection_search,
)
from paperbot.utils.json_parser import parse_json

logger = logging.getLogger(__name__)


class ReflectionSearchNode(LLMNode):
    """
    反思搜索节点
    
    根据当前分析状态，识别信息空白并生成补充搜索查询
    """
    
    def __init__(
        self,
        llm_client: Any,
        max_reflections: int = 3,
        node_name: str = "ReflectionSearchNode"
    ):
        """
        初始化节点
        
        Args:
            llm_client: LLM 客户端
            max_reflections: 最大反思次数
            node_name: 节点名称
        """
        super().__init__(
            node_name=node_name,
            llm_client=llm_client,
        )
        self.system_prompt = SYSTEM_PROMPT_REFLECTION_SEARCH
        self.max_reflections = max_reflections
    
    async def _execute(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        执行反思分析
        
        Args:
            input_data: 包含当前分析状态的字典
                - scholar_info: 学者信息
                - collected_papers: 已收集的论文
                - current_summary: 当前分析总结
                - reflection_count: 当前反思次数
                - analysis_goal: 分析目标
            
        Returns:
            包含补充搜索查询的字典
        """
        reflection_count = input_data.get("reflection_count", 0)
        
        # 检查是否超过最大反思次数
        if reflection_count >= self.max_reflections:
            logger.info(f"已达到最大反思次数 ({self.max_reflections})，停止反思")
            return {
                "should_continue": False,
                "reason": f"已达到最大反思次数 ({self.max_reflections})",
            }
        
        # 构建反思提示
        user_prompt = self._build_reflection_prompt(input_data)
        
        logger.info(f"执行第 {reflection_count + 1}/{self.max_reflections} 次反思分析")
        
        try:
            # 调用 LLM
            response = self.llm_client.invoke(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            
            # 解析响应
            result = parse_json(response)
            
            # 检查是否有有价值的补充搜索
            if self._is_valuable_search(result):
                logger.info(f"  → 补充搜索: {result.get('search_query', '')}")
                logger.info(f"  → 搜索类型: {result.get('search_type', '')}")
                logger.info(f"  → 理由: {result.get('reasoning', '')[:100]}...")
                
                return {
                    "should_continue": True,
                    "search_query": result.get("search_query", ""),
                    "search_type": result.get("search_type", "papers"),
                    "reasoning": result.get("reasoning", ""),
                    "gaps_identified": result.get("gaps_identified", []),
                    "reflection_count": reflection_count + 1,
                }
            else:
                logger.info("  → 信息已充分，无需继续搜索")
                return {
                    "should_continue": False,
                    "reason": "信息已充分",
                }
                
        except Exception as e:
            logger.error(f"反思分析失败: {e}")
            return {
                "should_continue": False,
                "error": str(e),
            }
    
    def _build_reflection_prompt(self, input_data: Dict[str, Any]) -> str:
        """
        构建反思提示
        
        Args:
            input_data: 输入数据
            
        Returns:
            用户提示字符串
        """
        scholar_info = input_data.get("scholar_info", {})
        collected_papers = input_data.get("collected_papers", [])
        current_summary = input_data.get("current_summary", "")
        analysis_goal = input_data.get("analysis_goal", "全面分析学者的学术影响力")
        
        # 格式化论文摘要
        paper_titles = [p.get("title", "") for p in collected_papers[:10]]
        papers_summary = "\n".join(f"  - {t}" for t in paper_titles) if paper_titles else "无"
        
        prompt = f"""
## 分析目标
{analysis_goal}

## 学者信息
- 姓名: {scholar_info.get('name', '未知')}
- 机构: {scholar_info.get('affiliations', '未知')}
- H-index: {scholar_info.get('h_index', '未知')}
- 总引用: {scholar_info.get('citation_count', '未知')}

## 已收集论文 ({len(collected_papers)} 篇)
{papers_summary}
{"... 及更多" if len(collected_papers) > 10 else ""}

## 当前分析状态
{current_summary if current_summary else "尚未开始分析"}

## 任务
请分析当前收集的信息是否充分，识别需要补充搜索的内容。
如果信息已经足够完成分析目标，请设置 search_query 为空字符串。
"""
        return prompt
    
    def _is_valuable_search(self, result: Dict[str, Any]) -> bool:
        """
        判断搜索是否有价值
        
        Args:
            result: LLM 返回的结果
            
        Returns:
            是否有价值
        """
        search_query = result.get("search_query", "").strip()
        gaps = result.get("gaps_identified", [])
        
        # 如果没有搜索查询或空白，认为信息已充分
        if not search_query:
            return False
        
        # 如果没有识别到信息空白，也认为信息已充分
        if not gaps:
            return False
        
        return True
    
    def validate_input(self, input_data: Any) -> bool:
        """验证输入"""
        if not isinstance(input_data, dict):
            return False
        # 至少需要学者信息或分析目标
        return bool(input_data.get("scholar_info") or input_data.get("analysis_goal"))


class ReflectionSummaryNode(LLMNode):
    """
    反思总结节点
    
    根据新收集的信息更新分析总结
    """
    
    def __init__(
        self,
        llm_client: Any,
        node_name: str = "ReflectionSummaryNode"
    ):
        """
        初始化节点
        
        Args:
            llm_client: LLM 客户端
            node_name: 节点名称
        """
        super().__init__(
            node_name=node_name,
            llm_client=llm_client,
        )
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """构建系统提示"""
        return """
你是一位专业的学术情报分析师。你的任务是整合新收集的信息，更新对学者的分析总结。

**输入**: 
- 学者基本信息
- 已有分析总结
- 新搜索结果

**任务**:
1. 整合新旧信息
2. 更新关键发现
3. 补充遗漏的内容
4. 保持总结简洁有条理

**输出格式**: 请返回JSON格式：
{
    "updated_summary": "更新后的分析总结",
    "key_findings": ["发现1", "发现2", ...],
    "completeness_score": 0.85  // 0-1，信息完整度评估
}

只返回JSON对象，不要有额外解释。
"""
    
    async def _execute(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        更新分析总结
        
        Args:
            input_data: 包含原有总结和新信息的字典
            
        Returns:
            包含更新后总结的字典
        """
        current_summary = input_data.get("current_summary", "")
        new_results = input_data.get("new_results", [])
        scholar_info = input_data.get("scholar_info", {})
        
        logger.info("更新分析总结...")
        
        # 构建用户提示
        user_prompt = f"""
## 学者信息
{scholar_info.get('name', '未知')} - {scholar_info.get('affiliations', '未知')}

## 已有总结
{current_summary if current_summary else "尚无总结"}

## 新收集的信息
{self._format_new_results(new_results)}

请整合以上信息，输出更新后的分析总结。
"""
        
        try:
            response = self.llm_client.invoke(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            
            result = parse_json(response)
            
            logger.info(f"  → 完整度评分: {result.get('completeness_score', 0):.2f}")
            
            return {
                "updated_summary": result.get("updated_summary", current_summary),
                "key_findings": result.get("key_findings", []),
                "completeness_score": result.get("completeness_score", 0),
            }
            
        except Exception as e:
            logger.error(f"更新总结失败: {e}")
            return {
                "updated_summary": current_summary,
                "key_findings": [],
                "completeness_score": 0,
                "error": str(e),
            }
    
    def _format_new_results(self, results: List[Any]) -> str:
        """格式化新结果"""
        if not results:
            return "无新信息"
        
        formatted = []
        for i, r in enumerate(results[:5], 1):  # 最多显示5条
            if isinstance(r, dict):
                title = r.get("title", r.get("name", f"结果 {i}"))
                formatted.append(f"- {title}")
            else:
                formatted.append(f"- {str(r)[:100]}")
        
        if len(results) > 5:
            formatted.append(f"... 及其他 {len(results) - 5} 条结果")
        
        return "\n".join(formatted)
    
    def validate_input(self, input_data: Any) -> bool:
        """验证输入"""
        return isinstance(input_data, dict)
