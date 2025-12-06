# securipaperbot/agents/research_agent.py

from typing import Dict, List, Any, Optional
import re
from .base_agent import BaseAgent


class ResearchAgent(BaseAgent):
    """
    单篇论文增强 Agent：摘要总结、代码链接提取、关键贡献生成
    （会议抓取已拆分至 ConferenceResearchAgent）
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    async def process(self, *args, **kwargs) -> Dict[str, Any]:
        """仅支持单篇论文分析"""
        if "paper_title" in kwargs:
            return await self._analyze_single_paper(
                title=kwargs.get("paper_title"),
                paper_id=kwargs.get("paper_id"),
                abstract=kwargs.get("abstract")
            )
        raise ValueError("Invalid arguments for ResearchAgent.process (expect paper_title/paper_id/abstract)")

    async def _analyze_single_paper(self, title: str, paper_id: str, abstract: Optional[str]) -> Dict[str, Any]:
        """分析单篇论文"""
        result = {
            "paper_id": paper_id,
            "title": title,
            "executive_summary": None,
            "github_url": None,
            "key_contributions": []
        }
        
        if abstract:
            # 1. 生成摘要总结
            result["executive_summary"] = await self.summarize_abstract(abstract)
            
            # 2. 尝试从摘要中提取 GitHub 链接 (简单的正则)
            github_pattern = r'https?://github\.com/[\w-]+/[\w-]+'
            links = re.findall(github_pattern, abstract)
            if links:
                result["github_url"] = links[0]
        
        # 3. 如果启用了 LLM，可以生成 key_contributions
        if self.client and abstract:
             try:
                 prompt = f"Based on the abstract below, list 3 key contributions of the paper '{title}'.\n\nAbstract: {abstract}"
                 response = await self.ask_claude(prompt)
                 # 简单的按行分割
                 contributions = [line.strip('- *') for line in response.split('\n') if line.strip()]
                 result["key_contributions"] = contributions[:3]
             except Exception as e:
                 self.log_error(e, {"context": "generate_key_contributions"})
             
        return result
    async def summarize_abstract(self, abstract: str) -> str:
        """使用Claude总结摘要"""
        prompt = f"Please summarize the following academic paper abstract in 2-3 sentences:\n\n{abstract}"
        return await self.ask_claude(prompt, system="You are a helpful research assistant.")
    async def _extract_github_links(self, pdf_path: str) -> List[str]:
        """从PDF中提取GitHub链接"""
        github_pattern = r'https?://github\.com/[\w-]+/[\w-]+'
        links = []

        try:
            # 使用pdfplumber提取文本
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                text = '\n'.join(page.extract_text() for page in pdf.pages)
                links = re.findall(github_pattern, text)

            return list(set(links))  # 去重

        except Exception as e:
            self.log_error(e, {'pdf_path': pdf_path})
            return []

    def validate_config(self) -> bool:
        """验证配置是否完整"""
        required_keys = ['acm_base_url', 'ieee_base_url', 'download_path']
        return all(key in self.config for key in required_keys)