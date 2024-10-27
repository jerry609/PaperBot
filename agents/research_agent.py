# securipaperbot/agents/research_agent.py

from typing import Dict, List, Any, Optional
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import re
from .base_agent import BaseAgent
from ..utils.downloader import PaperDownloader


class ResearchAgent(BaseAgent):
    """负责论文下载和初步分析的代理"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.downloader = PaperDownloader(config)
        self.supported_conferences = {
            'ccs': self._process_ccs,
            'sp': self._process_sp,
            'ndss': self._process_ndss,
            'usenix': self._process_usenix
        }

    async def process(self, conference: str, year: str) -> Dict[str, Any]:
        """处理指定会议和年份的论文"""
        try:
            if conference not in self.supported_conferences:
                raise ValueError(f"Unsupported conference: {conference}")

            processor = self.supported_conferences[conference]
            papers = await processor(year)

            return {
                'conference': conference,
                'year': year,
                'papers': papers
            }

        except Exception as e:
            self.log_error(e, {'conference': conference, 'year': year})
            raise

    async def _process_ccs(self, year: str) -> List[Dict[str, Any]]:
        """处理CCS会议论文"""
        base_url = self.config.get('acm_base_url')
        papers = []

        try:
            # 获取论文列表
            async with aiohttp.ClientSession() as session:
                papers_list = await self._fetch_ccs_papers(session, base_url, year)

                # 并行下载论文
                download_tasks = [
                    self.downloader.download_paper(paper['url'], paper['title'])
                    for paper in papers_list
                ]
                downloaded_papers = await asyncio.gather(*download_tasks)

                # 提取代码链接
                for paper, download_info in zip(papers_list, downloaded_papers):
                    paper['local_path'] = download_info['path']
                    paper['github_links'] = await self._extract_github_links(
                        download_info['path']
                    )
                    papers.append(paper)

            return papers

        except Exception as e:
            self.log_error(e, {'conference': 'ccs', 'year': year})
            raise

    async def _process_sp(self, year: str) -> List[Dict[str, Any]]:
        """处理S&P会议论文"""
        # 实现S&P论文处理逻辑
        pass

    async def _process_ndss(self, year: str) -> List[Dict[str, Any]]:
        """处理NDSS会议论文"""
        # 实现NDSS论文处理逻辑
        pass

    async def _process_usenix(self, year: str) -> List[Dict[str, Any]]:
        """处理USENIX会议论文"""
        # 实现USENIX论文处理逻辑
        pass

    async def _fetch_ccs_papers(
            self,
            session: aiohttp.ClientSession,
            base_url: str,
            year: str
    ) -> List[Dict[str, Any]]:
        """获取CCS论文列表"""
        url = f"{base_url}/ccs{year}"
        papers = []

        try:
            async with session.get(url) as response:
                text = await response.text()
                soup = BeautifulSoup(text, 'html.parser')

                for paper in soup.find_all('div', class_='paper'):
                    papers.append({
                        'title': paper.find('h2').text.strip(),
                        'authors': [a.text.strip() for a in paper.find_all('a', class_='author')],
                        'url': paper.find('a', class_='pdf')['href'],
                        'abstract': paper.find('div', class_='abstract').text.strip()
                    })

            return papers

        except Exception as e:
            self.log_error(e, {'url': url})
            raise

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