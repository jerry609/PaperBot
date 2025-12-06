from typing import Dict, List, Any, Optional
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from .base_agent import BaseAgent
from utils.downloader import PaperDownloader


class ConferenceResearchAgent(BaseAgent):
    """
    会议抓取/下载 Agent
    负责按会议+年份获取论文列表、下载 PDF，并尝试提取 GitHub 链接。
    """

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
        if conference not in self.supported_conferences:
            raise ValueError(f"Unsupported conference: {conference}")
        processor = self.supported_conferences[conference]
        papers = await processor(year)
        return {
            'conference': conference,
            'year': year,
            'papers': papers
        }

    async def _process_ccs(self, year: str) -> List[Dict[str, Any]]:
        base_url = self.config.get('acm_base_url')
        papers = []

        async with aiohttp.ClientSession() as session:
            papers_list = await self._fetch_ccs_papers(session, base_url, year)
            download_tasks = [
                self.downloader.download_paper(p['url'], p['title'])
                for p in papers_list
            ]
            downloaded_papers = await asyncio.gather(*download_tasks)
            for paper, download_info in zip(papers_list, downloaded_papers):
                paper['local_path'] = download_info.get('path')
                paper['github_links'] = await self._extract_github_links(download_info.get('path'))
                papers.append(paper)
        return papers

    async def _process_sp(self, year: str) -> List[Dict[str, Any]]:
        papers = []
        papers_list = await self.downloader.get_conference_papers('sp', year)
        download_tasks = [
            self.downloader.download_paper(
                paper['url'],
                paper['title'],
                paper_index=idx,
                total_papers=len([p for p in papers_list if p.get('url')])
            )
            for idx, paper in enumerate(papers_list) if paper.get('url')
        ]
        downloaded_papers = await asyncio.gather(*download_tasks)
        for paper, download_info in zip(papers_list, downloaded_papers):
            if download_info.get('success'):
                paper['local_path'] = download_info['path']
                paper['github_links'] = await self._extract_github_links(download_info['path'])
            papers.append(paper)
        return papers

    async def _process_ndss(self, year: str) -> List[Dict[str, Any]]:
        papers = []
        papers_list = await self.downloader.get_conference_papers('ndss', year)
        download_tasks = [
            self.downloader.download_paper(
                paper['url'],
                paper['title'],
                paper_index=idx,
                total_papers=len([p for p in papers_list if p.get('url')])
            )
            for idx, paper in enumerate(papers_list) if paper.get('url')
        ]
        downloaded_papers = await asyncio.gather(*download_tasks)
        for paper, download_info in zip(papers_list, downloaded_papers):
            if download_info.get('success'):
                paper['local_path'] = download_info['path']
                paper['github_links'] = await self._extract_github_links(download_info['path'])
            papers.append(paper)
        return papers

    async def _process_usenix(self, year: str) -> List[Dict[str, Any]]:
        papers = []
        papers_list = await self.downloader.get_conference_papers('usenix', year)
        download_tasks = [
            self.downloader.download_paper(
                paper['url'],
                paper['title'],
                paper_index=idx,
                total_papers=len([p for p in papers_list if p.get('url')])
            )
            for idx, paper in enumerate(papers_list) if paper.get('url')
        ]
        downloaded_papers = await asyncio.gather(*download_tasks)
        for paper, download_info in zip(papers_list, downloaded_papers):
            if download_info.get('success'):
                paper['local_path'] = download_info['path']
                paper['github_links'] = await self._extract_github_links(download_info['path'])
            papers.append(paper)
        return papers

    async def _fetch_ccs_papers(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        year: str
    ) -> List[Dict[str, Any]]:
        url = f"{base_url}/ccs{year}"
        papers = []
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

    async def _extract_github_links(self, pdf_path: Optional[str]) -> List[str]:
        """从PDF中提取 GitHub 链接"""
        if not pdf_path:
            return []
        github_pattern = r'https?://github\.com/[\w-]+/[\w-]+'
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                text = '\n'.join(page.extract_text() or "" for page in pdf.pages)
            return list(set(re.findall(github_pattern, text)))
        except Exception:
            return []

