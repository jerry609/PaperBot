from typing import Dict, List, Any, Optional
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import re
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
        self.max_concurrency = int((config or {}).get("max_concurrency", 3))
        self.rate_limit_per_sec = int((config or {}).get("rate_limit_per_sec", 2))
        self.retry_times = int((config or {}).get("retry_times", 2))

    async def process(self, conference: str, year: str) -> Dict[str, Any]:
        if conference not in self.supported_conferences:
            raise ValueError(f"Unsupported conference: {conference}")
        processor = self.supported_conferences[conference]
        papers = await processor(year)
        total = len(papers)
        downloaded = sum(1 for p in papers if p.get("local_path"))
        with_code = sum(1 for p in papers if p.get("github_links"))
        print(f"[ConferenceResearchAgent] {conference.upper()} {year}: total={total}, downloaded={downloaded}, with_code={with_code}")
        return {
            'conference': conference,
            'year': year,
            'papers': papers
        }

    async def _process_ccs(self, year: str) -> List[Dict[str, Any]]:
        base_url = self.config.get('acm_base_url')
        return await self._process_generic(base_url, year, self._fetch_ccs_papers)

    async def _process_sp(self, year: str) -> List[Dict[str, Any]]:
        papers_list = await self.downloader.get_conference_papers('sp', year)
        return await self._download_and_extract(papers_list)

    async def _process_ndss(self, year: str) -> List[Dict[str, Any]]:
        papers_list = await self.downloader.get_conference_papers('ndss', year)
        return await self._download_and_extract(papers_list)

    async def _process_usenix(self, year: str) -> List[Dict[str, Any]]:
        papers_list = await self.downloader.get_conference_papers('usenix', year)
        return await self._download_and_extract(papers_list)

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

    async def _download_and_extract(self, papers_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        semaphore = asyncio.Semaphore(self.max_concurrency)
        results: List[Dict[str, Any]] = []

        async def worker(paper: Dict[str, Any]):
            if not paper.get("url"):
                results.append(paper)
                return
            for attempt in range(self.retry_times + 1):
                try:
                    async with semaphore:
                        if self.rate_limit_per_sec > 0:
                            await asyncio.sleep(1 / self.rate_limit_per_sec)
                        download_info = await self.downloader.download_paper(
                            paper['url'],
                            paper['title'],
                            paper_index=0,
                            total_papers=len(papers_list)
                        )
                    paper['local_path'] = download_info.get('path')
                    links = await self._extract_github_links(download_info.get('path'))
                    if not links:
                        links = await self._extract_github_from_html(paper.get('url'))
                    paper['github_links'] = links
                    results.append(paper)
                    return
                except Exception:
                    if attempt >= self.retry_times:
                        results.append(paper)
                    continue

        await asyncio.gather(*(worker(p) for p in papers_list))
        return results

    async def _process_generic(self, base_url: str, year: str, fetcher):
        papers_list = []
        async with aiohttp.ClientSession() as session:
            papers_list = await fetcher(session, base_url, year)
        return await self._download_and_extract(papers_list)

    async def _extract_github_links(self, pdf_path: Optional[str]) -> List[str]:
        """从PDF中提取 GitHub 链接"""
        if not pdf_path:
            return []
        github_pattern = r'https?://github\.com/[\w-]+/[\w-]+'
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                text = '\n'.join(page.extract_text() or "" for page in pdf.pages)
            links = re.findall(github_pattern, text)
            return list(set(links))
        except Exception:
            return []

    async def _extract_github_from_html(self, url: Optional[str]) -> List[str]:
        if not url:
            return []
        github_pattern = r'https?://github\\.com/[\\w-]+/[\\w-]+'
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=15) as resp:
                    html = await resp.text()
            links = re.findall(github_pattern, html)
            return list(set(links))
        except Exception:
            return []

