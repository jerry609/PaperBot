from typing import Optional, Dict, Any, List
from bs4 import BeautifulSoup
from curl_cffi.requests import AsyncSession
import re
import logging
import json
import asyncio
from pathlib import Path

def setup_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

class ConferenceHelpers:
    def __init__(self):
        self.logger = setup_logger(__name__)

    async def get_sp_papers(self, session: AsyncSession, base_url: str, year: str) -> List[Dict[str, Any]]:
        """获取 IEEE S&P 论文列表"""
        papers = []
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
            }
            
            async with session.get(base_url, headers=headers) as response:
                if response.status == 200:
                    soup = BeautifulSoup(await response.text(), 'html.parser')
                    paper_items = soup.find_all('div', class_='article-list__item')

                    for item in paper_items:
                        title_elem = item.find('h3', class_='article-list__title')
                        if not title_elem:
                            continue

                        title = title_elem.text.strip()
                        pdf_url = await self._get_ieee_pdf_url(item)
                        
                        if pdf_url:
                            papers.append({
                                'title': title,
                                'url': pdf_url,
                                'conference': 'SP',
                                'year': year
                            })

                else:
                    raise Exception(f"Failed to fetch SP {year} papers list")

            return papers

        except Exception as e:
            self.logger.error(f"解析 SP {year} 论文列表失败: {e}")
            raise

    async def get_ndss_papers(self, session: AsyncSession, base_url: str, year: str) -> List[Dict[str, Any]]:
        """获取 NDSS 论文列表"""
        papers = []
        try:
            url = f"{base_url}ndss{year}/accepted-papers"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = await session.get(url, headers=headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                paper_items = soup.find_all(['div', 'article'], class_=['paper-item', 'accepted-paper'])

                for item in paper_items:
                    title_elem = item.find(['h2', 'h3', 'h4'], class_=['title', 'paper-title'])
                    if not title_elem:
                        continue

                        title = title_elem.text.strip()
                        pdf_link = item.find('a', href=lambda x: x and x.endswith('.pdf'))
                        
                        if pdf_link and 'href' in pdf_link.attrs:
                            pdf_url = pdf_link['href']
                            if not pdf_url.startswith('http'):
                                pdf_url = f"https://www.ndss-symposium.org{pdf_url}"
                            papers.append({
                                'title': title,
                                'url': pdf_url,
                                'conference': 'NDSS',
                                'year': year
                            })

            else:
                raise Exception(f"Failed to fetch NDSS {year} papers list")

            return papers

        except Exception as e:
            self.logger.error(f"解析 NDSS {year} 论文列表失败: {e}")
            raise

    async def get_usenix_papers(self, session: AsyncSession, base_url: str, year: str) -> List[Dict[str, Any]]:
        """获取 USENIX Security 论文列表"""
        papers = []
        full_year = f"20{year}" if len(year) == 2 else year
        url = f"{base_url}{full_year}/technical-sessions"
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    soup = BeautifulSoup(await response.text(), 'html.parser')
                    paper_nodes = soup.find_all(['article', 'div'], class_=['node-paper', 'paper-item'])

                    for node in paper_nodes:
                        title_elem = node.find(['h2', 'div'], class_=['node-title', 'field-title'])
                        if not title_elem:
                            continue

                        title = title_elem.text.strip()
                        pdf_url = await self._get_usenix_pdf_url(node)
                        
                        if pdf_url:
                            papers.append({
                                'title': title,
                                'url': pdf_url,
                                'conference': 'USENIX',
                                'year': year
                            })

                else:
                    raise Exception(f"Failed to fetch USENIX {year} papers list")

            return papers

        except Exception as e:
            self.logger.error(f"解析 USENIX {year} 论文列表失败: {e}")
            raise

    async def _get_ieee_pdf_url(self, paper_element) -> Optional[str]:
        """从IEEE页面元素中提取PDF URL"""
        try:
            pdf_link = paper_element.find('a', href=re.compile(r'.*\.pdf'))
            if pdf_link:
                return pdf_link['href']
            
            article_link = paper_element.find('a', href=re.compile(r'/document/'))
            if article_link:
                doc_id = re.search(r'/document/(\d+)', article_link['href'])
                if doc_id:
                    return f"https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber={doc_id.group(1)}"
            
            return None
        except Exception as e:
            self.logger.error(f"Error extracting IEEE PDF URL: {str(e)}")
            return None

    async def _get_usenix_pdf_url(self, node) -> Optional[str]:
        """从USENIX论文节点获取PDF URL"""
        try:
            pdf_link = node.find('a', href=re.compile(r'\.pdf$'))
            if pdf_link and 'href' in pdf_link.attrs:
                url = pdf_link['href']
                if not url.startswith('http'):
                    url = f"https://www.usenix.org{url}"
                return url
            
            return None
        except Exception as e:
            self.logger.error(f"Error extracting USENIX PDF URL: {str(e)}")
            return None
