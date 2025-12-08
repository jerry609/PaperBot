import asyncio
from curl_cffi.requests import AsyncSession
from bs4 import BeautifulSoup
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

def setup_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

class ConferenceParsers:
    def __init__(self):
        self.logger = setup_logger(__name__)

    async def parse_ndss_papers(self, base_url: str, year: str, session: AsyncSession) -> List[Dict[str, Any]]:
        """解析NDSS论文列表 - 优化版本带进度显示"""
        papers = []
        full_year = f"20{year}" if len(year) == 2 else year
        url = f"{base_url}ndss{full_year}/accepted-papers/"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        print(f"🌐 访问 NDSS {year} 会议页面...")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with session.get(url, headers=headers) as response:
                    print(f"⚡ 尝试 {attempt + 1}/{max_retries}: HTTP {response.status}")
                    
                    if response.status == 200:
                        print(f"📝 正在解析页面内容...")
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # 查找NDSS论文容器
                        paper_containers = soup.find_all('div', class_='tag-box rel-paper')
                        print(f"📚 找到 {len(paper_containers)} 个论文容器")
                        
                        if not paper_containers:
                            print(f"⚠️  未找到论文容器，尝试其他选择器...")
                            # 尝试其他可能的选择器
                            paper_containers = soup.find_all('div', class_='paper') or soup.find_all('article')
                            print(f"🔄 备用选择器找到 {len(paper_containers)} 个容器")
                        
                        # 处理论文容器并显示简单进度
                        for idx, container in enumerate(paper_containers):
                            if idx % 5 == 0 or idx == len(paper_containers) - 1:  # 每5个显示一次进度
                                progress = (idx + 1) / len(paper_containers) * 100
                                print(f"� 解析进度: {progress:.1f}% ({idx+1}/{len(paper_containers)})")
                            
                            try:
                                # 提取标题 - 尝试多种选择器
                                title_elem = (container.find('h3', class_='blog-post-title') or 
                                            container.find('h3') or 
                                            container.find('h2') or 
                                            container.find('h1'))
                                
                                if not title_elem:
                                    continue
                                
                                title = title_elem.get_text().strip()
                                
                                # 显示找到的论文标题
                                print(f"📄 [{idx+1}/{len(paper_containers)}] 找到论文: {title[:70]}{'...' if len(title) > 70 else ''}")
                                
                                # 提取作者信息
                                author_elem = container.find('p')
                                authors_text = author_elem.get_text().strip() if author_elem else ''
                                authors = [author.strip() for author in authors_text.split(',')] if authors_text else []
                                
                                # 提取详情页链接
                                detail_link = (container.find('a', class_='paper-link-abs') or 
                                            container.find('a', href=True))
                                detail_url = detail_link.get('href') if detail_link else ''
                                
                                # 提取PDF链接
                                pdf_url = ''
                                if detail_url:
                                    # 将详情页URL转换为绝对URL
                                    if not detail_url.startswith('http'):
                                        if detail_url.startswith('//'):
                                            detail_url = f'https:{detail_url}'
                                        elif detail_url.startswith('/'):
                                            detail_url = f'https://www.ndss-symposium.org{detail_url}'
                                        else:
                                            detail_url = f'https://www.ndss-symposium.org/{detail_url}'
                                    pdf_url = await self._get_ndss_pdf_from_detail_page(session, detail_url)
                                
                                paper_info = {
                                    'title': title,
                                    'authors': authors,
                                    'abstract': '',
                                    'url': pdf_url,
                                    'detail_url': detail_url,
                                    'doi': ''
                                }
                                
                                if title and len(title) > 10:
                                    papers.append(paper_info)
                            except Exception as e:
                                self.logger.warning(f"Error parsing paper container {idx}: {str(e)}")
                                continue
                                
                        print(f"\n✅ 成功解析 {len(papers)} 篇论文")
                        return papers

            except Exception as e:
                print(f"❌ 尝试 {attempt + 1} 失败: {str(e)}")
                if attempt < max_retries - 1:
                    delay = 3 * (attempt + 1)
                    print(f"⏳ 等待 {delay} 秒后重试...")
                    await asyncio.sleep(delay)
                else:
                    return []
        
        print("❌ 所有重试均失败")
        return []

    async def _get_ndss_pdf_from_detail_page(self, session: AsyncSession, detail_url: str) -> str:
        """从NDSS论文详情页提取PDF链接 - 优化版本"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            async with session.get(detail_url, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # 尝试多种PDF选择器
                    pdf_selectors = [
                        ('a', {'href': lambda x: x and x.endswith('.pdf')}),
                        ('a', {'href': lambda x: x and 'paper' in x.lower() and '.pdf' in x.lower()}),
                        ('a', {'class': 'file-pdf'}),
                        ('a', {'class': 'download-pdf'}),
                        ('a', {'title': lambda x: x and 'pdf' in x.lower()})
                    ]
                    
                    for tag, attrs in pdf_selectors:
                        pdf_link = soup.find(tag, attrs)
                        if pdf_link and 'href' in pdf_link.attrs:
                            pdf_url = pdf_link['href']
                            # 处理相对URL
                            if not pdf_url.startswith('http'):
                                if pdf_url.startswith('//'):
                                    pdf_url = f'https:{pdf_url}'
                                elif pdf_url.startswith('/'):
                                    pdf_url = f'https://www.ndss-symposium.org{pdf_url}'
                                else:
                                    pdf_url = f'https://www.ndss-symposium.org/{pdf_url}'
                            return pdf_url
                            
            return ''
            
        except Exception as e:
            self.logger.warning(f"Error getting PDF from detail page: {str(e)}")
            return ''

    async def parse_usenix_papers(self, base_url: str, year: str, session: AsyncSession) -> List[Dict[str, Any]]:
        """解析USENIX Security论文列表"""
        papers = []
        url = f"https://www.usenix.org/conference/usenixsecurity{year}/technical-sessions"
        
        print(f"🌐 正在解析 USENIX Security {year} 论文列表...")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            response = await session.get(url, headers=headers)
            if response.status_code == 200:
                print(f"✅ 页面访问成功，开始解析...")
                html = response.text
                soup = BeautifulSoup(html, 'html.parser')
                    
                # 使用多种选择器查找论文节点
                paper_nodes = soup.find_all(['article', 'div'], class_=['node-paper', 'paper-item'])
                
                print(f"📚 找到 {len(paper_nodes)} 个论文节点")
                
                if not paper_nodes:
                    print("⚠️ 未找到论文节点，尝试备用选择器...")
                    paper_nodes = soup.find_all(['div', 'article'], class_=['paper', 'technical-paper'])
                
                for idx, node in enumerate(paper_nodes, 1):
                    try:
                        # 查找标题
                        title_elem = node.find(['h2', 'h3'], class_=['node-title', 'paper-title']) or \
                                   node.find('div', class_='field-title')
                        if not title_elem:
                            continue
                        
                        title = title_elem.text.strip()
                        
                        # 查找PDF链接
                        pdf_url = await self._get_usenix_pdf_url(node)
                        if pdf_url:
                            papers.append({
                                'title': title,
                                'url': pdf_url,
                                'conference': 'USENIX',
                                'year': year
                            })
                            
                            print(f"\r📄 处理论文 {idx}/{len(paper_nodes)}: {title[:50]}...", end='', flush=True)
                
                print(f"\n✅ USENIX解析完成: {len(papers)} 篇论文")
                return papers
                
            elif response.status_code == 404:
                print(f"❌ USENIX {year} 页面不存在")
                return []
            else:
                raise Exception(f"HTTP {response.status_code}")
                    
        except Exception as e:
            print(f"❌ USENIX解析错误: {str(e)}")
            return []

    async def _get_usenix_pdf_url(self, node) -> Optional[str]:
        """从USENIX论文节点获取PDF链接"""
        try:
            # 直接查找PDF链接
            pdf_link = node.find('a', href=re.compile(r'\.pdf$', re.I))
            if pdf_link and pdf_link.get('href'):
                pdf_url = pdf_link['href']
                return self._complete_usenix_url(pdf_url)
            
            # 查找presentation链接
            pres_link = node.find('a', href=re.compile(r'/presentation/', re.I))
            if pres_link and pres_link.get('href'):
                pres_url = pres_link['href']
                return self._complete_usenix_url(pres_url)
            
            return None
            
        except Exception as e:
            print(f"⚠️ PDF链接提取失败: {str(e)}")
            return None

    def _complete_usenix_url(self, url: str) -> str:
        """补全USENIX URL"""
        if not url:
            return ''
        
        if url.startswith('http'):
            return url
        elif url.startswith('//'):
            return f"https:{url}"
        elif url.startswith('/'):
            return f"https://www.usenix.org{url}"
        else:
            return f"https://www.usenix.org/{url}"

    async def parse_sp_papers(self, base_url: str, year: str, session: AsyncSession) -> List[Dict[str, Any]]:
        """解析IEEE S&P论文列表"""
        papers = []
        full_year = f"20{year}" if len(year) == 2 else year
        
        print(f"🌐 正在获取 IEEE S&P {full_year} 论文列表...")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1'
        }
        
        try:
            response = await session.get(base_url, headers=headers)
            if response.status_code == 200:
                print(f"📝 正在解析页面内容...")
                html = response.text
                soup = BeautifulSoup(html, 'html.parser')
                    
                # 查找论文项
                paper_items = soup.find_all(['div', 'article'], class_=['paper-item', 'article-item'])
                print(f"📚 找到 {len(paper_items)} 个论文项")
                
                if not paper_items:
                    paper_items = soup.find_all(['div', 'article'], class_=['paper', 'article'])
                
                for idx, item in enumerate(paper_items, 1):
                    try:
                        # 查找标题
                        title_elem = item.find(['h3', 'h2'], class_=['paper-title', 'article-title'])
                        if not title_elem:
                            continue
                            
                        title = title_elem.text.strip()
                        
                        # 查找PDF链接
                        pdf_url = await self._get_ieee_pdf_url(item)
                        if pdf_url:
                            papers.append({
                                'title': title,
                                'url': pdf_url,
                                'conference': 'SP',
                                'year': year
                            })
                            
                            print(f"\r📄 处理论文 {idx}/{len(paper_items)}: {title[:50]}...", end='', flush=True)
                    except Exception as e:
                        print(f"\n⚠️ 处理论文时出错: {str(e)}")
                        continue
                            
                    except Exception as e:
                        print(f"\n⚠️ 处理论文时出错: {str(e)}")
                        continue
                
                print(f"\n✅ SP解析完成: {len(papers)} 篇论文")
                return papers
                    
            else:
                raise Exception(f"HTTP {response.status_code}")
                    
        except Exception as e:
            print(f"❌ SP解析错误: {str(e)}")
            return []

    async def _get_ieee_pdf_url(self, paper_element) -> Optional[str]:
        """从IEEE论文元素中提取PDF URL"""
        try:
            # 直接查找PDF链接
            pdf_link = paper_element.find('a', href=re.compile(r'\.pdf$', re.I))
            if pdf_link and pdf_link.get('href'):
                return pdf_link['href']
            
            # 查找文章链接并构造PDF URL
            article_link = paper_element.find('a', href=re.compile(r'/document/'))
            if article_link and article_link.get('href'):
                doc_match = re.search(r'/document/(\d+)', article_link['href'])
                if doc_match:
                    doc_id = doc_match.group(1)
                    return f"https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber={doc_id}"
            
            return None
            
        except Exception as e:
            print(f"⚠️ PDF链接提取失败: {str(e)}")
            return None
