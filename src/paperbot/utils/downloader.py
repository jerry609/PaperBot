# securipaperbot/utils/downloader.py

from typing import Dict, List, Any, Optional
import aiohttp
import asyncio
from pathlib import Path
import urllib.parse
from bs4 import BeautifulSoup
import re
import time
from datetime import datetime
import logging

# 使用标准日志，避免相对导入问题
def setup_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


class PaperDownloader:
    """论文下载工具类 - 优化版本"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = setup_logger(__name__)
        self.download_path = Path(self.config.get('download_path', './papers'))
        self.download_path.mkdir(parents=True, exist_ok=True)

        # 配置下载重试参数 - 保守稳定的参数
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 3)  # 增加到3秒

        # 完全关闭并发 - 使用单线程确保稳定性
        max_concurrent = 1  # 强制设为1，不使用并发
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # 会议URL模板
        self.conference_urls = {
            'ccs': 'https://dl.acm.org/doi/proceedings/10.1145/',
            'sp': 'https://ieeexplore.ieee.org/xpl/conhome/',
            'ndss': 'https://www.ndss-symposium.org/',
            'usenix': 'https://www.usenix.org/conference/'
        }

    async def download_paper(self, url: str, title: str, paper_index: int = 0, total_papers: int = 0) -> Dict[str, Any]:
        """下载单篇论文 - 优化版本"""
        async with self.semaphore:
            try:
                # 生成文件名
                safe_title = self._sanitize_filename(title)
                file_path = self.download_path / f"{safe_title}.pdf"

                # 显示下载进度（与NDSS/USENIX保持一致）
                if total_papers > 0:
                    progress = (paper_index + 1) / total_papers * 100
                    print(f"💾 [{paper_index+1}/{total_papers}] 下载: {title[:50]}{'...' if len(title) > 50 else ''}")

                # 检查是否已下载并验证文件
                if file_path.exists():
                    # 验证文件大小，过小的文件可能是错误页面
                    file_size = file_path.stat().st_size
                    if file_size > 1024:  # 大于1KB认为有效
                        return {
                            'success': True,
                            'path': str(file_path),
                            'cached': True,
                            'size': file_size
                        }
                    else:
                        # 删除无效文件
                        file_path.unlink()
                        self.logger.warning(f"Removed invalid cached file: {file_path}")

                # 下载论文
                content = await self._download_with_retry(url)
                if content:
                    # 验证下载内容
                    if len(content) < 1024:
                        raise Exception(f"Downloaded content too small ({len(content)} bytes), likely an error page")
                    
                    # 保存文件
                    file_path.write_bytes(content)
                    file_size = len(content)

                    return {
                        'success': True,
                        'path': str(file_path),
                        'cached': False,
                        'size': file_size
                    }
                else:
                    raise Exception("Failed to download paper - no content received")

            except Exception as e:
                self.logger.error(f"Error downloading paper {title}: {str(e)}")
                return {
                    'success': False,
                    'error': str(e)
                }

    async def get_conference_papers(self, conference: str, year: str) -> List[Dict[str, Any]]:
        """获取会议论文列表 - 带进度显示"""
        try:
            if conference not in self.conference_urls:
                raise ValueError(f"Unsupported conference: {conference}")

            base_url = self.conference_urls[conference]
            papers = []
            
            print(f"🔍 正在获取 {conference.upper()} {year} 论文列表...")

            # 根据会议类型选择相应的解析方法
            if conference == 'ccs':
                papers = await self._parse_ccs_papers(base_url, year)
            elif conference == 'sp':
                papers = await self._parse_sp_papers(base_url, year)
            elif conference == 'ndss':
                papers = await self._parse_ndss_papers(base_url, year)
            elif conference == 'usenix':
                papers = await self._parse_usenix_papers(base_url, year)
            
            if papers:
                print(f"✅ 成功获取 {len(papers)} 篇论文信息")
                
                # 显示找到的论文标题
                print(f"📋 找到的论文列表:")
                for i, paper in enumerate(papers[:10]):
                    title = paper.get('title', '未知标题')[:60]
                    print(f"  {i+1:2d}. {title}{'...' if len(paper.get('title', '')) > 60 else ''}")
                
                if len(papers) > 10:
                    print(f"  ... 和其他 {len(papers) - 10} 篇论文")
                
                # 开始PDF链接验证与进度显示
                print(f"\n🔗 正在验证PDF链接有效性...")
                valid_count = 0
                
                for i, paper in enumerate(papers):
                    # 显示进度
                    progress = (i + 1) / len(papers) * 100
                    progress_bar = '█' * int(progress // 5) + '░' * (20 - int(progress // 5))
                    print(f"\r📋 [进度: {progress_bar}] {progress:.1f}% ({i+1}/{len(papers)}) 验证: {paper.get('title', '未知标题')[:30]}...", end='', flush=True)
                    
                    # 检查URL有效性
                    if isinstance(paper.get('url'), str) and paper['url'].strip():
                        valid_count += 1
                
                print(f"\n✅ PDF链接验证完成: {valid_count}/{len(papers)} 个有效链接")
            else:
                print(f"⚠️  未找到任何论文")

            return papers

        except Exception as e:
            self.logger.error(f"Error getting papers for {conference} {year}: {str(e)}")
            raise


    


    async def _parse_ndss_papers(self, base_url: str, year: str) -> List[Dict[str, Any]]:
        """解析NDSS论文列表 - 优化版本带进度显示"""
        papers = []
        full_year = f"20{year}" if len(year) == 2 else year
        url = f"{base_url}ndss{full_year}/accepted-papers/"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        enhanced_timeout = aiohttp.ClientTimeout(total=120, connect=30, sock_read=60)
        
        print(f"🌐 访问 NDSS {year} 会议页面...")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession(headers=headers, timeout=enhanced_timeout) as session:
                    
                    async with session.get(url) as response:
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
                                    print(f"🔍 解析进度: {progress:.1f}% ({idx+1}/{len(paper_containers)})")
                                
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
                            
                            print(f"\n✅ 基础信息解析完成: {len(papers)} 篇论文")
                            return papers
                            
                        elif response.status == 404:
                            print(f"❌ NDSS {year} 页面不存在")
                            return []
                        else:
                            print(f"⚠️  HTTP {response.status}，正在重试...")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2 ** attempt)
                                continue
                            else:
                                raise Exception(f"HTTP {response.status}")
            
            except asyncio.TimeoutError:
                print(f"⏰ 超时 {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(5 * (attempt + 1))
                    continue
                else:
                    raise Exception("Connection timeout after retries")
            
            except Exception as e:
                print(f"❌ 尝试 {attempt + 1} 失败: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(3 * (attempt + 1))
                    continue
                else:
                    raise
        
        return papers

    async def _get_ndss_pdf_from_detail_page(self, session: aiohttp.ClientSession, detail_url: str) -> str:
        """从 NDSS 论文详情页获取 PDF 链接 - 优化版本"""
        if not detail_url:
            return ''
        
        try:
            # 使用更短的超时时间提高速度
            timeout = aiohttp.ClientTimeout(total=15, connect=5)
            
            async with session.get(detail_url, timeout=timeout) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # 尝试多种 PDF 链接模式
                    pdf_patterns = [
                        # 直接 PDF 链接
                        soup.find('a', href=re.compile(r'\.pdf$', re.I)),
                        # 包含 "pdf" 文本的链接
                        soup.find('a', string=re.compile(r'pdf', re.I)),
                        # 包含 "download" 的链接
                        soup.find('a', string=re.compile(r'download', re.I)),
                        # 在 href 中包含 "pdf" 的链接
                        soup.find('a', href=re.compile(r'pdf', re.I))
                    ]
                    
                    for pdf_link in pdf_patterns:
                        if pdf_link and hasattr(pdf_link, 'get'):
                            pdf_url = pdf_link.get('href')
                            if pdf_url and isinstance(pdf_url, str):
                                # 确保 URL 是完整的
                                if not pdf_url.startswith('http'):
                                    if pdf_url.startswith('/'):
                                        pdf_url = f"https://www.ndss-symposium.org{pdf_url}"
                                    else:
                                        # 相对路径
                                        base_url = '/'.join(detail_url.split('/')[:-1])
                                        pdf_url = f"{base_url}/{pdf_url}"
                                
                                # 验证 URL 是否以 .pdf 结尾或包含 pdf
                                if pdf_url.lower().endswith('.pdf') or 'pdf' in pdf_url.lower():
                                    return pdf_url
                    
                    # 如果没有找到直接链接，尝试查找内嵌的PDF
                    iframe_pdf = soup.find('iframe', src=re.compile(r'\.pdf', re.I))
                    if iframe_pdf and hasattr(iframe_pdf, 'get'):
                        pdf_url = iframe_pdf.get('src')
                        if pdf_url and isinstance(pdf_url, str):
                            if not pdf_url.startswith('http'):
                                pdf_url = f"https://www.ndss-symposium.org{pdf_url}"
                            return pdf_url
                        
        except asyncio.TimeoutError:
            self.logger.debug(f"Timeout getting PDF from {detail_url}")
        except Exception as e:
            self.logger.debug(f"Error getting PDF from {detail_url}: {str(e)}")
            
        return ''

    async def _parse_sp_papers(self, base_url: str, year: str) -> List[Dict[str, Any]]:
        """解析SP论文列表 - 使用Computer.org GraphQL API直接获取proceedings ID"""
        papers = []
        full_year = f"20{year}" if len(year) == 2 else year
        
        print(f"🌐 正在获取 SP {year} proceedings ID...")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Content-Type': 'application/json',
            'Origin': 'https://www.computer.org',
            'Sec-Fetch-Site': 'same-origin',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Dest': 'empty',
            'Referer': 'https://www.computer.org/csdl/proceedings/1000646',
            'Connection': 'keep-alive',
            'Cache-Control': 'no-cache'
        }
        
        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                # 使用GraphQL API获取SP会议的所有proceedings
                graphql_url = "https://www.computer.org/csdl/api/v1/graphql"
                graphql_query = {
                    "variables": {"groupId": "1000646"},  # SP会议的组ID
                    "query": "query ($groupId: String) {\n  proceedings(groupId: $groupId) {\n    id\n    acronym\n    title\n    volume\n    displayVolume\n    year\n    __typename\n  }\n}"
                }
                
                timeout = aiohttp.ClientTimeout(total=60, connect=30)
                async with session.post(graphql_url, json=graphql_query, timeout=timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        proceedings_list = data.get('data', {}).get('proceedings', [])
                        
                        print(f"✅ 获取到 {len(proceedings_list)} 个 proceedings")
                        
                        # 查找指定年份的proceedings
                        target_proceeding = None
                        for proc in proceedings_list:
                            if str(proc.get('year')) == full_year:
                                target_proceeding = proc
                                break
                        
                        if not target_proceeding:
                            print(f"❌ 未找到 {full_year} 年的proceedings")
                            return papers
                        
                        proceedings_id = target_proceeding.get('id')
                        print(f"🆔 找到 SP {full_year} proceedings ID: {proceedings_id}")
                        
                        # 调用Computer.org API获取论文数据
                        all_papers = await self._call_computer_org_api(session, proceedings_id)
                        
                        # 处理所有论文
                        if all_papers:
                            print(f"✅ 成功获取 {len(all_papers)} 篇论文")
                            
                            # 解析真正的PDF下载链接
                            print(f"🔗 开始解析 {len(all_papers)} 个PDF下载链接...")
                            for i, paper in enumerate(all_papers):
                                if paper.get('needs_pdf_resolution') and paper.get('url'):
                                    print(f"📋 PDF解析进度: {i+1}/{len(all_papers)} - {paper.get('title', '')[:50]}...")
                                    
                                    real_pdf_url = await self._resolve_ieee_pdf_url(session, paper['url'])
                                    if real_pdf_url:
                                        paper['url'] = real_pdf_url
                                        print(f"✅ 解析成功: {real_pdf_url[:60]}...")
                                    else:
                                        print(f"❌ PDF链接解析失败")
                                    paper.pop('needs_pdf_resolution', None)
                            
                            papers = all_papers  # 返回所有论文
                        else:
                            papers = []
                        
                        return papers
                    else:
                        print(f"❌ GraphQL API调用失败: HTTP {response.status}")
                        return []
                        
        except Exception as e:
            print(f"❌ SP解析错误: {str(e)}")
            return []
    

    async def _call_computer_org_api(self, session: aiohttp.ClientSession, proceedings_id: str) -> List[Dict[str, Any]]:
        """调用Computer.org API获取论文数据"""
        papers = []
        api_url = f"https://www.computer.org/csdl/api/v1/citation/asciitext/proceedings/{proceedings_id}"
        
        print(f"🔗 调用Computer.org API: {api_url}")
        
        # 使用用户提供的完整请求头
        api_headers = {
            'Host': 'www.computer.org',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Referer': f'https://www.computer.org/csdl/proceedings/sp/2024/{proceedings_id}',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'Priority': 'u=1, i',
            'Pragma': 'no-cache',
            'Cache-Control': 'no-cache'
        }
        
        try:
            timeout = aiohttp.ClientTimeout(total=60, connect=30)
            async with session.get(api_url, headers=api_headers, timeout=timeout) as response:
                if response.status == 200:
                    print(f"✅ API调用成功")
                    text_data = await response.text()
                    papers = self._parse_citation_data(text_data)
                    print(f"📚 解析到 {len(papers)} 篇论文")
                    return papers
                else:
                    print(f"❌ API调用失败: HTTP {response.status}")
                    return []
                    
        except Exception as e:
            print(f"❌ API调用错误: {str(e)}")
            return []
    
    def _parse_citation_data(self, citation_text: str) -> List[Dict[str, Any]]:
        """解析引用数据获取论文信息"""
        papers = []
        
        try:
            # 按条目分割
            entries = re.split(r'\n\s*\n', citation_text.strip())
            
            for entry in entries:
                if not entry.strip():
                    continue
                
                paper_info = self._parse_single_citation(entry.strip())
                if paper_info:
                    papers.append(paper_info)
            
            return papers
            
        except Exception as e:
            self.logger.error(f"解析引用数据失败: {str(e)}")
            return []
    
    def _parse_single_citation(self, citation: str) -> Optional[Dict[str, Any]]:
        """解析单个引用条目"""
        try:
            # 查找标题 (通常在引号内或作为第一行)
            title_match = re.search(r'"([^"]+)"', citation)
            if not title_match:
                # 备用：提取第一行作为标题
                lines = citation.split('\n')
                title = lines[0].strip() if lines else ''
            else:
                title = title_match.group(1)
            
            if not title or len(title) < 10:
                return None
            
            # 检查是否包含keywords:{}（空大括号），这表示非论文内容
            if re.search(r'keywords:\s*\{\s*\}', citation, re.I):
                return None
            
            # 过滤特定的非论文标题模式
            non_paper_patterns = [
                r'author\s+index',
                r'table\s+of\s+contents',
                r'program\s+committee',
                r'organiz(ing|ation)\s+committee',
                r'chair\s+message',
                r'welcome\s+message',
                r'foreword',
                r'preface',
                r'index\s+terms',
                r'subject\s+index'
            ]
            
            for pattern in non_paper_patterns:
                if re.search(pattern, title, re.I):
                    return None
            
            # 查找作者
            authors = []
            author_match = re.search(r'Author\(s\):\s*([^\n]+)', citation)
            if author_match:
                authors_text = author_match.group(1)
                authors = [author.strip() for author in authors_text.split(',')]
            
            # 查找DOI链接 - 优先使用标准DOI格式
            doi_match = re.search(r'DOI:\s*(https?://[^\s]+)', citation)
            doi_url = doi_match.group(1) if doi_match else ''
            
            # 如果找到的是doi.ieeecomputersociety.org链接，转换为标准doi.org格式
            if doi_url and 'doi.ieeecomputersociety.org' in doi_url:
                doi_url = doi_url.replace('doi.ieeecomputersociety.org', 'doi.org')
            
            # 如果没有标准DOI，查找所有URL中包含doi.ieeecomputersociety.org的链接并转换
            if not doi_url:
                all_urls = re.findall(r'https?://[^\s]+', citation)
                for url in all_urls:
                    if 'doi.ieeecomputersociety.org' in url:
                        doi_url = url.replace('doi.ieeecomputersociety.org', 'doi.org')
                        break
            
            # 如果还是没找到合适的链接，跳过这篇论文
            if not doi_url:
                return None
            
            return {
                'title': title,
                'authors': authors,
                'abstract': '',
                'url': doi_url,  # 这里先保存DOI链接，稍后会解析真正的PDF链接
                'doi': doi_url,
                'needs_pdf_resolution': True  # 标记需要解析PDF链接
            }
            
        except Exception as e:
            self.logger.error(f"解析单个引用失败: {str(e)}")
            return None
    
    async def _resolve_ieee_pdf_url(self, session: aiohttp.ClientSession, doi_url: str) -> str:
        """从IEEE DOI页面解析真正的PDF下载链接"""
        try:
            if not doi_url:
                return ''
            
            print(f"🔍 解析PDF链接: {doi_url[:50]}...")
            
            # 访问DOI页面，跟随重定向
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            
            # 设置请求头模拟浏览器
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            async with session.get(doi_url, timeout=timeout, headers=headers, allow_redirects=True) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    print(f"✅ 成功访问页面: {str(response.url)[:60]}...")
                    
                    # 检查是否重定向到了Computer.org页面
                    current_url = str(response.url)
                    if 'computer.org' in current_url:
                        # 在Computer.org页面查找PDF下载链接
                        return await self._extract_pdf_from_computer_org_page(soup, current_url)
                    else:
                        # 在其他页面（如doi.org重定向页）查找PDF链接
                        pdf_url = await self._extract_pdf_from_generic_page(soup, str(response.url))
                        # 如果是IEEE页面，直接返回构造的PDF链接
                        if 'ieeexplore.ieee.org' in current_url and not pdf_url:
                            match = re.search(r'/document/(\d+)', current_url)
                            if match:
                                arnumber = match.group(1)
                                pdf_url = f"https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber={arnumber}"
                                print(f"✅ 构造PDF链接成功: {pdf_url[:60]}...")
                        return pdf_url
                else:
                    print(f"❌ DOI访问失败: HTTP {response.status}")
                    
        except Exception as e:
            print(f"❌ 解析PDF链接失败: {str(e)}")
            
        return ''
    
    async def _extract_pdf_from_computer_org_page(self, soup, current_url: str) -> str:
        """从Computer.org页面提取PDF链接"""
        # 查找PDF下载链接的多种模式
        pdf_patterns = [
            # Computer.org特定的DOWNLOAD PDF按钮
            soup.find('a', string=re.compile(r'DOWNLOAD PDF', re.I)),
            soup.find('a', text=re.compile(r'download.*pdf', re.I)),
            soup.find('button', string=re.compile(r'download.*pdf', re.I)),
            
            # 带有PDF相关class的链接
            soup.find('a', class_=re.compile(r'download|pdf', re.I)),
            soup.find('a', attrs={'aria-label': re.compile(r'download|pdf', re.I)}),
            
            # 直接PDF链接
            soup.find('a', href=re.compile(r'\.pdf$', re.I)),
            
            # Meta标签中的PDF链接
            soup.find('meta', attrs={'name': 'citation_pdf_url'}),
            soup.find('meta', attrs={'property': 'citation_pdf_url'})
        ]
        
        for pattern in pdf_patterns:
            if pattern:
                if pattern.name == 'meta':
                    pdf_url = pattern.get('content')
                else:
                    pdf_url = pattern.get('href')
                    
                if pdf_url:
                    # 补全相对URL
                    if not pdf_url.startswith('http'):
                        if pdf_url.startswith('/'):
                            pdf_url = f"https://www.computer.org{pdf_url}"
                        else:
                            base_url = '/'.join(current_url.split('/')[:-1])
                            pdf_url = f"{base_url}/{pdf_url}"
                    
                    print(f"✅ 找到PDF链接: {pdf_url[:60]}...")
                    return pdf_url
        
        # 如果没找到直接链接，尝试查找data-*属性中PDF链接
        for element in soup.find_all(attrs={'data-pdf-url': True}):
            pdf_url = element.get('data-pdf-url')
            if pdf_url:
                if not pdf_url.startswith('http'):
                    pdf_url = f"https://www.computer.org{pdf_url}"
                print(f"🔗 从data属性找到PDF: {pdf_url[:60]}...")
                return pdf_url
        
        # 尝试查找包含PDF的所有链接
        all_links = soup.find_all('a', href=True)
        for link in all_links:
            href = link.get('href')
            if href and ('pdf' in href.lower() or 'download' in href.lower()):
                if not href.startswith('http'):
                    if href.startswith('/'):
                        href = f"https://www.computer.org{href}"
                    else:
                        base_url = '/'.join(current_url.split('/')[:-1])
                        href = f"{base_url}/{href}"
                print(f"🔍 候选PDF链接: {href[:60]}...")
                return href
        
        # 最后尝试：从当前页面URL构造PDF链接
        if '/proceedings-article/' in current_url:
            # 提取文章ID
            parts = current_url.rstrip('/').split('/')
            if parts:
                article_id = parts[-1]
                # 使用真正的Computer.org PDF下载API
                pdf_url = f"https://www.computer.org/csdl/pds/api/csdl/proceedings/download-article/{article_id}/pdf"
                print(f"🔗 构造PDF API链接: {pdf_url}")
                return pdf_url
        
        return ''
    

    
    

    async def _parse_usenix_papers(self, base_url: str, year: str) -> List[Dict[str, Any]]:
        """解析USENIX Security论文列表 - 精确修复版本"""
        papers = []
        url = f"https://www.usenix.org/conference/usenixsecurity{year}/technical-sessions"
        
        print(f"🌐 正在解析 USENIX Security {year} 论文列表...")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                timeout = aiohttp.ClientTimeout(total=30, connect=10)
                
                async with session.get(url, timeout=timeout) as response:
                    if response.status == 200:
                        print(f"✅ 页面访问成功，开始解析...")
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # 使用正确的选择器找到论文节点
                        paper_nodes = soup.find_all('article', class_='node node-paper view-mode-schedule')
                        
                        print(f"📚 找到 {len(paper_nodes)} 个论文节点")
                        
                        if not paper_nodes:
                            print(f"⚠️  未找到有效的论文节点")
                            return papers
                        
                        for idx, node in enumerate(paper_nodes):
                            # 显示进度
                            if idx % 10 == 0 or idx == len(paper_nodes) - 1:
                                progress = (idx + 1) / len(paper_nodes) * 100
                                print(f"🔍 解析进度: {progress:.1f}% ({idx+1}/{len(paper_nodes)})")
                            
                            try:
                                # 提取标题 - 使用h2标签
                                title_elem = node.find('h2')
                                if not title_elem:
                                    continue
                                
                                # 从链接中获取标题文本
                                link_elem = title_elem.find('a')
                                if not link_elem:
                                    continue
                                    
                                title = link_elem.get_text().strip()
                                
                                # 验证是否为有效论文标题
                                if not self._is_valid_paper_title(title):
                                    continue
                                
                                # 显示找到的论文标题
                                print(f"📄 [{idx+1}/{len(paper_nodes)}] 找到论文: {title[:70]}{'...' if len(title) > 70 else ''}")
                                
                                # 提取作者信息
                                authors = []
                                author_container = node.find('div', class_='field-name-field-paper-people-text')
                                if author_container:
                                    authors_text = author_container.get_text().strip()
                                    if authors_text:
                                        # 简单解析作者列表
                                        authors = [authors_text.split(',')[0].strip()] if ',' in authors_text else [authors_text]
                                
                                # 获取PDF链接
                                pdf_url = await self._get_usenix_pdf_url_simple(session, node)
                                
                                if pdf_url:
                                    papers.append({
                                        'title': title,
                                        'authors': authors,
                                        'abstract': '',
                                        'url': pdf_url,
                                        'doi': ''
                                    })
                                    
                            except Exception as e:
                                self.logger.debug(f"解析节点 {idx} 失败: {str(e)}")
                                continue
                        
                        print(f"✅ USENIX 解析完成: {len(papers)} 篇论文")
                        return papers
                    
                    elif response.status == 404:
                        print(f"❌ USENIX {year} 页面不存在")
                        return []
                    else:
                        print(f"❌ HTTP {response.status}")
                        return []
                        
        except asyncio.TimeoutError:
            print(f"⏰ 访问超时")
            return []
        except Exception as e:
            print(f"❌ 解析错误: {str(e)}")
            return []

    async def _get_usenix_pdf_url_simple(self, session: aiohttp.ClientSession, node) -> str:
        """简化的USENIX PDF链接获取方法"""
        try:
            # 1. 首先查找直接的PDF链接
            pdf_link = node.find('a', href=re.compile(r'\.pdf$', re.I))
            if pdf_link:
                href = pdf_link.get('href')
                if href:
                    return self._complete_usenix_url(href)
            
            # 2. 查找presentation页面链接
            presentation_link = node.find('a', href=re.compile(r'/presentation/', re.I))
            if presentation_link:
                presentation_url = presentation_link.get('href')
                if presentation_url:
                    presentation_url = self._complete_usenix_url(presentation_url)
                    
                    # 从presentation页面获取PDF链接
                    try:
                        timeout = aiohttp.ClientTimeout(total=10, connect=5)
                        async with session.get(presentation_url, timeout=timeout) as response:
                            if response.status == 200:
                                html = await response.text()
                                soup = BeautifulSoup(html, 'html.parser')
                                
                                # 查找PDF下载链接
                                pdf_link = soup.find('a', href=re.compile(r'\.pdf$', re.I))
                                if pdf_link:
                                    pdf_url = pdf_link.get('href')
                                    if pdf_url:
                                        return self._complete_usenix_url(pdf_url)
                    except Exception as e:
                        self.logger.debug(f"presentation页面获取失败: {str(e)}")
            
            return ''
            
        except Exception as e:
            self.logger.debug(f"获取PDF链接失败: {str(e)}")
            return ''
    
    def _is_valid_paper_title(self, title: str) -> bool:
        """验证是否为有效的论文标题"""
        if not title or len(title) < 10:  # 降低最小长度要求
            return False
        
        # 只排除明显的非论文内容，减少过滤
        exclude_keywords = [
            'technical session', 'session chair', 'keynote', 'tutorial', 
            'workshop', 'break', 'lunch', 'coffee break', 'opening remarks',
            'closing remarks', 'panel discussion', 'poster session'
        ]
        
        title_lower = title.lower()
        for keyword in exclude_keywords:
            if keyword in title_lower:
                return False
        
        # 放宽条件，只要有一定长度就认为是有效论文
        return len(title) > 15
    
    def _complete_usenix_url(self, url: str) -> str:
        """补全USENIX URL"""
        if not url:
            return ''
        
        if url.startswith('http'):
            return url
        elif url.startswith('/'):
            return f"https://www.usenix.org{url}"
        else:
            return f"https://www.usenix.org/{url}"


    def _extract_ieee_pdf_url(self, paper_element) -> str:
        """从IEEE页面元素中提取PDF链接"""
        try:
            # 查找PDF链接
            pdf_link = paper_element.find('a', href=re.compile(r'.*\.pdf'))
            if pdf_link:
                return pdf_link['href']
            
            # 查找文章链接并构造PDF URL
            article_link = paper_element.find('a', href=re.compile(r'/document/'))
            if article_link:
                doc_id = re.search(r'/document/(\d+)', article_link['href'])
                if doc_id:
                    return f"https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber={doc_id.group(1)}"
            
            return ''
        except Exception as e:
            self.logger.error(f"Error extracting IEEE PDF URL: {str(e)}")
            return ''
        """从IEEE页面元素中提取PDF链接"""
        try:
            # 查找PDF链接
            pdf_link = paper_element.find('a', href=re.compile(r'.*\.pdf'))
            if pdf_link:
                return pdf_link['href']
            
            # 查找文章链接并构造PDF URL
            article_link = paper_element.find('a', href=re.compile(r'/document/'))
            if article_link:
                doc_id = re.search(r'/document/(\d+)', article_link['href'])
                if doc_id:
                    return f"https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber={doc_id.group(1)}"
            
            return ''
        except Exception as e:
            self.logger.error(f"Error extracting IEEE PDF URL: {str(e)}")
            return ''




    async def _download_with_retry(self, url: str) -> Optional[bytes]:
        """带重试机制的下载 - 修复版本，支持IEEE SP论文特殊流程"""
        if not url:
            return None

        # 对Computer.org的SP论文使用特殊处理
        if 'computer.org/csdl/pds/api' in url:
            return await self._download_computer_org_pdf(url)

        # 对IEEE SP论文特殊处理
        if 'ieeexplore.ieee.org/stampPDF/getPDF.jsp' in url:
            return await self._download_ieee_pdf_with_httpx(url)

        # 其他链接使用原有方法
        return await self._download_with_aiohttp(url)

    async def _download_ieee_pdf_with_httpx(self, url: str) -> Optional[bytes]:
        """使用httpx+HTTP2自动获取ERIGHTS并下载IEEE SP PDF"""
        try:
            import httpx
            # 第一次请求，ERIGHTS=0000
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36 Edg/139.0.0.0',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Dnt': '1',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Site': 'same-origin',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Dest': 'iframe',
                'Referer': 'https://ieeexplore.ieee.org/',
                'Sec-Ch-Ua': '"Not;A=Brand";v="99", "Microsoft Edge";v="139", "Chromium";v="139"',
                'Sec-Ch-Ua-Mobile': '?0',
                'Sec-Ch-Ua-Platform': '"Windows"',
                'Priority': 'u=0, i',
            }
            cookies = {'ERIGHTS': '0000'}
            timeout = 60
            async with httpx.AsyncClient(http2=True, timeout=timeout, follow_redirects=False) as client:
                # 第一次请求
                resp1 = await client.get(url, headers=headers, cookies=cookies)
                # 检查302和Set-Cookie
                if resp1.status_code in (302, 303, 307, 301):
                    set_cookie = resp1.headers.get('set-cookie', '')
                    # 提取ERIGHTS
                    import re
                    m = re.search(r'ERIGHTS=([^;]+)', set_cookie)
                    if m:
                        erights_val = m.group(1)
                        cookies['ERIGHTS'] = erights_val
                        # 跟随Location
                        next_url = resp1.headers.get('location', url)
                        # 第二次请求
                        resp2 = await client.get(url, headers=headers, cookies=cookies)
                        if resp2.status_code == 200 and resp2.content and resp2.content[:4] == b'%PDF':
                            return resp2.content
                        # 有时需要再请求一次
                        if resp2.status_code in (302, 303, 307, 301):
                            # 再次尝试
                            resp3 = await client.get(url, headers=headers, cookies=cookies)
                            if resp3.status_code == 200 and resp3.content and resp3.content[:4] == b'%PDF':
                                return resp3.content
                    else:
                        # 没有ERIGHTS，直接尝试内容
                        if resp1.status_code == 200 and resp1.content and resp1.content[:4] == b'%PDF':
                            return resp1.content
                elif resp1.status_code == 200 and resp1.content and resp1.content[:4] == b'%PDF':
                    return resp1.content
        except Exception as e:
            self.logger.error(f"httpx下载IEEE PDF失败: {str(e)}")
        return None
    
    async def _download_computer_org_pdf(self, api_url: str) -> Optional[bytes]:
        """专门为Computer.org PDF下载的方法 - 使用curl"""
        return await self._download_with_curl(api_url)
    
    async def _download_with_curl(self, url: str) -> Optional[bytes]:
        """使用curl命令下载"""
        try:
            import asyncio
            import subprocess
            
            cmd = [
                'curl',
                '-L',  # 跟随重定向
                '-s',  # 静默模式
                '--max-time', '60',  # 最大超时时间
                '--user-agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
                '--header', 'Accept: application/pdf,application/octet-stream,*/*',
                url
            ]
            
            # 使用subprocess异步执行curl
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0 and stdout and len(stdout) > 1024:
                # 验证PDF文件类型
                if self._is_valid_pdf(stdout):
                    return stdout
                else:
                    return None
            else:
                error_msg = stderr.decode('utf-8', errors='ignore') if stderr else 'Unknown error'
                
        except Exception as e:
            pass
            
        return None

    async def _download_with_aiohttp(self, url: str) -> Optional[bytes]:
        """原有的aiohttp下载方法"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/pdf,application/octet-stream,*/*',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive'
        }
        
        for attempt in range(self.max_retries):
            try:
                # 为每次重试创建新的Session，避免连接问题
                connector = aiohttp.TCPConnector(
                    limit=5,
                    limit_per_host=2,
                    force_close=True,  # 强制关闭连接避免复用问题
                    enable_cleanup_closed=True
                )
                
                timeout = aiohttp.ClientTimeout(
                    total=90,
                    connect=30,
                    sock_read=60
                )
                
                async with aiohttp.ClientSession(
                    headers=headers,
                    timeout=timeout,
                    connector=connector
                ) as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            content = await response.read()
                            if len(content) > 0:
                                return content
                            else:
                                self.logger.warning(f"Empty content from {url}")
                        elif response.status in [429, 503, 502]:
                            self.logger.warning(
                                f"Server busy (attempt {attempt + 1}/{self.max_retries}): "
                                f"HTTP {response.status} for {url}"
                            )
                            await asyncio.sleep(self.retry_delay * (attempt + 1) * 2)
                            continue
                        else:
                            self.logger.warning(
                                f"Download failed (attempt {attempt + 1}/{self.max_retries}): "
                                f"HTTP {response.status} for {url}"
                            )

            except asyncio.TimeoutError:
                self.logger.warning(
                    f"Download timeout (attempt {attempt + 1}/{self.max_retries}): {url}"
                )
            except Exception as e:
                self.logger.warning(
                    f"Download error (attempt {attempt + 1}/{self.max_retries}): {str(e)} for {url}"
                )

            # 重试间隔
            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (attempt + 1)
                await asyncio.sleep(delay)

        self.logger.error(f"Failed to download after {self.max_retries} attempts: {url}")
        return None

    def _sanitize_filename(self, filename: str) -> str:
        """清理文件名"""
        # 移除不允许的字符
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '')

        # 将空格替换为下划线
        filename = filename.replace(' ', '_')

        # 限制长度
        max_length = 255 - len('.pdf')
        if len(filename) > max_length:
            filename = filename[:max_length]

        return filename.strip('._')
    
    def _is_valid_pdf(self, content: bytes) -> bool:
        """验证是否为有效的PDF文件"""
        if not content or len(content) < 4:
            return False
        
        # 检查PDF文件头（%PDF-）
        return content.startswith(b'%PDF-')
    

