# securipaperbot/utils/downloader.py

from typing import Dict, List, Any, Optional
import aiohttp
import asyncio
import httpx
from pathlib import Path
import urllib.parse
from bs4 import BeautifulSoup
import re
import json
import time
import random
from datetime import datetime
import logging
import traceback

# 添加动态cookie获取支持
import traceback
try:
    # curl_cffi 0.5.x 版本中, AsyncSession 位于 requests 模块下
    from curl_cffi.requests import AsyncSession
    CURL_CFFI_AVAILABLE = True
except ImportError:
    from typing import Any as AsyncSession # Mock for type hinting
    CURL_CFFI_AVAILABLE = False
    print("❌ 'curl_cffi' 导入失败。详细错误信息如下:")
    traceback.print_exc()
    print("警告: curl_cffi 未安装或无法加载，动态cookie获取功能(如ACM)将受限")

try:
    import cloudscraper
    CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    CLOUDSCRAPER_AVAILABLE = False
    print("警告: cloudscraper 未安装，动态cookie获取功能(如ACM)将受限")


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
    """论文下载工具类 - 优化版本，使用持久化会话"""
    
    # 会议基本信息配置
    CONFERENCE_INFO = {
        'sp': {
            'base_url': 'https://ieeexplore.ieee.org/xpl/conhome/1000487/all-proceedings',
            'headers': {
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Upgrade-Insecure-Requests': '1',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/116.0'
            }
        },
        'ndss': {
            'base_url': 'https://www.ndss-symposium.org',
            'headers': {
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Upgrade-Insecure-Requests': '1',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/116.0'
            }
        },
        'usenix': {
            'base_url': 'https://www.usenix.org/conference/usenixsecurity',
            'headers': {
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Upgrade-Insecure-Requests': '1',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/116.0'
            }
        }
    }
    
    async def _download_with_retry(self, url: str) -> Optional[bytes]:
        """
        智能下载实现，带自动重试和反爬处理。
        
        Args:
            url (str): 要下载的URL
            
        Returns:
            Optional[bytes]: 下载的内容，失败返回None
        """
        # 验证并确保会话可用
        if not self.session:
            try:
                self.logger.info("正在重新创建持久化会话...")
                self.session = AsyncSession()
            except Exception as e:
                self.logger.error(f"创建持久化会话失败: {e}")
                return None

        last_error = None
        content = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                self.logger.info(f"下载尝试 {attempt}/{self.max_retries}: {url}")
                
                # 配置特殊headers以绕过反爬
                headers = {
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                # 验证会话状态
                if not self.session or getattr(self.session, '_closed', False):
                    self.logger.warning("会话已关闭，正在重新创建...")
                    self.session = AsyncSession()
                
                # 使用curl_cffi的持久化会话和浏览器仿真
                response = await self.session.get(
                    url,
                    impersonate="chrome110",
                    headers=headers,
                    timeout=60
                )
                
                # 检查HTTP状态码
                if response.status_code == 403:
                    self.logger.warning(f"遇到403 Forbidden，可能是反爬限制 (尝试 {attempt}/{self.max_retries})")
                    await asyncio.sleep(self.retry_delay * attempt)  # 指数退避
                    continue
                    
                elif response.status_code == 429:
                    self.logger.warning(f"遇到429 Too Many Requests，开始等待 (尝试 {attempt}/{self.max_retries})")
                    await asyncio.sleep(self.retry_delay * 2 * attempt)  # 更长的等待
                    continue
                    
                elif response.status_code != 200:
                    self.logger.warning(f"HTTP {response.status_code} (尝试 {attempt}/{self.max_retries})")
                    await asyncio.sleep(self.retry_delay)
                    continue
                
                # 获取响应内容
                content = response.content
                
                # 验证内容
                if not content or len(content) < 1024:  # 小于1KB可能是错误页面
                    self.logger.warning(f"响应内容过小: {len(content) if content else 0} bytes")
                    continue
                
                # 对于PDF，验证文件头
                if url.lower().endswith('.pdf') and not content.startswith(b'%PDF'):
                    self.logger.warning("响应不是有效的PDF格式")
                    continue
                    
                self.logger.info(f"✅ 成功下载: {len(content)} bytes")
                return content
                
            except Exception as e:
                last_error = e
                self.logger.warning(f"下载出错 (尝试 {attempt}/{self.max_retries}): {e}")
                await asyncio.sleep(self.retry_delay)
                continue
        
        # 所有重试都失败
        if last_error:
            self.logger.error(f"❌ 下载失败，已达到最大重试次数。最后错误: {last_error}")
        else:
            self.logger.error("❌ 下载失败，未获得有效内容")
        return None

    def _sanitize_filename(self, filename: str) -> str:
        """清理并规范化文件名，移除非法字符"""
        # 替换 Windows 文件系统不允许的字符
        invalid_chars = r'[\\/:"*?<>|]+'
        filename = re.sub(invalid_chars, '_', filename)
        
        # 将连续的空白字符替换为单个空格
        filename = re.sub(r'\s+', ' ', filename)
        
        # 去除首尾空白
        filename = filename.strip()
        
        # 如果文件名为空，使用默认名称
        if not filename:
            filename = f"paper_{int(time.time())}"
            
        # 限制文件名长度（Windows 最大路径长度为 260 字符）
        max_length = 200  # 留一些余地给路径和扩展名
        if len(filename) > max_length:
            filename = filename[:max_length-3] + "..."
            
        return filename

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = setup_logger(__name__)
        self.download_path = Path(self.config.get('download_path', './papers'))
        self.download_path.mkdir(parents=True, exist_ok=True)
        
        self.session: Optional[AsyncSession] = None

        # 配置下载重试参数
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 3)

        # 并发控制
        max_concurrent = 1
        self.semaphore = asyncio.Semaphore(max_concurrent)


        # 会议URL模板
        self.conference_urls = {
            'ccs': 'https://dl.acm.org/doi/proceedings/',
            'sp': 'https://ieeexplore.ieee.org/xpl/conhome/',
            'ndss': 'https://www.ndss-symposium.org/',
            'usenix': 'https://www.usenix.org/conference/'
        }

    async def __aenter__(self):
        """创建并返回一个持久化的 curl_cffi 会话."""
        try:
            if self.session and not getattr(self.session, '_closed', False):
                self.logger.info("使用现有的持久化会话...")
                return self
                
            self.logger.info("正在创建新的持久化会话...")
            self.session = AsyncSession()
            return self
        except Exception as e:
            self.logger.error(f"创建持久化会话失败: {e}")
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """关闭持久化会话."""
        try:
            if self.session:
                # 检查会话是否已经关闭
                is_closed = getattr(self.session, '_closed', True)
                
                if not is_closed and hasattr(self.session, 'close'):
                    try:
                        self.logger.info("正在关闭持久化会话...")
                        await self.session.close()
                    except Exception as e:
                        self.logger.warning(f"关闭会话时出现异常: {e}")
                else:
                    self.logger.info("会话已经关闭，无需再次关闭")
        except Exception as e:
            self.logger.warning(f"处理会话关闭时出现异常: {e}")
        finally:
            # 确保会话对象被清理
            self.session = None

    async def download_paper(self, url: str, title: str, paper_index: int = 0, total_papers: int = 0) -> Dict[str, Any]:
        """下载单篇论文 - 优化版本"""
        async with self.semaphore:
            try:
                # 生成文件名 - 为IEEE论文添加特殊前缀
                safe_title = self._sanitize_filename(title)
                
                # 使用简化的文件名：只使用论文标题
                filename = f"{safe_title}.pdf"
                
                file_path = self.download_path / filename

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

    async def _parse_sp_papers(self, year: str) -> List[Dict[str, Any]]:
        """解析 IEEE S&P 论文列表"""
        papers = []
        full_year = f"20{year}" if len(year) == 2 else year
        
        try:
            print(f"📚 正在获取 IEEE S&P {full_year} 论文列表...")
            conf_info = self.CONFERENCE_INFO['sp']
            base_url = f"{conf_info['base_url']}"
            
            # 使用会话发送请求
            if not self.session:
                self.session = AsyncSession()
                
            response = await self.session.get(
                base_url,
                headers=conf_info['headers'],
                impersonate="chrome110"
            )
            
            if response.status_code != 200:
                raise Exception(f"获取会议页面失败: HTTP {response.status_code}")
                
            # 解析页面内容
            soup = BeautifulSoup(response.text, 'html.parser')
            paper_items = soup.select('div.paper-item')
            
            for item in paper_items:
                title_elem = item.select_one('h3.paper-title')
                if not title_elem:
                    continue
                    
                title = title_elem.text.strip()
                url = item.select_one('a[href*=".pdf"]')
                if not url:
                    continue
                    
                pdf_url = url['href']
                if not pdf_url.startswith('http'):
                    pdf_url = f"https://ieeexplore.ieee.org{pdf_url}"
                    
                papers.append({
                    'title': title,
                    'url': pdf_url
                })
                
            print(f"✅ 找到 {len(papers)} 篇论文")
            return papers
            
        except Exception as e:
            print(f"❌ 获取 IEEE S&P {full_year} 论文列表失败: {str(e)}")
            return []

    async def _parse_ndss_papers(self, year: str) -> List[Dict[str, Any]]:
        """解析 NDSS 论文列表"""
        papers = []
        full_year = f"20{year}" if len(year) == 2 else year
        
        try:
            print(f"📚 正在获取 NDSS {full_year} 论文列表...")
            conf_info = self.CONFERENCE_INFO['ndss']
            base_url = f"{conf_info['base_url']}/ndss{year}/accepted-papers"
            
            # 使用会话发送请求
            if not self.session:
                self.session = AsyncSession()
                
            response = await self.session.get(
                base_url,
                headers=conf_info['headers'],
                impersonate="chrome110"
            )
            
            if response.status_code != 200:
                raise Exception(f"获取会议页面失败: HTTP {response.status_code}")
                
            # 解析页面内容
            soup = BeautifulSoup(response.text, 'html.parser')
            paper_items = soup.select('div.paper-item, div.accepted-paper')
            
            for item in paper_items:
                title_elem = item.select_one('h3.paper-title, h4.paper-title, div.paper-title')
                if not title_elem:
                    continue
                    
                title = title_elem.text.strip()
                url = item.select_one('a[href*=".pdf"]')
                if not url:
                    continue
                    
                pdf_url = url['href']
                if not pdf_url.startswith('http'):
                    pdf_url = f"{conf_info['base_url']}{pdf_url}"
                    
                papers.append({
                    'title': title,
                    'url': pdf_url
                })
                
            print(f"✅ 找到 {len(papers)} 篇论文")
            return papers
            
        except Exception as e:
            print(f"❌ 获取 NDSS {full_year} 论文列表失败: {str(e)}")
            return []

    async def _parse_usenix_papers(self, year: str) -> List[Dict[str, Any]]:
        """解析 USENIX Security 论文列表"""
        papers = []
        full_year = f"20{year}" if len(year) == 2 else year
        
        try:
            print(f"📚 正在获取 USENIX Security {full_year} 论文列表...")
            conf_info = self.CONFERENCE_INFO['usenix']
            base_url = f"{conf_info['base_url']}{full_year}/technical-sessions"
            
            # 使用会话发送请求
            if not self.session:
                self.session = AsyncSession()
                
            response = await self.session.get(
                base_url,
                headers=conf_info['headers'],
                impersonate="chrome110"
            )
            
            if response.status_code != 200:
                raise Exception(f"获取会议页面失败: HTTP {response.status_code}")
                
            # 解析页面内容
            soup = BeautifulSoup(response.text, 'html.parser')
            paper_items = soup.select('div.paper-item, div.node-paper')
            
            for item in paper_items:
                title_elem = item.select_one('h2.node-title, div.field-title')
                if not title_elem:
                    continue
                    
                title = title_elem.text.strip()
                url = item.select_one('a[href*=".pdf"]')
                if not url:
                    continue
                    
                pdf_url = url['href']
                if not pdf_url.startswith('http'):
                    pdf_url = f"https://www.usenix.org{pdf_url}"
                    
                papers.append({
                    'title': title,
                    'url': pdf_url
                })
                
            print(f"✅ 找到 {len(papers)} 篇论文")
            return papers
            
        except Exception as e:
            print(f"❌ 获取 USENIX Security {full_year} 论文列表失败: {str(e)}")
            return []

    async def get_conference_papers(self, conference: str, year: str) -> List[Dict[str, Any]]:
        """获取会议论文列表 - 带进度显示"""
        try:
            conf_info = self.CONFERENCE_INFO.get(conference)
            if not conf_info and conference != 'ccs':
                raise ValueError(f"不支持的会议: {conference}")

            papers = []
            print(f"🔍 正在获取 {conference.upper()} {year} 论文列表...")

            # 根据会议类型选择相应的解析方法
            if conference == 'ccs':
                papers = await self._parse_ccs_papers(self.conference_urls[conference], year)
            elif conference == 'sp':
                papers = await self._parse_sp_papers(year)
            elif conference == 'ndss':
                papers = await self._parse_ndss_papers(year)
            elif conference == 'usenix':
                papers = await self._parse_usenix_papers(year)

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
            self.logger.error(f"获取论文列表失败: {str(e)}")
            raise
        """获取会议论文列表 - 带进度显示"""
        try:
            if conference not in self.conference_urls:
                raise ValueError(f"不支持的会议: {conference}")

            base_url = self.conference_urls[conference]
            papers = []
            
            print(f"🔍 正在获取 {conference.upper()} {year} 论文列表...")

            # 规范化年份格式
            year = self.year_formats[conference](year)

            # 根据会议类型选择相应的解析方法
            papers = await self._get_papers_by_conference(conference, base_url, year)
            if papers:
                print(f"✨ 成功获取 {len(papers)} 篇论文信息")
            return papers

        except Exception as e:
            self.logger.error(f"获取论文列表失败: {e}")
            raise

    async def _get_papers_by_conference(self, conference: str, base_url: str, year: str) -> List[Dict[str, Any]]:
        """根据会议类型获取论文列表"""
        try:
            if conference == 'sp':
                # IEEE S&P
                full_url = f"{base_url}{year}"
                return await self._get_sp_papers(full_url, year)
            elif conference == 'ndss':
                # NDSS
                return await self._get_ndss_papers(base_url, year)
            elif conference == 'usenix':
                # USENIX Security
                return await self._get_usenix_papers(base_url, year)
            elif conference == 'ccs':
                # ACM CCS
                return await self._get_ccs_papers(base_url, year)
            else:
                raise ValueError(f"不支持的会议: {conference}")
        except Exception as e:
            self.logger.error(f"获取{conference.upper()} {year}论文列表失败: {e}")
            raise

    async def _get_sp_papers(self, base_url: str, year: str) -> List[Dict[str, Any]]:
        """获取 IEEE S&P 论文列表"""
        papers = []
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/94.0.4606.81'
            }
            async with self.session.get(base_url, headers=headers) as response:
                if response.status_code != 200:
                    raise Exception(f"Failed to fetch SP {year} papers list")

                soup = BeautifulSoup(response.text, 'lxml')
                paper_items = soup.find_all('div', class_='article-list__item')

                for item in paper_items:
                    title_elem = item.find('h3', class_='article-list__title')
                    if not title_elem:
                        continue

                    title = title_elem.text.strip()
                    pdf_link = item.find('a', class_='pdf-link')
                    
                    if pdf_link and 'href' in pdf_link.attrs:
                        url = pdf_link['href']
                        if not url.startswith('http'):
                            url = f"https://www.computer.org{url}"
                        papers.append({
                            'title': title,
                            'url': url
                        })

        except Exception as e:
            self.logger.error(f"解析 SP {year} 论文列表失败: {e}")
            raise

        return papers

    async def _get_ndss_papers(self, base_url: str, year: str) -> List[Dict[str, Any]]:
        """获取 NDSS 论文列表"""
        papers = []
        try:
            url = f"{base_url}ndss{year}/accepted-papers"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/94.0.4606.81'
            }
            async with self.session.get(url, headers=headers) as response:
                if response.status_code != 200:
                    raise Exception(f"Failed to fetch NDSS {year} papers list")

                soup = BeautifulSoup(response.text, 'lxml')
                paper_items = soup.find_all('div', class_='paper-item')

                for item in paper_items:
                    title_elem = item.find('h2', class_='title')
                    if not title_elem:
                        continue

                    title = title_elem.text.strip()
                    pdf_link = item.find('a', href=lambda x: x and x.endswith('.pdf'))
                    
                    if pdf_link and 'href' in pdf_link.attrs:
                        url = pdf_link['href']
                        if not url.startswith('http'):
                            url = f"https://www.ndss-symposium.org{url}"
                        papers.append({
                            'title': title,
                            'url': url
                        })

        except Exception as e:
            self.logger.error(f"解析 NDSS {year} 论文列表失败: {e}")
            raise

        return papers

    async def _get_usenix_papers(self, base_url: str, year: str) -> List[Dict[str, Any]]:
        """获取 USENIX Security 论文列表"""
        papers = []
        try:
            url = f"{base_url}{year}/technical-sessions"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/94.0.4606.81'
            }
            async with self.session.get(url, headers=headers) as response:
                if response.status_code != 200:
                    raise Exception(f"Failed to fetch USENIX {year} papers list")

                soup = BeautifulSoup(response.text, 'lxml')
                paper_items = soup.find_all('div', class_='node-paper')

                for item in paper_items:
                    title_elem = item.find('h2', class_='node-title')
                    if not title_elem:
                        continue

                    title = title_elem.text.strip()
                    pdf_link = item.find('a', href=lambda x: x and x.endswith('.pdf'))
                    
                    if pdf_link and 'href' in pdf_link.attrs:
                        url = pdf_link['href']
                        if not url.startswith('http'):
                            url = f"https://www.usenix.org{url}"
                        papers.append({
                            'title': title,
                            'url': url
                        })

        except Exception as e:
            self.logger.error(f"解析 USENIX {year} 论文列表失败: {e}")
            raise

        return papers

    async def get_papers(self, conference: str, year: str) -> List[Dict[str, Any]]:
        """
        获取指定会议和年份的论文列表
        """
        try:
            base_url = self.conference_urls.get(conference, {}).get(year)
            if not base_url:
                self.logger.error(f"未找到 {conference} {year} 的URL配置")
                return []

            if conference == 'ccs':
                papers = await self._parse_ccs_papers(base_url, year)
            elif conference == 'sp':
                papers = await self._parse_sp_papers(base_url, year)
            elif conference == 'ndss':
                papers = await self._parse_ndss_papers(base_url, year)
            elif conference == 'usenix':
                papers = await self._parse_usenix_papers(base_url, year)
            else:
                self.logger.error(f"不支持的会议类型: {conference}")
                return []
            
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



    async def _parse_ccs_papers(self, base_url: str, year: str) -> List[Dict[str, Any]]:
        """
        解析CCS论文列表的主入口函数。
        使用持久化会话来执行所有相关请求。
        
        Args:
            base_url: 论文列表的基础URL
            year: 会议年份
            
        Returns:
            论文信息列表
        """
        papers = []
        try:
            if not self.session:
                raise RuntimeError("持久化会话未初始化。请在 'async with' 块中使用 PaperDownloader。")
                
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            paper_dois = await self._get_all_ccs_dois_from_proceedings_page(self.session, year)
            if not paper_dois:
                self.logger.error(f"❌ 未能为CCS {year} 获取任何论文的DOI。")
                return []
            
            self.logger.info(f"📚 开始通过API批量解析 {len(paper_dois)} 篇CCS论文的详细信息...")
            
            papers = await self._fetch_all_ccs_paper_details_via_api(self.session, paper_dois, year)
            return papers
            
        except Exception as e:
            self.logger.error(f"❌ CCS论文解析主流程错误: {str(e)}")
            raise

    async def _fetch_all_ccs_paper_details_via_api(self, session: AsyncSession, dois: List[str], year: str) -> List[Dict[str, Any]]:
        """
        使用POST请求批量获取所有CCS论文的JSON数据并解析。
        """
        api_url = "https://dl.acm.org/action/exportCiteProcCitation"
        headers = {
            'Host': 'dl.acm.org',
            'Cookie': '_cf_bm=12; _cfuvid=eKvDTOvVWyHDD5bNf_GLEG_fzdrvwq1g_7YIL.aZOJU-1756624678973-0.0.1.1-604800000',
            'Pragma': 'no-cache',
            'Accept': '*/*',
            'Dnt': '1',
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36 Edg/139.0.0.0',
            'Sec-Ch-Ua-Platform-Version': '"19.0.0"',
            'Origin': 'https://dl.acm.org',
            'Referer': 'https://dl.acm.org/doi/proceedings/10.1145/3658644',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
            'Priority': 'u=1, i',
        }
        # 只保留最后一段有点的DOI（即真正论文）
        filtered_dois = []
        for doi in dois:
            last = doi.split('/')[-1]
            if '.' in last:
                filtered_dois.append(doi)
        if not filtered_dois:
            self.logger.error("❌ 没有合格的DOI可用于API请求，全部被过滤。")
            return []
        dois_payload = ",".join(filtered_dois)
        data_string = f"dois={dois_payload}&targetFile=custom-bibtex&format=json"
        content_length = str(len(data_string.encode('utf-8')))
        headers['Content-Length'] = content_length
        try:
            self.logger.info(f"🚀 正在向 {api_url} 发送单次POST请求以获取 {len(dois)} 篇论文的JSON数据...")
            response = await session.post(
                api_url,
                data=data_string,
                headers=headers,
                impersonate="chrome110",
                timeout=180
            )
            if response.status_code != 200:
                self.logger.error(f"❌ 批量获取JSON失败，HTTP状态码: {response.status_code}")
                self.logger.error(f"响应内容: {response.text[:500]}")
                debug_file = Path(f"debug_ccs_api_error_{response.status_code}.html")
                debug_file.write_text(response.text, encoding='utf-8')
                self.logger.info(f"🐛 已将错误响应保存到 {debug_file.absolute()} 以供调试。")
                return []
            json_data_str = response.text
            self.logger.info(f"✅ 成功获取JSON数据，大小: {len(json_data_str)}字节。开始解析...")
            debug_json_file = Path("debug_ccs_json_response.json")
            debug_json_file.write_text(json_data_str, encoding='utf-8')
            self.logger.info(f"🐛 已将原始JSON响应保存到 {debug_json_file.absolute()} 以供分析。")
            papers = self._parse_json_data(json_data_str, year)
            self.logger.info(f"✅ 成功解析 {len(papers)}/{len(dois)} 篇论文的元数据。")
            return papers
        except Exception as e:
            self.logger.error(f"❌ 批量获取和解析JSON数据时发生严重错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def _parse_json_data(self, json_data_str: str, year: str) -> List[Dict[str, Any]]:
        """
        解析从ACM API返回的JSON数据。
        """
        papers = []
        try:
            data = json.loads(json_data_str)
            for item_dict in data.get('items', []):
                for doi, details in item_dict.items():
                    try:
                        title = details.get('title', f"未知标题 (DOI: {doi})")
                        authors_list = []
                        for author_info in details.get('author', []):
                            given_name = author_info.get('given', '')
                            family_name = author_info.get('family', '')
                            authors_list.append(f"{given_name} {family_name}".strip())
                        abstract = details.get('abstract', '摘要不可用')
                        pdf_url = f"https://dl.acm.org/doi/pdf/{doi}"
                        papers.append({
                            'title': title,
                            'authors': authors_list,
                            'abstract': abstract,
                            'url': pdf_url,
                            'conference': "CCS",
                            'year': year
                        })
                    except Exception as e:
                        self.logger.warning(f"解析单个JSON条目时出错 (DOI: {doi}): {e}")
                        continue
            return papers
        except json.JSONDecodeError as e:
            self.logger.error(f"❌ JSON解析失败: {e}")
            return []
        except Exception as e:
            self.logger.error(f"❌ 处理JSON数据时发生未知错误: {e}")
            return []
  

    def _parse_bibtex_data(self, bibtex_data: str, year: str) -> List[Dict[str, Any]]:
        """
        解析BibTeX数据字符串并返回论文列表。
        使用更健壮的正则表达式来处理复杂的BibTeX格式。
        """
        papers = []
        # 使用更可靠的方式分割条目：按换行符后的'@'分割
        entries = re.split(r'\n@', bibtex_data)
        
        for entry in entries:
            if not entry.strip() or not entry.startswith('inproceedings'):
                continue

            try:
                # 健壮的DOI提取
                doi_match = re.search(r'doi\s*=\s*\{([^}]+)\}', entry, re.IGNORECASE)
                doi = doi_match.group(1).strip() if doi_match else "未知DOI"

                # 健壮的标题提取，能处理嵌套花括号
                title_match = re.search(r'title\s*=\s*\{((?:[^{}]|\{[^{}]*\})+)\}', entry, re.IGNORECASE)
                title = title_match.group(1).strip().replace("{", "").replace("}", "") if title_match else f"未知标题 (DOI: {doi})"

                # 健壮的作者提取
                author_match = re.search(r'author\s*=\s*\{([^}]+)\}', entry, re.IGNORECASE)
                authors_str = author_match.group(1) if author_match else ""
                authors_list = [name.strip().replace("{", "").replace("}", "") for name in authors_str.split(' and ')]

                abstract = "摘要需访问论文页面查看"
                pdf_url = f"https://dl.acm.org/doi/pdf/{doi}"

                papers.append({
                    'title': title,
                    'authors': authors_list,
                    'abstract': abstract,
                    'url': pdf_url,
                    'conference': "CCS",
                    'year': year
                })
            except Exception as e:
                self.logger.warning(f"解析单个BibTeX条目时出错: {e}\n条目内容: {entry[:300]}...")
                continue
        return papers

    async def _get_all_ccs_dois_from_proceedings_page(self, session: AsyncSession, year: str) -> Optional[List[str]]:
        """
        获取CCS会议指定年份所有论文的DOI列表。
        使用传入的持久化会话。
        """
        if not CURL_CFFI_AVAILABLE:
            self.logger.error("❌ curl_cffi 未安装，无法执行CCS论文抓取。")
            return None

        try:
            short_year_str = f"'{year}"
            full_year_str = f"20{year}" if len(year) == 2 else year
            
            proceedings_list_url = 'https://dl.acm.org/conference/ccs/proceedings'
            self.logger.info(f"🌐 正在通过持久化会话访问CCS会议列表页面: {proceedings_list_url}")

            # 使用更完整的请求头来模拟浏览器行为
            headers = {
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Cache-Control': 'max-age=0',
                'Connection': 'keep-alive',
                'DNT': '1',
                'Host': 'dl.acm.org',
                'Sec-Ch-Ua': '"Chromium";v="116", "Not)A;Brand";v="24", "Microsoft Edge";v="116"',
                'Sec-Ch-Ua-Mobile': '?0',
                'Sec-Ch-Ua-Platform': '"Windows"',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Upgrade-Insecure-Requests': '1',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36 Edg/116.0.0.0'
            }

            # 尝试多次，使用不同的浏览器模拟配置
            for browser in ["chrome110", "chrome99", "chrome100", "safari15_3"]:
                try:
                    response = await session.get(
                        proceedings_list_url,
                        impersonate=browser,
                        headers=headers,
                        timeout=45
                    )
                    
                    if response.status_code == 200:
                        self.logger.info(f"✅ 使用 {browser} 成功访问")
                        break
                    else:
                        self.logger.warning(f"使用 {browser} 失败，HTTP状态码: {response.status_code}")
                except Exception as e:
                    self.logger.warning(f"使用 {browser} 时出错: {str(e)}")
                    await asyncio.sleep(2)  # 失败后等待一下再重试
                    continue
            
            if response.status_code != 200:
                self.logger.error(f"❌ 访问CCS会议列表页面失败，HTTP状态码: {response.status_code}")
                return None
            
            self.logger.info("✅ 成功获取会议列表页面。")
            soup = BeautifulSoup(response.text, 'html.parser')
            
            target_proc_url = None
            proc_items = soup.select('li.conference__proceedings div.conference__title a')
            for item in proc_items:
                link_text = item.get_text(strip=True)
                if short_year_str in link_text or full_year_str in link_text:
                    target_proc_url = urllib.parse.urljoin(proceedings_list_url, item['href'])
                    self.logger.info(f"✅ 找到 CCS {year} 会议录链接: {target_proc_url}")
                    break
            
            if not target_proc_url:
                self.logger.error(f"❌ 未能在页面上找到 CCS {year} 的会议录链接。")
                debug_file = Path("debug_acm_proceedings_list.html")
                debug_file.write_text(response.text, encoding='utf-8')
                self.logger.info(f"🐛 已将会议列表页面内容保存到 {debug_file.absolute()} 以供调试。")
                return None

            self.logger.info(f"🌐 正在访问 CCS {year} 论文列表页面...")
            response = await session.get(target_proc_url, impersonate="chrome110", timeout=45)

            if response.status_code != 200:
                self.logger.error(f"❌ 访问 CCS {year} 论文列表页面失败，HTTP状态码: {response.status_code}")
                return None
            
            self.logger.info(f"✅ 成功获取 CCS {year} 论文列表页面。")
            soup = BeautifulSoup(response.text, 'html.parser')
            
            all_dois = []
            # 从隐藏的input中提取所有DOI
            doi_inputs = soup.select('input.section--dois')
            for doi_input in doi_inputs:
                dois_str = doi_input.get('value', '')
                if dois_str:
                    all_dois.extend(dois_str.split(','))

            if not all_dois:
                self.logger.warning(f"未能在 CCS {year} 页面提取到任何论文DOI。请检查页面结构是否已更改。")
                debug_file = Path(f"debug_ccs_{year}_papers.html")
                debug_file.write_text(response.text, encoding='utf-8')
                self.logger.info(f"🐛 已将论文列表页面内容保存到 {debug_file.absolute()} 以供调试。")
                return None
            
            # 去重并清洗
            unique_dois = sorted(list(set(doi.strip() for doi in all_dois if doi.strip())))
            self.logger.info(f"✅ 成功提取 {len(unique_dois)} 个唯一的 CCS {year} 论文DOI。")
            return unique_dois

        except Exception as e:
            self.logger.error(f"❌ 获取CCS论文DOI时发生严重错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


