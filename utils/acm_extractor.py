#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版ACM论文提取器
专为downloader模块设计
"""

import cloudscraper
import time
import random
import urllib3
import json
import gzip
import io
import brotli
from bs4 import BeautifulSoup
import re

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class ACMPaperExtractor:
    """简化版ACM论文提取器"""
    
    def __init__(self):
        self.base_url = "https://dl.acm.org"
        self.scraper = None
        self._init_scraper()
        
    def _init_scraper(self):
        """初始化cloudscraper"""
        self.scraper = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'windows',
                'mobile': False
            },
            delay=10
        )
        
        # 设置基础headers
        self.scraper.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36 Edg/139.0.0.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
            'DNT': '1',
            'sec-ch-ua': '"Microsoft Edge";v="139", "Chromium";v="139", "Not A(Brand";v="99"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
        })
    
    def get_homepage(self):
        """访问主页获取cookies"""
        print("🏠 访问ACM主页获取cookies...")
        
        try:
            response = self.scraper.get(self.base_url, timeout=30)
            print(f"主页状态码: {response.status_code}")
            
            if response.status_code == 200:
                print("✅ 成功获取主页cookies")
                return True
            else:
                print(f"❌ 获取主页失败: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ 访问主页异常: {str(e)}")
            return False
    
    def get_proceedings_page(self, proceedings_doi):
        """访问proceedings页面获取cookies和必要的上下文"""
        print(f"📂 访问CCS proceedings页面...")
        
        if not self.get_homepage():
            return None
        
        url = f"{self.base_url}/doi/proceedings/{proceedings_doi}"
        
        # 添加延迟
        time.sleep(random.uniform(2, 5))
        
        try:
            response = self.scraper.get(url, timeout=30)
            print(f"Proceedings页面状态码: {response.status_code}")
            
            if response.status_code == 200:
                print("✅ 成功访问proceedings页面")
                return response.text
            else:
                print(f"❌ 访问proceedings页面失败: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ 访问proceedings页面异常: {str(e)}")
            return None
    
    def extract_all_paper_dois(self, proceedings_content):
        """从proceedings页面内容中提取所有论文DOI"""
        print("🔍 从proceedings页面提取所有论文DOI...")
        
        if not proceedings_content:
            print("❌ proceedings页面内容为空")
            return []
        
        try:
            soup = BeautifulSoup(proceedings_content, 'html.parser')
            dois = []
            
            # 查找所有论文链接
            paper_links = soup.find_all('a', href=re.compile(r'/doi/10\.1145/'))
            
            for link in paper_links:
                href = link.get('href', '')
                # 从href中提取DOI
                doi_match = re.search(r'/doi/([^/]+/[^/?]+)', href)
                if doi_match:
                    doi = doi_match.group(1)
                    # 确保DOI格式正确且不重复
                    if doi.startswith('10.1145/') and doi not in dois:
                        dois.append(doi)
            
            print(f"✅ 成功提取到 {len(dois)} 个论文DOI")
            return dois
            
        except Exception as e:
            print(f"❌ 提取论文DOI时出错: {str(e)}")
            return []
    
    def export_citations(self, doi_list, proceedings_doi):
        """使用export citation API导出引用信息"""
        print("📚 使用export citation API导出引用信息...")
        
        # 确保已访问主页和proceedings页面
        self.get_proceedings_page(proceedings_doi)
        
        # 构建API URL
        api_url = f"{self.base_url}/action/exportCiteProcCitation"
        
        # 格式化DOI列表
        formatted_doi_list = []
        for doi in doi_list:
            if doi.startswith('10.1145/'):
                formatted_doi_list.append(doi)
            else:
                formatted_doi_list.append(f"10.1145/{doi}")
        
        # 构建请求数据
        data = {
            'dois': ','.join(formatted_doi_list),
            'targetFile': 'custom-bibtex',
            'format': 'bibTex'
        }
        
        # 设置API请求headers
        api_headers = {
            'X-Requested-With': 'XMLHttpRequest',
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'Accept': '*/*',
            'Origin': self.base_url,
            'Referer': f"{self.base_url}/doi/proceedings/{proceedings_doi}",
            'Sec-Fetch-Site': 'same-origin',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Dest': 'empty',
            'DNT': '1',
            'sec-ch-ua': '"Microsoft Edge";v="139", "Chromium";v="139", "Not A(Brand";v="99"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
        }
        
        # 更新请求头
        self.scraper.headers.update(api_headers)
        
        # 添加延迟
        time.sleep(random.uniform(2, 4))
        
        try:
            response = self.scraper.post(api_url, data=data, timeout=30)
            print(f"API状态码: {response.status_code}")
            
            if response.status_code == 200:
                print("✅ API请求成功")
                
                # 处理响应内容
                content = None
                content_encoding = response.headers.get('Content-Encoding', '').lower()
                
                # 根据编码类型解压
                if 'br' in content_encoding:
                    print("🔍 检测到Brotli压缩，正在解压...")
                    try:
                        content = brotli.decompress(response.content).decode('utf-8')
                        print("✅ Brotli解压成功")
                    except Exception as e:
                        print(f"⚠️ Brotli解压失败: {str(e)}")
                        content = response.text
                elif 'gzip' in content_encoding:
                    print("🔍 检测到gzip压缩，正在解压...")
                    try:
                        compressed_data = io.BytesIO(response.content)
                        with gzip.GzipFile(fileobj=compressed_data) as gzip_file:
                            content = gzip_file.read().decode('utf-8')
                        print("✅ gzip解压成功")
                    except Exception as e:
                        print(f"⚠️ gzip解压失败: {str(e)}")
                        content = response.text
                else:
                    content = response.text
                    print("📄 无压缩或未知压缩格式")
                
                # 解析JSON响应
                try:
                    citation_data = json.loads(content)
                    print("✅ 成功解析JSON响应")
                    return citation_data
                except Exception as e:
                    print(f"⚠️ JSON解析失败: {str(e)}")
                    return None
            else:
                print(f"❌ API请求失败: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ API请求异常: {str(e)}")
            return None
    
    def extract_paper_info(self, citation_data):
        """从引用数据中提取论文信息"""
        print("🔍 从引用数据中提取论文信息...")
        
        if not citation_data or 'items' not in citation_data:
            print("❌ 无效的引用数据")
            return []
        
        paper_info_list = []
        items = citation_data['items']
        
        print(f"  处理 {len(items)} 个条目...")
        
        for item in items:
            # 每个item是一个字典，键是DOI
            for doi, paper_data in item.items():
                try:
                    # 提取基本信息
                    title = paper_data.get('title', 'Unknown Title')
                    
                    # 提取作者
                    authors = paper_data.get('author', [])
                    author_names = []
                    for author in authors:
                        if 'family' in author and 'given' in author:
                            author_names.append(f"{author['given']} {author['family']}")
                        elif 'literal' in author:
                            author_names.append(author['literal'])
                    
                    # 提取其他信息
                    url = f"{self.base_url}/doi/{doi}"
                    pdf_url = f"{self.base_url}/doi/pdf/{doi}"
                    abstract = paper_data.get('abstract', '')
                    
                    # 创建论文信息字典
                    paper_info = {
                        'title': title,
                        'doi': doi,
                        'url': url,
                        'authors': author_names,
                        'pdf_url': pdf_url,
                        'abstract': abstract,
                        'publisher': paper_data.get('publisher', ''),
                        'isbn': paper_data.get('ISBN', ''),
                        'pages': paper_data.get('page', ''),
                        'keywords': paper_data.get('keyword', '')
                    }
                    
                    paper_info_list.append(paper_info)
                except Exception as e:
                    print(f"  ⚠️ 处理条目时出错: {str(e)}")
                    continue
        
        print(f"✅ 提取到 {len(paper_info_list)} 个论文信息")
        return paper_info_list