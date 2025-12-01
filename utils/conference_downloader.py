import asyncio
import aiohttp
import os
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re

class ConferenceDownloader:
    def __init__(self, config):
        self.base_url = ""
        self.download_path = Path(config.get('download_path', './papers'))
        self.download_path.mkdir(parents=True, exist_ok=True)
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()

    async def get_conference_papers(self, conference, year):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def sanitize_filename(self, filename):
        """清理文件名，移除无效字符"""
        return re.sub(r'[\\/*?:"<>|]', "", filename)

    async def download_paper(self, url, title, paper_index, total_papers):
        if not url or not isinstance(url, str):
            return {'success': False, 'error': 'Invalid URL'}

        sanitized_title = self.sanitize_filename(title)
        pdf_filename = self.download_path / f"{sanitized_title}.pdf"

        if pdf_filename.exists():
            print(f"[{paper_index + 1}/{total_papers}] 📄 '{sanitized_title}' 已存在 (缓存).")
            return {'success': True, 'cached': True, 'path': str(pdf_filename)}

        try:
            print(f"[{paper_index + 1}/{total_papers}] 📥 开始下载: {title}")
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=300)) as response:
                if response.status == 200:
                    content = await response.read()
                    with open(pdf_filename, 'wb') as f:
                        f.write(content)
                    return {'success': True, 'cached': False, 'path': str(pdf_filename), 'size': len(content)}
                else:
                    return {'success': False, 'error': f"HTTP status {response.status}"}
        except Exception as e:
            return {'success': False, 'error': str(e)}
