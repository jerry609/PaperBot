"""
通用 API 客户端封装
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Any, Optional

try:
    import aiohttp
    from aiohttp import ClientTimeout
except ImportError:
    aiohttp = None
    ClientTimeout = None

logger = logging.getLogger(__name__)


class APIClient:
    """通用异步 HTTP API 客户端"""
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
        request_interval: float = 1.0,
    ):
        if aiohttp is None:
            raise RuntimeError("需要安装 aiohttp 库: pip install aiohttp")
        
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = ClientTimeout(total=timeout)
        self.request_interval = request_interval
        self._last_request_time = 0.0
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建 aiohttp session"""
        if self._session is None or self._session.closed:
            headers = {"User-Agent": "PaperBot/1.0"}
            if self.api_key:
                headers["x-api-key"] = self.api_key
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                headers=headers,
            )
        return self._session
    
    async def _wait_for_rate_limit(self):
        """等待以遵守速率限制"""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.request_interval:
            await asyncio.sleep(self.request_interval - elapsed)
        self._last_request_time = time.time()
    
    async def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """发送 GET 请求"""
        await self._wait_for_rate_limit()
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        session = await self._get_session()
        
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 404:
                    logger.warning(f"Resource not found: {url}")
                    return {}
                elif response.status == 429:
                    logger.warning(f"Rate limit exceeded for {url}")
                    # 等待更长时间后重试一次
                    await asyncio.sleep(5)
                    async with session.get(url, params=params) as retry_response:
                        if retry_response.status == 200:
                            return await retry_response.json()
                        raise Exception(f"Rate limit still exceeded: {retry_response.status}")
                else:
                    text = await response.text()
                    logger.error(f"API error {response.status}: {text[:200]}")
                    raise Exception(f"API error: {response.status}")
        except asyncio.TimeoutError:
            logger.error(f"Request timeout: {url}")
            raise
        except Exception as e:
            logger.error(f"Request failed: {url} - {e}")
            raise
    
    async def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """发送 POST 请求"""
        await self._wait_for_rate_limit()
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        session = await self._get_session()
        
        try:
            async with session.post(url, data=data, json=json_data) as response:
                if response.status in (200, 201):
                    return await response.json()
                else:
                    text = await response.text()
                    logger.error(f"API error {response.status}: {text[:200]}")
                    raise Exception(f"API error: {response.status}")
        except asyncio.TimeoutError:
            logger.error(f"Request timeout: {url}")
            raise
        except Exception as e:
            logger.error(f"Request failed: {url} - {e}")
            raise
    
    async def close(self):
        """关闭 session"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

