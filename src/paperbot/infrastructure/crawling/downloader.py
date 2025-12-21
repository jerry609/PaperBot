from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from paperbot.infrastructure.crawling.request_layer import AsyncRequestLayer, RequestPolicy


@dataclass
class DownloadResult:
    success: bool
    path: Optional[str] = None
    cached: bool = False
    size: int = 0
    error: Optional[str] = None


class PdfDownloader:
    def __init__(self, download_dir: Path, policy: Optional[RequestPolicy] = None):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.request = AsyncRequestLayer(policy)

    async def download(self, url: str, filename: str) -> DownloadResult:
        try:
            path = self.download_dir / filename
            if path.exists() and path.stat().st_size > 1024:
                return DownloadResult(success=True, path=str(path), cached=True, size=path.stat().st_size)

            data = await self.request.get_bytes(url)
            if len(data) < 1024:
                return DownloadResult(success=False, error=f"content too small: {len(data)}")
            path.write_bytes(data)
            return DownloadResult(success=True, path=str(path), cached=False, size=len(data))
        except Exception as e:
            return DownloadResult(success=False, error=str(e))


