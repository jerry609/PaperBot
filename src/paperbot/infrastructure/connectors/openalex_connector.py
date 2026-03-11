from __future__ import annotations

from typing import Any, Dict, List, Optional
from urllib.parse import quote

from paperbot.domain.paper_identity import normalize_arxiv_id, normalize_doi
from paperbot.infrastructure.crawling.request_layer import AsyncRequestLayer, RequestPolicy


class OpenAlexConnector:
    """Minimal OpenAlex work lookup for graph-style discovery expansion."""

    def __init__(
        self,
        *,
        timeout_s: float = 30.0,
        base_url: str = "https://api.openalex.org",
        request_layer: Optional[AsyncRequestLayer] = None,
    ):
        self.timeout_s = timeout_s
        self.base_url = base_url.rstrip("/")
        self._headers = {"User-Agent": "PaperBot/2.0"}
        self._request = request_layer or AsyncRequestLayer(RequestPolicy(timeout_s=timeout_s))

    async def resolve_work(self, *, seed_type: str, seed_id: str) -> Optional[Dict[str, Any]]:
        seed_type = str(seed_type or "").strip().lower()
        seed_id = str(seed_id or "").strip()
        if not seed_id:
            return None

        if seed_type == "openalex":
            return await self.get_work(seed_id)
        if seed_type == "doi":
            return await self._search_one(f"doi:{normalize_doi(seed_id) or seed_id}")
        if seed_type == "arxiv":
            arxiv_id = normalize_arxiv_id(seed_id) or seed_id
            return await self._search_one(
                f"locations.landing_page_url:https://arxiv.org/abs/{arxiv_id}"
            )
        if seed_type == "semantic_scholar":
            return await self._search_one(f"ids.semantic_scholar:{seed_id}")
        return None

    async def get_work(self, work_id: str) -> Optional[Dict[str, Any]]:
        resolved = self._normalize_work_id(work_id)
        if not resolved:
            return None
        url = f"{self.base_url}/works/{quote(resolved)}"
        payload = await self._request.get_json(url, headers=self._headers)
        return payload if isinstance(payload, dict) else None

    async def get_related_works(
        self, work: Dict[str, Any], *, limit: int = 20
    ) -> List[Dict[str, Any]]:
        ids = []
        for item in work.get("related_works") or []:
            if isinstance(item, str):
                ids.append(item)
            if len(ids) >= max(1, int(limit)):
                break
        return await self.get_works_by_ids(ids, limit=limit)

    async def get_referenced_works(
        self, work: Dict[str, Any], *, limit: int = 20
    ) -> List[Dict[str, Any]]:
        ids = []
        for item in work.get("referenced_works") or []:
            if isinstance(item, str):
                ids.append(item)
            if len(ids) >= max(1, int(limit)):
                break
        return await self.get_works_by_ids(ids, limit=limit)

    async def get_citing_works(
        self, work: Dict[str, Any], *, limit: int = 20
    ) -> List[Dict[str, Any]]:
        cited_by_api_url = str(work.get("cited_by_api_url") or "").strip()
        if not cited_by_api_url:
            return []
        payload = await self._request.get_json(
            cited_by_api_url,
            headers=self._headers,
            params={"per-page": max(1, min(int(limit), 200))},
        )
        if not isinstance(payload, dict):
            return []
        rows = payload.get("results")
        return rows if isinstance(rows, list) else []

    async def get_works_by_ids(self, ids: List[str], *, limit: int = 20) -> List[Dict[str, Any]]:
        normalized_ids: List[str] = []
        for raw_id in ids[: max(1, int(limit))]:
            normalized = self._normalize_work_id(raw_id)
            if normalized:
                normalized_ids.append(normalized)
        if not normalized_ids:
            return []

        results: List[Dict[str, Any]] = []
        chunk_size = 50
        for idx in range(0, len(normalized_ids), chunk_size):
            batch = normalized_ids[idx : idx + chunk_size]
            payload = await self._request.get_json(
                f"{self.base_url}/works",
                headers=self._headers,
                params={
                    "filter": f"openalex_id:{'|'.join(batch)}",
                    "per-page": len(batch),
                },
            )
            if not isinstance(payload, dict):
                continue
            rows = payload.get("results")
            if not isinstance(rows, list):
                continue

            by_id: Dict[str, Dict[str, Any]] = {}
            for row in rows:
                if not isinstance(row, dict):
                    continue
                normalized = self._normalize_work_id(str(row.get("id") or ""))
                if normalized:
                    by_id[normalized] = row

            for batch_id in batch:
                row = by_id.get(batch_id)
                if row:
                    results.append(row)

        return results

    async def _search_one(self, filter_expr: str) -> Optional[Dict[str, Any]]:
        payload = await self._request.get_json(
            f"{self.base_url}/works",
            headers=self._headers,
            params={"filter": filter_expr, "per-page": 1},
        )
        if not isinstance(payload, dict):
            return None
        rows = payload.get("results")
        if not isinstance(rows, list) or not rows:
            return None
        first = rows[0]
        return first if isinstance(first, dict) else None

    @staticmethod
    def _normalize_work_id(value: str) -> Optional[str]:
        text = str(value or "").strip()
        if not text:
            return None
        lowered = text.lower()
        marker = "openalex.org/"
        idx = lowered.find(marker)
        if idx >= 0:
            text = text[idx + len(marker) :]
        text = text.strip().strip("/")
        if not text:
            return None
        if text[0].upper() != "W":
            text = f"W{text}"
        return text.upper()

    async def close(self) -> None:
        await self._request.close()
