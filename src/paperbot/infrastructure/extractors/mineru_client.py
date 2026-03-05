"""MinerU Cloud API client for PDF figure extraction.

This client uses MinerU v4 async task API:
- POST /extract/task
- GET /extract/task/{task_id}

API constraints (as documented by MinerU Cloud):
- Remote file URL only (no direct upload)
- File size <= 200MB
- Page count <= 600 pages
- Some overseas hosts (e.g. github/aws) may timeout from MinerU side
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://mineru.net/api/v4"
_DEFAULT_TIMEOUT = 60.0
_DEFAULT_MODEL_VERSION = "vlm"
_DEFAULT_POLL_INTERVAL_SECONDS = 2.0
_DEFAULT_MAX_WAIT_SECONDS = 180.0

_UNSUPPORTED_HOST_HINTS = (
    "github.com",
    "githubusercontent.com",
    "amazonaws.com",
)


@dataclass
class Figure:
    """Extracted figure from a PDF document."""

    url: str
    caption: str = ""
    page: int = 0
    width: int = 0
    height: int = 0
    index: int = 0
    inline_data_url: str = ""

    @property
    def area(self) -> int:
        return self.width * self.height


class MineruClient:
    """Client for MinerU Cloud API figure extraction."""

    def __init__(
        self,
        *,
        api_key: str = "",
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT,
        cache_dir: str = "",
        cache_ttl_seconds: int = 24 * 3600,
        model_version: str = _DEFAULT_MODEL_VERSION,
        poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
        max_wait_seconds: float = _DEFAULT_MAX_WAIT_SECONDS,
    ):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        self._cache_ttl_seconds = max(0, int(cache_ttl_seconds))
        self._model_version = (model_version or _DEFAULT_MODEL_VERSION).strip() or _DEFAULT_MODEL_VERSION
        self._poll_interval_seconds = max(0.5, float(poll_interval_seconds))
        self._max_wait_seconds = max(5.0, float(max_wait_seconds))

    def extract_figures(self, pdf_url: str) -> List[Figure]:
        """Extract figures from a PDF URL via MinerU Cloud API.

        Returns an empty list if extraction fails or API key is not set.
        """
        if not self._api_key:
            logger.debug("MinerU API key not set, skipping figure extraction")
            return []

        normalized_url = (pdf_url or "").strip()
        if not normalized_url:
            return []

        cached = self._load_cached_figures(normalized_url)
        if cached is not None:
            return cached

        try:
            figures = self._call_extract(normalized_url)
            self._store_cached_figures(normalized_url, figures)
            return figures
        except Exception as exc:
            logger.warning("MinerU figure extraction failed: %s", exc)
            return []

    def _call_extract(self, pdf_url: str) -> List[Figure]:
        self._validate_source_url(pdf_url)

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=self._timeout) as client:
            task_id = self._create_task(client, headers=headers, pdf_url=pdf_url)
            detail = self._poll_until_done(client, headers=headers, task_id=task_id)

            # Some deployments may return figures directly in task payload.
            parsed = self._parse_figures(detail)
            if parsed:
                return parsed

            zip_url = str(detail.get("full_zip_url") or "").strip()
            if not zip_url:
                return []
            return self._extract_figures_from_zip(client, zip_url)

    def _validate_source_url(self, pdf_url: str) -> None:
        parsed = urlparse(pdf_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("MinerU source URL must be an absolute http(s) URL")

        host = (parsed.netloc or "").lower()
        if any(host == hint or host.endswith(f".{hint}") for hint in _UNSUPPORTED_HOST_HINTS):
            raise ValueError(
                "MinerU Cloud may timeout on github/aws URLs; provide a publicly "
                "accessible non-github/non-aws URL"
            )

    def _create_task(self, client: httpx.Client, *, headers: Dict[str, str], pdf_url: str) -> str:
        payload = {"url": pdf_url, "model_version": self._model_version}
        body = self._request_json(
            client,
            method="POST",
            url=f"{self._base_url}/extract/task",
            headers=headers,
            json_payload=payload,
        )
        task_id = str((body.get("data") or {}).get("task_id") or "").strip()
        if not task_id:
            raise RuntimeError(f"invalid MinerU task response: {body}")
        return task_id

    def _poll_until_done(
        self,
        client: httpx.Client,
        *,
        headers: Dict[str, str],
        task_id: str,
    ) -> Dict[str, Any]:
        deadline = time.time() + self._max_wait_seconds
        while time.time() <= deadline:
            body = self._request_json(
                client,
                method="GET",
                url=f"{self._base_url}/extract/task/{task_id}",
                headers=headers,
            )
            detail = body.get("data") or {}
            state = str(detail.get("state") or "").strip().lower()

            if state == "done":
                return detail
            if state == "failed":
                err_msg = str(detail.get("err_msg") or "task failed")
                raise RuntimeError(err_msg)

            time.sleep(self._poll_interval_seconds)

        raise TimeoutError(f"MinerU task timed out after {self._max_wait_seconds:.0f}s")

    def _request_json(
        self,
        client: httpx.Client,
        *,
        method: str,
        url: str,
        headers: Dict[str, str],
        json_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        response = client.request(method, url, headers=headers, json=json_payload)
        response.raise_for_status()

        body = response.json()
        if not isinstance(body, dict):
            raise RuntimeError("invalid MinerU response payload")

        code = body.get("code")
        if code is not None and int(code) != 0:
            msg = str(body.get("msg") or "unknown error")
            raise RuntimeError(f"MinerU API error code={code}: {msg}")

        return body

    def _extract_figures_from_zip(self, client: httpx.Client, zip_url: str) -> List[Figure]:
        """Download MinerU result zip and parse figures from generated markdown."""
        response = client.get(zip_url)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content), "r") as zf:
            md_members = [name for name in zf.namelist() if name.lower().endswith(".md")]
            if not md_members:
                return []

            # Prefer the shortest markdown path, usually the top-level document markdown.
            target_md = sorted(md_members, key=len)[0]
            markdown_text = zf.read(target_md).decode("utf-8", errors="ignore")
            return self._parse_figures_from_markdown(markdown_text, zip_url=zip_url, zip_file=zf)

    def _parse_figures_from_markdown(
        self,
        markdown_text: str,
        *,
        zip_url: str = "",
        zip_file: Optional[zipfile.ZipFile] = None,
    ) -> List[Figure]:
        """Parse figures from MinerU markdown output.

        Typical pattern:
            ![](images/xxx.jpg)
            Figure 1: ...
        """
        if not markdown_text.strip():
            return []

        lines = markdown_text.splitlines()
        figures: List[Figure] = []
        image_pattern = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
        caption_pattern = re.compile(r"^\s*(?:Figure|Fig\.?)\s*\d+\s*[:.-]", re.IGNORECASE)

        for idx, line in enumerate(lines):
            match = image_pattern.search(line)
            if not match:
                continue

            raw_ref = str(match.group(1) or "").strip()
            if not raw_ref:
                continue

            caption = ""
            for offset in (1, 2, 3):
                cursor = idx + offset
                if cursor >= len(lines):
                    break
                candidate = (lines[cursor] or "").strip()
                if not candidate:
                    continue
                if caption_pattern.match(candidate):
                    caption = candidate
                    break

            figure_url = self._resolve_figure_url(raw_ref=raw_ref, zip_url=zip_url)
            inline_data_url = self._build_inline_data_url(zip_file=zip_file, raw_ref=raw_ref)
            figures.append(
                Figure(
                    url=figure_url,
                    caption=caption,
                    index=len(figures),
                    inline_data_url=inline_data_url,
                )
            )

        return figures

    def _build_inline_data_url(
        self,
        *,
        zip_file: Optional[zipfile.ZipFile],
        raw_ref: str,
    ) -> str:
        if zip_file is None:
            return ""
        if raw_ref.startswith(("http://", "https://")):
            return ""

        target = raw_ref.strip().lstrip("/")
        if not target:
            return ""

        candidates = {target, f"./{target}"}
        if "/" in target:
            candidates.add(target.split("/", 1)[1])

        member_name = ""
        namelist = set(zip_file.namelist())
        for c in candidates:
            if c in namelist:
                member_name = c
                break
        if not member_name:
            return ""

        try:
            blob = zip_file.read(member_name)
        except Exception:
            return ""
        if not blob:
            return ""
        # Keep inline payload bounded to avoid very large report/cache artifacts.
        if len(blob) > 1_500_000:
            return ""

        ext = member_name.lower().rsplit(".", 1)[-1] if "." in member_name else ""
        mime = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "webp": "image/webp",
            "gif": "image/gif",
        }.get(ext)
        if not mime:
            return ""

        encoded = base64.b64encode(blob).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def _resolve_figure_url(self, *, raw_ref: str, zip_url: str) -> str:
        if raw_ref.startswith(("http://", "https://")):
            return raw_ref
        if zip_url:
            return f"{zip_url}#/{raw_ref}"
        return raw_ref

    def _cache_path(self, pdf_url: str) -> Optional[Path]:
        if self._cache_dir is None:
            return None
        key = hashlib.sha256(pdf_url.strip().encode("utf-8")).hexdigest()
        return self._cache_dir / f"{key}.json"

    def _load_cached_figures(self, pdf_url: str) -> Optional[List[Figure]]:
        cache_path = self._cache_path(pdf_url)
        if cache_path is None or not cache_path.exists():
            return None

        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return None

            saved_at = float(raw.get("saved_at") or 0)
            if self._cache_ttl_seconds > 0 and (time.time() - saved_at) > self._cache_ttl_seconds:
                return None

            rows = raw.get("figures") or []
            figures: List[Figure] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                figures.append(
                    Figure(
                        url=str(row.get("url") or "").strip(),
                        caption=str(row.get("caption") or "").strip(),
                        page=int(row.get("page") or 0),
                        width=int(row.get("width") or 0),
                        height=int(row.get("height") or 0),
                        index=int(row.get("index") or 0),
                        inline_data_url=str(row.get("inline_data_url") or "").strip(),
                    )
                )
            if figures:
                return figures
        except Exception:
            return None

        return None

    def _store_cached_figures(self, pdf_url: str, figures: List[Figure]) -> None:
        cache_path = self._cache_path(pdf_url)
        if cache_path is None:
            return
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "saved_at": time.time(),
                "figures": [
                    {
                        "url": figure.url,
                        "caption": figure.caption,
                        "page": figure.page,
                        "width": figure.width,
                        "height": figure.height,
                        "index": figure.index,
                        "inline_data_url": figure.inline_data_url,
                    }
                    for figure in figures
                ],
            }
            cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except Exception:
            return

    def _parse_figures(self, data: Dict[str, Any]) -> List[Figure]:
        figures: List[Figure] = []
        raw_figures = data.get("figures") or data.get("images") or []

        for idx, fig_data in enumerate(raw_figures):
            if not isinstance(fig_data, dict):
                continue
            url = str(fig_data.get("url") or fig_data.get("image_url") or "").strip()
            if not url:
                continue
            figures.append(
                Figure(
                    url=url,
                    caption=str(fig_data.get("caption") or "").strip(),
                    page=int(fig_data.get("page") or fig_data.get("page_number") or 0),
                    width=int(fig_data.get("width") or 0),
                    height=int(fig_data.get("height") or 0),
                    index=idx,
                )
            )
        return figures

    def identify_main_figure(self, figures: List[Figure]) -> Optional[Figure]:
        """Identify the most representative figure from a list.

        Heuristics:
        1. Prefer figures with captions containing keywords like "overview",
           "architecture", "framework", "pipeline", "main", "proposed"
        2. Among candidates, prefer larger figures (by area)
        3. Prefer figures from earlier pages (page 1-3)
        4. Filter out very small figures (icons, logos)
        """
        if not figures:
            return None

        # Filter out tiny figures (likely icons/logos)
        min_area = 10000  # ~100x100 pixels
        candidates = [f for f in figures if f.area >= min_area or f.area == 0]
        if not candidates:
            candidates = figures

        scored: List[tuple[float, Figure]] = []
        for fig in candidates:
            score = 0.0
            caption_lower = fig.caption.lower()

            main_keywords = [
                "overview",
                "architecture",
                "framework",
                "pipeline",
                "main",
                "proposed",
                "system",
                "model",
                "approach",
            ]
            for kw in main_keywords:
                if kw in caption_lower:
                    score += 10.0

            if re.search(r"(?:figure|fig\.?)\s*1\b", caption_lower):
                score += 15.0

            if 1 <= fig.page <= 3:
                score += 5.0

            if fig.area > 0:
                score += min(fig.area / 100000, 5.0)

            scored.append((score, fig))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored else None
