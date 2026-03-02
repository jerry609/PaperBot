"""MinerU Cloud API client for PDF figure extraction.

Uses the MinerU Cloud API (HTTP) to extract figures from PDFs.
Falls back gracefully when the service is unavailable.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://mineru.net/api/v4"
_DEFAULT_TIMEOUT = 60.0


@dataclass
class Figure:
    """Extracted figure from a PDF document."""

    url: str
    caption: str = ""
    page: int = 0
    width: int = 0
    height: int = 0
    index: int = 0

    @property
    def area(self) -> int:
        return self.width * self.height


class MineruClient:
    """Client for MinerU Cloud API figure extraction.

    Falls back gracefully when API is unavailable or API key is not set.
    """

    def __init__(
        self,
        *,
        api_key: str = "",
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT,
    ):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def extract_figures(self, pdf_url: str) -> List[Figure]:
        """Extract figures from a PDF URL via MinerU Cloud API.

        Returns an empty list if extraction fails or API is unavailable.
        """
        if not self._api_key:
            logger.debug("MinerU API key not set, skipping figure extraction")
            return []

        if not pdf_url or not pdf_url.strip():
            return []

        try:
            return self._call_extract(pdf_url)
        except Exception as exc:
            logger.warning("MinerU figure extraction failed: %s", exc)
            return []

    def _call_extract(self, pdf_url: str) -> List[Figure]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {"url": pdf_url, "extract_figures": True}

        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(
                f"{self._base_url}/extract",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

        return self._parse_figures(data)

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

        # Score each candidate
        scored: List[tuple[float, Figure]] = []
        for fig in candidates:
            score = 0.0
            caption_lower = fig.caption.lower()

            # Caption keyword bonus
            main_keywords = ["overview", "architecture", "framework", "pipeline",
                             "main", "proposed", "system", "model", "approach"]
            for kw in main_keywords:
                if kw in caption_lower:
                    score += 10.0

            # Figure 1 / Fig. 1 bonus
            if re.search(r"(?:figure|fig\.?)\s*1\b", caption_lower):
                score += 15.0

            # Early page bonus (pages 1-3)
            if 1 <= fig.page <= 3:
                score += 5.0

            # Larger figures get slight bonus
            if fig.area > 0:
                score += min(fig.area / 100000, 5.0)

            scored.append((score, fig))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored else None
