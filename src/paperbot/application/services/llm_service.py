from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, Generator, List, Optional, Sequence

from paperbot.application.prompts import PromptRegistry
from paperbot.infrastructure.llm.router import ModelRouter

logger = logging.getLogger(__name__)


class LLMService:
    """Project-level LLM facade with task routing, prompt templates, and light caching."""

    def __init__(
        self,
        router: Optional[ModelRouter] = None,
        prompt_registry: Optional[PromptRegistry] = None,
        *,
        enable_cache: bool = True,
        raise_errors: bool = False,
    ) -> None:
        self._router = router or ModelRouter.from_env()
        self._prompts = prompt_registry or PromptRegistry()
        self._enable_cache = enable_cache
        self._raise_errors = raise_errors
        self._cache: Dict[str, str] = {}

    def complete(
        self,
        *,
        task_type: str = "default",
        system: str,
        user: str,
        use_cache: bool = True,
        **kwargs,
    ) -> str:
        cache_key = self._cache_key(task_type=task_type, system=system, user=user, kwargs=kwargs)
        if self._enable_cache and use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            provider = self._router.get_provider(task_type)
            result = (provider.invoke_simple(system, user, **kwargs) or "").strip()
        except Exception as exc:  # pragma: no cover - exercised via fallback tests
            logger.warning("LLM complete failed task_type=%s error=%s", task_type, exc)
            if self._raise_errors:
                raise
            result = ""

        if self._enable_cache and use_cache:
            self._cache[cache_key] = result
        return result

    def stream(
        self,
        *,
        task_type: str = "default",
        system: str,
        user: str,
        **kwargs,
    ) -> Generator[str, None, None]:
        try:
            provider = self._router.get_provider(task_type)
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            yield from provider.stream_invoke(messages, **kwargs)
        except Exception as exc:  # pragma: no cover - stream failures are runtime/network specific
            logger.warning("LLM stream failed task_type=%s error=%s", task_type, exc)
            if self._raise_errors:
                raise
            return

    def summarize_paper(self, title: str, abstract: str) -> str:
        prompt = self._prompts.get("paper_summary")
        return self.complete(
            task_type="summary",
            system=prompt.system,
            user=prompt.user.format(title=title or "", abstract=abstract or ""),
        )

    def analyze_trends(self, *, topic: str, papers: Sequence[Dict[str, Any]]) -> str:
        prompt = self._prompts.get("trend_analysis")
        serialized = _format_papers_for_prompt(papers)
        return self.complete(
            task_type="reasoning",
            system=prompt.system,
            user=prompt.user.format(topic=topic or "", papers=serialized),
        )

    def assess_relevance(self, *, paper: Dict[str, Any], query: str) -> Dict[str, Any]:
        prompt = self._prompts.get("relevance_assess")
        raw = self.complete(
            task_type="extraction",
            system=prompt.system,
            user=prompt.user.format(
                query=query or "",
                title=paper.get("title") or "",
                abstract=paper.get("snippet") or paper.get("abstract") or "",
                keywords=", ".join(paper.get("keywords") or []),
            ),
        )
        parsed = _safe_parse_json(raw)
        if parsed is None:
            return {
                "score": _overlap_relevance_score(query=query, paper=paper),
                "reason": "Fallback score from token overlap (LLM output unavailable).",
            }

        score = int(parsed.get("score") or 0)
        score = max(0, min(score, 100))
        reason = str(parsed.get("reason") or "")
        return {"score": score, "reason": reason}

    def generate_daily_insight(self, report: Dict[str, Any]) -> str:
        prompt = self._prompts.get("daily_insight")
        stats = report.get("stats") or {}
        highlights = []
        for query in report.get("queries") or []:
            q_name = query.get("normalized_query") or query.get("raw_query") or ""
            top_items = query.get("top_items") or []
            top_title = top_items[0].get("title") if top_items else ""
            hit_count = query.get("total_hits", 0)
            highlights.append(f"- {q_name}: hits={hit_count}, top={top_title}")

        return self.complete(
            task_type="reasoning",
            system=prompt.system,
            user=prompt.user.format(
                title=report.get("title") or "DailyPaper Digest",
                date=report.get("date") or "",
                stats=json.dumps(stats, ensure_ascii=False),
                highlights="\n".join(highlights),
            ),
        )

    def _cache_key(self, *, task_type: str, system: str, user: str, kwargs: Dict[str, Any]) -> str:
        payload = json.dumps(
            {
                "task_type": task_type,
                "system": system,
                "user": user,
                "kwargs": kwargs,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def describe_task_provider(self, task_type: str = "default") -> Dict[str, Any]:
        """Expose selected provider metadata for auditing/judge reports."""
        try:
            provider = self._router.get_provider(task_type)
            info = provider.info
            return {
                "provider_name": info.provider_name,
                "model_name": info.model_name,
                "cost_tier": info.cost_tier,
            }
        except Exception:
            return {
                "provider_name": "",
                "model_name": "",
                "cost_tier": 0,
            }


_default_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    global _default_llm_service
    if _default_llm_service is None:
        _default_llm_service = LLMService()
    return _default_llm_service


def _format_papers_for_prompt(papers: Sequence[Dict[str, Any]], limit: int = 12) -> str:
    rows: List[str] = []
    for idx, paper in enumerate(list(papers)[: max(1, int(limit))], start=1):
        rows.append(
            "{idx}. {title} | keywords={keywords} | snippet={snippet}".format(
                idx=idx,
                title=paper.get("title") or "Untitled",
                keywords=", ".join(paper.get("keywords") or []),
                snippet=(paper.get("snippet") or paper.get("abstract") or "")[:220],
            )
        )
    return "\n".join(rows)


def _safe_parse_json(raw: str) -> Optional[Dict[str, Any]]:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def _overlap_relevance_score(*, query: str, paper: Dict[str, Any]) -> int:
    query_tokens = _tokenize(query)
    haystack = " ".join(
        [
            paper.get("title") or "",
            paper.get("snippet") or paper.get("abstract") or "",
            " ".join(paper.get("keywords") or []),
        ]
    ).lower()
    if not query_tokens:
        return 0

    hit = 0
    for token in query_tokens:
        if token in haystack:
            hit += 1
    return int((hit / len(query_tokens)) * 100)


def _tokenize(text: str) -> List[str]:
    lowered = (text or "").strip().lower()
    tokens = [token for token in lowered.replace("_", " ").split() if token]
    dedup: List[str] = []
    seen = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        dedup.append(token)
    return dedup
