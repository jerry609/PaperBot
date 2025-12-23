from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from paperbot.context_engine.track_router import TrackRouter, TrackRouterConfig
from paperbot.domain.paper import PaperMeta
from paperbot.infrastructure.stores.memory_store import SqlAlchemyMemoryStore
from paperbot.infrastructure.stores.research_store import SqlAlchemyResearchStore

_TOKEN_RX = re.compile(r"[a-zA-Z0-9_+.-]+")


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RX.findall(text or "") if t.strip()}


def _merge_query(query: str, keywords: List[str]) -> str:
    base = (query or "").strip()
    kws = [k.strip() for k in (keywords or []) if k and k.strip()]
    if not kws:
        return base
    kws = sorted(set(kws), key=str.lower)[:12]
    return f"{base} " + " ".join(kws)


def _normalize_title(title: str) -> str:
    s = (title or "").strip().lower()
    s = re.sub(r"\\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s


def _paper_keyword_match_count(paper: Dict[str, Any], track: Optional[Dict[str, Any]]) -> int:
    if not track:
        return 0
    kws = {str(k).strip().lower() for k in (track.get("keywords") or []) if str(k).strip()}
    if not kws:
        return 0
    title_tokens = _tokenize(str(paper.get("title") or ""))
    fields = {str(x).strip().lower() for x in (paper.get("fields_of_study") or []) if str(x).strip()}
    count = 0
    for k in kws:
        if _tokenize(k) & title_tokens:
            count += 1
        elif k in fields:
            count += 1
    return count


def _paper_reasons(paper: Dict[str, Any], track: Optional[Dict[str, Any]]) -> List[str]:
    if not track:
        return []
    reasons: List[str] = []
    kws = {str(k).strip().lower() for k in (track.get("keywords") or []) if str(k).strip()}
    if not kws:
        return []

    title = str(paper.get("title") or "")
    fields = [str(x).strip().lower() for x in (paper.get("fields_of_study") or []) if str(x).strip()]
    venue = str(paper.get("venue") or "")

    matched = []
    title_tokens = _tokenize(title)
    for k in sorted(kws):
        if _tokenize(k) & title_tokens:
            matched.append(k)
    if matched:
        reasons.append(f"keyword match in title: {', '.join(matched[:5])}")

    field_matched = [k for k in sorted(kws) if k in set(fields)]
    if field_matched:
        reasons.append(f"field match: {', '.join(field_matched[:5])}")

    venues = {str(v).strip().lower() for v in (track.get("venues") or []) if str(v).strip()}
    if venues and venue and venue.strip().lower() in venues:
        reasons.append(f"venue match: {venue}")

    return reasons[:3]


def _paper_score(paper: Dict[str, Any], *, track: Optional[Dict[str, Any]], boosts: Dict[str, float]) -> float:
    pid = str(paper.get("paper_id") or "").strip()
    citations = float(paper.get("citation_count") or 0.0)
    year = paper.get("year")
    year_val = float(year) if isinstance(year, int) else 0.0

    kw = float(_paper_keyword_match_count(paper, track))
    cit = math.log1p(max(0.0, citations)) / 5.0
    rec = 0.0
    if year_val >= 2015:
        rec = min(1.0, (year_val - 2018.0) / 8.0)
    return 0.55 * min(1.0, kw / 6.0) + 0.30 * min(1.0, cit) + 0.15 * max(0.0, rec) + boosts.get(pid, 0.0)


def _select_diverse(papers: List[Dict[str, Any]], scores: Dict[str, float], limit: int) -> List[Dict[str, Any]]:
    # Limit same first-author dominance.
    selected: List[Dict[str, Any]] = []
    by_author_count: Dict[str, int] = {}
    for p in sorted(papers, key=lambda x: scores.get(str(x.get("paper_id") or ""), 0.0), reverse=True):
        if len(selected) >= limit:
            break
        authors = p.get("authors") or []
        first = ""
        if isinstance(authors, list) and authors:
            first = str(authors[0]).strip()
        if first:
            if by_author_count.get(first, 0) >= 2:
                continue
            by_author_count[first] = by_author_count.get(first, 0) + 1
        selected.append(p)

    # Exploration: if we have capacity, add one recent low-citation item not selected yet.
    if len(selected) < limit:
        remaining = [p for p in papers if p not in selected]
        remaining.sort(
            key=lambda x: (
                -(int(x.get("year") or 0)),
                float(x.get("citation_count") or 0.0),
            )
        )
        for p in remaining:
            selected.append(p)
            break

    return selected[:limit]


@dataclass(frozen=True)
class ContextEngineConfig:
    memory_limit: int = 8
    task_limit: int = 8
    milestone_limit: int = 6
    paper_limit: int = 8
    offline: bool = False
    track_router: TrackRouterConfig = field(default_factory=TrackRouterConfig)


class ContextEngine:
    def __init__(
        self,
        *,
        research_store: Optional[SqlAlchemyResearchStore] = None,
        memory_store: Optional[SqlAlchemyMemoryStore] = None,
        paper_searcher: Optional[Any] = None,
        track_router: Optional[TrackRouter] = None,
        config: Optional[ContextEngineConfig] = None,
    ):
        self.research_store = research_store or SqlAlchemyResearchStore()
        self.memory_store = memory_store or SqlAlchemyMemoryStore()
        self.paper_searcher = paper_searcher
        self.config = config or ContextEngineConfig()
        self.track_router = track_router or TrackRouter(
            research_store=self.research_store,
            memory_store=self.memory_store,
            config=self.config.track_router,
        )

    async def build_context_pack(
        self,
        *,
        user_id: str,
        query: str,
        track_id: Optional[int] = None,
        include_cross_track: bool = False,
    ) -> Dict[str, Any]:
        active_track = self.research_store.get_active_track(user_id=user_id)
        routed_track = (
            self.research_store.get_track(user_id=user_id, track_id=track_id) if track_id is not None else active_track
        )

        routing_suggestion: Optional[Dict[str, Any]] = None
        if track_id is None and active_track is not None:
            routing_suggestion = self.track_router.suggest_track(
                user_id=user_id,
                query=query,
                active_track_id=int(active_track["id"]),
                limit=50,
            )

        track_scope_id = str(routed_track["id"]) if routed_track else None

        user_prefs = self.memory_store.list_memories(
            user_id=user_id,
            limit=self.config.memory_limit,
            scope_type="global",
            include_pending=False,
        )
        self.memory_store.touch_usage(
            item_ids=[int(i["id"]) for i in user_prefs if i.get("id")],
            actor_id="context_engine",
        )

        progress_tasks: List[Dict[str, Any]] = []
        progress_milestones: List[Dict[str, Any]] = []
        if routed_track:
            progress_tasks = self.research_store.list_tasks(
                user_id=user_id, track_id=int(routed_track["id"]), limit=self.config.task_limit
            )
            progress_milestones = self.research_store.list_milestones(
                user_id=user_id, track_id=int(routed_track["id"]), limit=self.config.milestone_limit
            )

        relevant_memories: List[Dict[str, Any]] = []
        cross_track_memories: List[Dict[str, Any]] = []
        if track_scope_id:
            relevant_memories = self.memory_store.search_memories(
                user_id=user_id,
                query=query,
                limit=self.config.memory_limit,
                scope_type="track",
                scope_id=track_scope_id,
            )
            self.memory_store.touch_usage(
                item_ids=[int(i["id"]) for i in relevant_memories if i.get("id")],
                actor_id="context_engine",
            )

        if include_cross_track and track_id is None:
            tracks = self.research_store.list_tracks(user_id=user_id, include_archived=False, limit=50)
            for t in tracks:
                if routed_track and t.get("id") == routed_track.get("id"):
                    continue
                sid = str(t.get("id") or "")
                if not sid:
                    continue
                hits = self.memory_store.search_memories(
                    user_id=user_id,
                    query=query,
                    limit=max(2, self.config.memory_limit // 2),
                    scope_type="track",
                    scope_id=sid,
                )
                for h in hits:
                    h["track_id"] = t.get("id")
                    h["track_name"] = t.get("name")
                cross_track_memories.extend(hits)

        merged_query = query
        if routed_track:
            merged_query = _merge_query(query, routed_track.get("keywords") or [])

        papers: List[Dict[str, Any]] = []
        paper_scores: Dict[str, float] = {}
        paper_reasons: Dict[str, List[str]] = {}

        disliked_ids: set[str] = set()
        saved_ids: set[str] = set()
        liked_ids: set[str] = set()
        if routed_track:
            disliked_ids = self.research_store.list_paper_feedback_ids(
                user_id=user_id, track_id=int(routed_track["id"]), action="dislike"
            )
            saved_ids = self.research_store.list_paper_feedback_ids(
                user_id=user_id, track_id=int(routed_track["id"]), action="save"
            )
            liked_ids = self.research_store.list_paper_feedback_ids(
                user_id=user_id, track_id=int(routed_track["id"]), action="like"
            )

        boosts: Dict[str, float] = {}
        for pid in saved_ids:
            boosts[pid] = boosts.get(pid, 0.0) + 0.25
        for pid in liked_ids:
            boosts[pid] = boosts.get(pid, 0.0) + 0.15

        if not self.config.offline and self.config.paper_limit > 0:
            try:
                searcher = self.paper_searcher
                if searcher is None:
                    from paperbot.utils.search import SemanticScholarSearch  # local import

                    searcher = SemanticScholarSearch()

                fetch_limit = max(30, int(self.config.paper_limit) * 3)
                resp = await asyncio.to_thread(searcher.search_papers, merged_query, fetch_limit)

                raw: List[Dict[str, Any]] = []
                for p in getattr(resp, "papers", []) or []:
                    authors = []
                    for a in getattr(p, "authors", []) or []:
                        if isinstance(a, dict):
                            name = str(a.get("name") or "").strip()
                            if name:
                                authors.append(name)
                    raw.append(
                        PaperMeta(
                            paper_id=str(getattr(p, "paper_id", "") or ""),
                            title=str(getattr(p, "title", "") or ""),
                            abstract=getattr(p, "abstract", None),
                            year=getattr(p, "year", None),
                            venue=getattr(p, "venue", None),
                            citation_count=int(getattr(p, "citation_count", 0) or 0),
                            authors=authors,
                            url=getattr(p, "url", None),
                            fields_of_study=list(getattr(p, "fields_of_study", []) or []),
                            publication_date=getattr(p, "publication_date", None),
                        ).to_dict()
                    )

                # Feedback filtering + dedup
                seen_titles: set[str] = set()
                filtered: List[Dict[str, Any]] = []
                for p in raw:
                    pid = str(p.get("paper_id") or "").strip()
                    if pid and pid in disliked_ids:
                        continue
                    tkey = _normalize_title(str(p.get("title") or ""))
                    if tkey and tkey in seen_titles:
                        continue
                    if tkey:
                        seen_titles.add(tkey)
                    filtered.append(p)

                for p in filtered:
                    pid = str(p.get("paper_id") or "").strip()
                    if not pid:
                        continue
                    paper_scores[pid] = float(_paper_score(p, track=routed_track, boosts=boosts))
                    paper_reasons[pid] = _paper_reasons(p, routed_track)

                papers = _select_diverse(filtered, paper_scores, int(self.config.paper_limit))
            except Exception:
                papers = []

        routing = {
            "track_id": routed_track["id"] if routed_track else None,
            "used_active_track": track_id is None,
            "include_cross_track": bool(include_cross_track),
            "query": query,
            "merged_query": merged_query,
            "suggestion": routing_suggestion,
        }

        return {
            "user_id": user_id,
            "routing": routing,
            "user_prefs": user_prefs,
            "active_track": routed_track,
            "progress_state": {"tasks": progress_tasks, "milestones": progress_milestones},
            "relevant_memories": relevant_memories,
            "cross_track_memories": cross_track_memories,
            "paper_recommendations": papers,
            "paper_recommendation_scores": paper_scores,
            "paper_recommendation_reasons": paper_reasons,
        }

    async def close(self) -> None:
        return None

