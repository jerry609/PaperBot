from __future__ import annotations

import asyncio
import hashlib
import math
import random
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
    fields = {
        str(x).strip().lower() for x in (paper.get("fields_of_study") or []) if str(x).strip()
    }
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
    fields = [
        str(x).strip().lower() for x in (paper.get("fields_of_study") or []) if str(x).strip()
    ]
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


def _paper_score(
    paper: Dict[str, Any],
    *,
    track: Optional[Dict[str, Any]],
    boosts: Dict[str, float],
    kw_weight: float = 0.55,
    citation_weight: float = 0.30,
    recency_weight: float = 0.15,
) -> float:
    pid = str(paper.get("paper_id") or "").strip()
    citations = float(paper.get("citation_count") or 0.0)
    year = paper.get("year")
    year_val = float(year) if isinstance(year, int) else 0.0

    kw = float(_paper_keyword_match_count(paper, track))
    cit = math.log1p(max(0.0, citations)) / 5.0
    rec = 0.0
    if year_val >= 2015:
        rec = min(1.0, (year_val - 2018.0) / 8.0)

    total = max(0.0001, float(kw_weight + citation_weight + recency_weight))
    kw_weight = float(kw_weight) / total
    citation_weight = float(citation_weight) / total
    recency_weight = float(recency_weight) / total

    return (
        kw_weight * min(1.0, kw / 6.0)
        + citation_weight * min(1.0, cit)
        + recency_weight * max(0.0, rec)
        + boosts.get(pid, 0.0)
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _paper_title_tokens(paper: Dict[str, Any]) -> set[str]:
    return _tokenize(str(paper.get("title") or ""))


def _paper_fields(paper: Dict[str, Any]) -> set[str]:
    return {str(x).strip().lower() for x in (paper.get("fields_of_study") or []) if str(x).strip()}


def _paper_first_author(paper: Dict[str, Any]) -> str:
    authors = paper.get("authors") or []
    if isinstance(authors, list) and authors:
        return str(authors[0]).strip()
    return ""


def _paper_venue(paper: Dict[str, Any]) -> str:
    return str(paper.get("venue") or "").strip()


def _paper_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """
    Small, fast similarity heuristic used for diversification.

    Returns 0..1.
    """
    if not a or not b:
        return 0.0

    sim = 0.0
    a_first = _paper_first_author(a)
    b_first = _paper_first_author(b)
    if a_first and b_first and a_first.lower() == b_first.lower():
        sim += 0.7

    a_venue = _paper_venue(a)
    b_venue = _paper_venue(b)
    if a_venue and b_venue and a_venue.lower() == b_venue.lower():
        sim += 0.2

    a_fields = _paper_fields(a)
    b_fields = _paper_fields(b)
    if a_fields and b_fields and (a_fields & b_fields):
        sim += 0.1

    # Minor title token overlap (kept tiny to avoid over-penalizing synonyms).
    ta = _paper_title_tokens(a)
    tb = _paper_title_tokens(b)
    if ta and tb:
        j = len(ta & tb) / max(1, len(ta | tb))
        sim += 0.1 * min(1.0, j * 4.0)

    return min(1.0, sim)


@dataclass(frozen=True)
class RecommendationPolicy:
    stage: str = "auto"  # auto/survey/writing/rebuttal
    exploration_ratio: float = 0.15
    diversity_strength: float = 0.55
    max_per_first_author: int = 2
    max_per_venue: int = 3
    max_per_field: int = 4


def _normalize_stage(stage: Optional[str]) -> str:
    s = (stage or "").strip().lower()
    if s in {"survey", "writing", "rebuttal"}:
        return s
    if s in {"auto", ""}:
        return "auto"
    return "auto"


def _derive_stage(
    *, tasks: List[Dict[str, Any]], milestones: List[Dict[str, Any]], saved_count: int
) -> str:
    text = " ".join(
        [str(t.get("title") or "") for t in tasks] + [str(m.get("name") or "") for m in milestones]
    ).lower()
    if any(k in text for k in ["rebuttal", "camera-ready", "response to reviewers"]):
        return "rebuttal"

    done = 0
    for t in tasks:
        if str(t.get("status") or "").strip().lower() in {"done", "completed"}:
            done += 1
    if saved_count >= 6 or done >= 4:
        return "writing"
    return "survey"


def _stage_defaults(stage: str) -> RecommendationPolicy:
    if stage == "survey":
        return RecommendationPolicy(stage=stage, exploration_ratio=0.25, diversity_strength=0.70)
    if stage == "writing":
        return RecommendationPolicy(stage=stage, exploration_ratio=0.10, diversity_strength=0.45)
    if stage == "rebuttal":
        return RecommendationPolicy(stage=stage, exploration_ratio=0.05, diversity_strength=0.25)
    return RecommendationPolicy(stage="auto")


def _select_diverse(
    *,
    papers: List[Dict[str, Any]],
    scores: Dict[str, float],
    limit: int,
    policy: RecommendationPolicy,
    seed: str,
) -> List[Dict[str, Any]]:
    if limit <= 0 or not papers:
        return []

    ratio = float(policy.exploration_ratio or 0.0)
    ratio = max(0.0, min(0.5, ratio))
    explore_n = int(round(limit * ratio))
    exploit_n = max(0, int(limit - explore_n))

    # Deterministic RNG per (user/query/track) to keep UI stable across refreshes.
    rng = random.Random(int(_sha256_text(seed)[:8], 16))

    # Favor higher-scoring papers for exploitation.
    ranked = sorted(
        papers, key=lambda x: scores.get(str(x.get("paper_id") or ""), 0.0), reverse=True
    )
    exploit_pool = ranked[: max(30, limit * 6)]
    explore_pool = ranked[max(10, limit * 2) :]

    def can_take(p: Dict[str, Any], counts: Dict[str, Dict[str, int]]) -> bool:
        first = _paper_first_author(p).lower()
        venue = _paper_venue(p).lower()
        fields = _paper_fields(p)
        if first and counts["first"].get(first, 0) >= policy.max_per_first_author:
            return False
        if venue and counts["venue"].get(venue, 0) >= policy.max_per_venue:
            return False
        for f in fields:
            if counts["field"].get(f, 0) >= policy.max_per_field:
                return False
        return True

    def bump_counts(p: Dict[str, Any], counts: Dict[str, Dict[str, int]]) -> None:
        first = _paper_first_author(p).lower()
        venue = _paper_venue(p).lower()
        fields = _paper_fields(p)
        if first:
            counts["first"][first] = counts["first"].get(first, 0) + 1
        if venue:
            counts["venue"][venue] = counts["venue"].get(venue, 0) + 1
        for f in fields:
            counts["field"][f] = counts["field"].get(f, 0) + 1

    selected: List[Dict[str, Any]] = []
    used_ids: set[str] = set()
    counts: Dict[str, Dict[str, int]] = {"first": {}, "venue": {}, "field": {}}

    def greedy_pick(from_pool: List[Dict[str, Any]], n: int) -> None:
        nonlocal selected, used_ids
        if n <= 0:
            return

        pool = [p for p in from_pool if str(p.get("paper_id") or "").strip()]
        pool = [p for p in pool if str(p.get("paper_id") or "").strip() not in used_ids]
        for _ in range(n):
            best = None
            best_score = -1e9
            for p in pool:
                pid = str(p.get("paper_id") or "").strip()
                if not pid or pid in used_ids:
                    continue
                if not can_take(p, counts):
                    continue
                base = float(scores.get(pid, 0.0))
                div_penalty = 0.0
                if selected:
                    div_penalty = max(_paper_similarity(p, s) for s in selected)
                candidate_score = base - float(policy.diversity_strength) * div_penalty
                if candidate_score > best_score:
                    best = p
                    best_score = candidate_score
            if best is None:
                break
            pid = str(best.get("paper_id") or "").strip()
            selected.append(best)
            used_ids.add(pid)
            bump_counts(best, counts)

    greedy_pick(exploit_pool, exploit_n)

    if explore_n > 0 and explore_pool:
        explore_candidates = list(explore_pool)
        rng.shuffle(explore_candidates)
        explore_candidates.sort(
            key=lambda x: (
                -(int(x.get("year") or 0)),
                float(x.get("citation_count") or 0.0),
                rng.random(),
            )
        )
        greedy_pick(explore_candidates, explore_n)

    # Fill any remaining slots from the ranked pool.
    if len(selected) < limit:
        greedy_pick(ranked, int(limit - len(selected)))

    return selected[:limit]


@dataclass(frozen=True)
class ContextEngineConfig:
    memory_limit: int = 8
    task_limit: int = 8
    milestone_limit: int = 6
    paper_limit: int = 8
    offline: bool = False
    stage: str = "auto"
    exploration_ratio: Optional[float] = None
    diversity_strength: Optional[float] = None
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
            self.research_store.get_track(user_id=user_id, track_id=track_id)
            if track_id is not None
            else active_track
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
            tracks = self.research_store.list_tracks(
                user_id=user_id, include_archived=False, limit=50
            )
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
        stage_raw = _normalize_stage(self.config.stage)

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

        stage = stage_raw
        if stage_raw == "auto":
            stage = _derive_stage(
                tasks=progress_tasks, milestones=progress_milestones, saved_count=len(saved_ids)
            )

        stage_policy = _stage_defaults(stage)
        exploration_ratio = (
            float(self.config.exploration_ratio)
            if self.config.exploration_ratio is not None
            else float(stage_policy.exploration_ratio)
        )
        diversity_strength = (
            float(self.config.diversity_strength)
            if self.config.diversity_strength is not None
            else float(stage_policy.diversity_strength)
        )

        score_weights = {
            "survey": (0.45, 0.15, 0.40),
            "writing": (0.60, 0.30, 0.10),
            "rebuttal": (0.50, 0.40, 0.10),
        }.get(stage, (0.55, 0.30, 0.15))

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
                    paper_scores[pid] = float(
                        _paper_score(
                            p,
                            track=routed_track,
                            boosts=boosts,
                            kw_weight=score_weights[0],
                            citation_weight=score_weights[1],
                            recency_weight=score_weights[2],
                        )
                    )
                    paper_reasons[pid] = _paper_reasons(p, routed_track)

                policy = RecommendationPolicy(
                    stage=stage,
                    exploration_ratio=exploration_ratio,
                    diversity_strength=diversity_strength,
                )
                papers = _select_diverse(
                    papers=filtered,
                    scores=paper_scores,
                    limit=int(self.config.paper_limit),
                    policy=policy,
                    seed=f"{user_id}:{merged_query}:{stage}:{routed_track.get('id') if routed_track else ''}",
                )
            except Exception:
                papers = []

        routing = {
            "track_id": routed_track["id"] if routed_track else None,
            "used_active_track": track_id is None,
            "include_cross_track": bool(include_cross_track),
            "query": query,
            "merged_query": merged_query,
            "stage": stage,
            "exploration_ratio": float(exploration_ratio),
            "diversity_strength": float(diversity_strength),
            "suggestion": routing_suggestion,
        }

        context_run_id: Optional[int] = None
        try:
            created = self.research_store.create_context_run(
                user_id=user_id,
                track_id=int(routed_track["id"]) if routed_track else None,
                query=query,
                merged_query=merged_query,
                stage=stage,
                exploration_ratio=float(exploration_ratio),
                diversity_strength=float(diversity_strength),
                routing=routing,
                papers=papers,
                paper_scores=paper_scores,
                paper_reasons=paper_reasons,
            )
            if created and created.get("id"):
                context_run_id = int(created["id"])
        except Exception:
            context_run_id = None

        return {
            "user_id": user_id,
            "context_run_id": context_run_id,
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
