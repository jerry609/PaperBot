from __future__ import annotations

import asyncio
import hashlib
import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from paperbot.context_engine.track_router import TrackRouter, TrackRouterConfig
from paperbot.infrastructure.stores.memory_store import SqlAlchemyMemoryStore
from paperbot.infrastructure.stores.research_store import SqlAlchemyResearchStore
from paperbot.utils.logging_config import Logger, LogFiles

# Optional: PaperSearchService for unified search
try:
    from paperbot.application.services.paper_search_service import PaperSearchService
except ImportError:  # pragma: no cover
    PaperSearchService = None  # type: ignore

# Optional: AnchorService for personalized author boosts
try:
    from paperbot.application.services.anchor_service import AnchorService
except ImportError:  # pragma: no cover
    AnchorService = None  # type: ignore

_TOKEN_RX = re.compile(r"[a-zA-Z0-9_+.-]+")


_anchor_service: Optional["AnchorService"] = None


def _get_anchor_service() -> Optional["AnchorService"]:
    global _anchor_service
    if AnchorService is None:
        return None
    if _anchor_service is None:
        _anchor_service = AnchorService()
    return _anchor_service


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


# ── Short-query expansion: common CS/ML acronyms ──
_ACRONYM_MAP: Dict[str, str] = {
    "rag": "retrieval augmented generation RAG",
    "llm": "large language model LLM",
    "llms": "large language models LLM",
    "rlhf": "reinforcement learning from human feedback RLHF",
    "rl": "reinforcement learning RL",
    "nlp": "natural language processing NLP",
    "cv": "computer vision CV",
    "gan": "generative adversarial network GAN",
    "gnn": "graph neural network GNN",
    "vae": "variational autoencoder VAE",
    "moe": "mixture of experts MoE",
    "cot": "chain of thought CoT",
    "dpo": "direct preference optimization DPO",
    "grpo": "group relative policy optimization GRPO",
    "mllm": "multimodal large language model MLLM",
    "vla": "vision language action VLA",
    "vlm": "vision language model VLM",
    "sft": "supervised fine-tuning SFT",
    "ppo": "proximal policy optimization PPO",
    "lora": "low-rank adaptation LoRA",
    "dit": "diffusion transformer DiT",
}


def _expand_short_query(query: str) -> str:
    """Expand short acronym queries to include the full term for better search."""
    stripped = query.strip()
    key = stripped.lower()
    if key in _ACRONYM_MAP:
        return _ACRONYM_MAP[key]
    return stripped


def _is_academic_paper(paper: Dict[str, Any]) -> bool:
    """Filter out non-academic results (shopping pages, poetry, etc.)."""
    title = str(paper.get("title") or "")
    abstract = str(paper.get("abstract") or "")

    # Must have a non-trivial abstract (>30 chars)
    if len(abstract.strip()) < 30:
        return False

    # Reject titles that are mostly non-Latin characters (CJK, Arabic, etc.)
    latin_chars = sum(1 for c in title if c.isascii() and c.isalpha())
    total_alpha = sum(1 for c in title if c.isalpha())
    if total_alpha > 0 and latin_chars / total_alpha < 0.5:
        return False

    return True


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


def _learn_source_weights_from_feedback(
    *,
    feedback_rows: List[Dict[str, Any]],
    selected_sources: List[str],
    default_weights: Dict[str, float],
    min_samples: int = 8,
) -> Optional[Dict[str, float]]:
    """Estimate per-source RRF weights from recent explicit feedback metadata."""
    valid_sources = [str(s).strip() for s in (selected_sources or []) if str(s).strip()]
    if not valid_sources:
        return None

    stats: Dict[str, Dict[str, float]] = {
        source: {"pos": 0.0, "neg": 0.0, "n": 0.0} for source in valid_sources
    }

    action_signal = {
        "save": 2.0,
        "like": 1.5,
        "cite": 1.5,
        "dislike": -2.0,
        "not_relevant": -2.0,
        "not-relevant": -2.0,
        "skip": -0.5,
    }

    sample_count = 0
    for row in feedback_rows or []:
        action = str(row.get("action") or "").strip().lower()
        signal = float(action_signal.get(action, 0.0))
        if signal == 0.0:
            continue

        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        retrieval_sources = metadata.get("retrieval_sources")
        if not isinstance(retrieval_sources, list):
            retrieval_sources = []

        clean_sources = [
            str(source).strip() for source in retrieval_sources if str(source).strip() in stats
        ]
        if not clean_sources:
            continue

        sample_count += 1
        for source in clean_sources:
            stats[source]["n"] += 1.0
            if signal > 0:
                stats[source]["pos"] += signal
            else:
                stats[source]["neg"] += abs(signal)

    if sample_count < max(1, int(min_samples)):
        return None

    learned: Dict[str, float] = {}
    for source in valid_sources:
        base = float(default_weights.get(source, 0.5))
        s = stats[source]
        # Beta-style smoothing to avoid extreme swings under sparse feedback.
        pos = float(s["pos"])
        neg = float(s["neg"])
        ctr_like = (pos + 1.0) / (pos + neg + 2.0)
        scale = 0.6 + 0.8 * ctr_like  # [0.6, 1.4]
        learned[source] = min(1.8, max(0.3, base * scale))

    return learned


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
    search_sources: Optional[List[str]] = None
    exploration_ratio: Optional[float] = None
    diversity_strength: Optional[float] = None
    personalized: bool = True
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    track_router: TrackRouterConfig = field(default_factory=TrackRouterConfig)


class ContextEngine:
    def __init__(
        self,
        *,
        research_store: Optional[SqlAlchemyResearchStore] = None,
        memory_store: Optional[SqlAlchemyMemoryStore] = None,
        paper_store: Optional[Any] = None,
        search_service: Optional[Any] = None,
        track_router: Optional[TrackRouter] = None,
        config: Optional[ContextEngineConfig] = None,
    ):
        self.research_store = research_store or SqlAlchemyResearchStore()
        self.memory_store = memory_store or SqlAlchemyMemoryStore()
        self.paper_store = paper_store
        self.search_service = search_service
        self.config = config or ContextEngineConfig()
        self.track_router = track_router or TrackRouter(
            research_store=self.research_store,
            memory_store=self.memory_store,
            config=self.config.track_router,
        )

    def _attach_latest_judge(self, papers: List[Dict[str, Any]]) -> None:
        ids: List[int] = []
        for paper in papers:
            pid = str(paper.get("paper_id") or "").strip()
            if pid.isdigit():
                ids.append(int(pid))
        if not ids:
            return

        if self.paper_store is None:
            from paperbot.infrastructure.stores.paper_store import PaperStore

            self.paper_store = PaperStore(auto_create_schema=False)

        judge_map = self.paper_store.get_latest_judge_scores(ids)
        for paper in papers:
            pid = str(paper.get("paper_id") or "").strip()
            if pid.isdigit() and int(pid) in judge_map:
                paper["latest_judge"] = judge_map[int(pid)]

    @staticmethod
    def _attach_feedback_flags(
        papers: List[Dict[str, Any]], *, saved_ids: set[str], liked_ids: set[str]
    ) -> None:
        for paper in papers:
            pid = str(paper.get("paper_id") or "").strip()
            if not pid:
                continue
            if pid in saved_ids:
                paper["is_saved"] = True
            if pid in liked_ids:
                paper["is_liked"] = True

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

        merged_query = _expand_short_query(query)
        if routed_track:
            merged_query = _merge_query(merged_query, routed_track.get("keywords") or [])

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
        if self.config.personalized:
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

        learned_source_weights: Optional[Dict[str, float]] = None

        Logger.info(
            f"Paper search config: offline={self.config.offline}, "
            f"paper_limit={self.config.paper_limit}",
            file=LogFiles.HARVEST,
        )
        if not self.config.offline and self.config.paper_limit > 0:
            try:
                fetch_limit = max(30, int(self.config.paper_limit) * 3)

                # Prefer PaperSearchService if available
                if self.search_service is not None:
                    selected_sources = [
                        str(x).strip() for x in (self.config.search_sources or []) if str(x).strip()
                    ]
                    if not selected_sources:
                        selected_sources = ["semantic_scholar"]

                    if self.config.personalized and routed_track:
                        try:
                            feedback_rows = self.research_store.list_paper_feedback(
                                user_id=user_id,
                                track_id=int(routed_track["id"]),
                                limit=500,
                            )
                            default_rrf_weights = getattr(
                                self.search_service,
                                "DEFAULT_SOURCE_WEIGHTS",
                                {},
                            )
                            learned_source_weights = _learn_source_weights_from_feedback(
                                feedback_rows=feedback_rows,
                                selected_sources=selected_sources,
                                default_weights=dict(default_rrf_weights or {}),
                            )
                        except Exception as exc:
                            Logger.warning(
                                f"Failed to learn source weights: {exc}",
                                file=LogFiles.HARVEST,
                            )

                    Logger.info(
                        f"Using PaperSearchService for query='{merged_query}'",
                        file=LogFiles.HARVEST,
                    )
                    search_result = await self.search_service.search(
                        merged_query,
                        sources=selected_sources,
                        max_results=fetch_limit,
                        year_from=self.config.year_from,
                        year_to=self.config.year_to,
                        persist=True,
                        source_weights=learned_source_weights,
                    )
                    raw = [p.to_dict() for p in search_result.papers]
                    # Inject paper_id from canonical_id or first identity
                    for p_dict, p_obj in zip(raw, search_result.papers):
                        pid = str(p_obj.canonical_id or "")
                        if not pid:
                            pid = p_obj.get_identity("semantic_scholar") or ""
                        p_dict["paper_id"] = pid
                    Logger.info(
                        f"PaperSearchService returned {len(raw)} papers",
                        file=LogFiles.HARVEST,
                    )
                else:
                    Logger.warning(
                        "No search_service provided — skipping paper search. "
                        "Pass a PaperSearchService instance to ContextEngine.",
                        file=LogFiles.HARVEST,
                    )
                    raw = []

                # Local DB fallback when external search returns no results
                if not raw and self.paper_store is not None:
                    Logger.info(
                        f"External search returned 0 results, falling back to local DB for query='{merged_query}'",
                        file=LogFiles.HARVEST,
                    )
                    try:
                        from paperbot.infrastructure.stores.paper_store import paper_to_dict

                        local_papers, _ = self.paper_store.search_papers(
                            query=merged_query, limit=fetch_limit, sort_by="citation_count"
                        )
                        raw = []
                        for p in local_papers:
                            d = paper_to_dict(p)
                            d["paper_id"] = str(d.get("id") or "")
                            year_val = d.get("year")
                            if self.config.year_from is not None and isinstance(year_val, int):
                                if int(year_val) < int(self.config.year_from):
                                    continue
                            if self.config.year_to is not None and isinstance(year_val, int):
                                if int(year_val) > int(self.config.year_to):
                                    continue
                            raw.append(d)
                        Logger.info(
                            f"Local DB fallback returned {len(raw)} papers",
                            file=LogFiles.HARVEST,
                        )
                    except Exception as local_exc:
                        Logger.warning(
                            f"Local DB fallback failed: {local_exc}",
                            file=LogFiles.HARVEST,
                        )

                # Feedback filtering + dedup + relevance + year range
                seen_titles: set[str] = set()
                filtered: List[Dict[str, Any]] = []
                for p in raw:
                    pid = str(p.get("paper_id") or "").strip()
                    if pid and pid in disliked_ids:
                        continue
                    if not _is_academic_paper(p):
                        continue
                    # Year range filter (post-search safety net)
                    year_val = p.get("year")
                    if isinstance(year_val, int):
                        if self.config.year_from is not None and year_val < self.config.year_from:
                            continue
                        if self.config.year_to is not None and year_val > self.config.year_to:
                            continue
                    tkey = _normalize_title(str(p.get("title") or ""))
                    if tkey and tkey in seen_titles:
                        continue
                    if tkey:
                        seen_titles.add(tkey)
                    filtered.append(p)

                if self.config.personalized and routed_track:
                    try:
                        anchor_service = _get_anchor_service()
                        if anchor_service is not None:
                            numeric_ids = [
                                int(str(p.get("paper_id") or 0))
                                for p in filtered
                                if str(p.get("paper_id") or "").isdigit()
                            ]
                            if numeric_ids:
                                anchor_boosts = anchor_service.get_followed_paper_anchor_scores(
                                    user_id=user_id,
                                    track_id=int(routed_track["id"]),
                                    paper_ids=numeric_ids,
                                )
                                for paper_id, anchor_score in anchor_boosts.items():
                                    pid = str(paper_id)
                                    boosts[pid] = boosts.get(pid, 0.0) + min(
                                        0.35,
                                        0.20 * max(0.0, float(anchor_score)),
                                    )
                    except Exception as exc:
                        Logger.warning(
                            f"Failed to apply anchor boost: {exc}",
                            file=LogFiles.HARVEST,
                        )

                try:
                    self._attach_latest_judge(filtered)
                except Exception as exc:
                    Logger.warning(
                        f"Failed to attach latest judge scores: {exc}",
                        file=LogFiles.HARVEST,
                    )

                self._attach_feedback_flags(
                    filtered,
                    saved_ids=set(saved_ids),
                    liked_ids=set(liked_ids),
                )

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
            except Exception as e:
                import traceback

                tb = traceback.format_exc()
                Logger.error(f"Error fetching papers: {e}\n{tb}", file=LogFiles.HARVEST)
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
            "personalized": bool(self.config.personalized),
            "learned_source_weights": dict(learned_source_weights or {}),
            "year_from": self.config.year_from,
            "year_to": self.config.year_to,
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
        if self.search_service is not None:
            close_fn = getattr(self.search_service, "close", None)
            if callable(close_fn):
                try:
                    maybe_coro = close_fn()
                    if asyncio.iscoroutine(maybe_coro):
                        await maybe_coro
                except Exception:
                    pass

        if self.paper_store is not None:
            close_fn = getattr(self.paper_store, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
        return None
