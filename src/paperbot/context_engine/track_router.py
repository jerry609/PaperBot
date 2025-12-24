from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from paperbot.context_engine.embeddings import (
    EmbeddingConfig,
    EmbeddingProvider,
    try_build_default_embedding_provider,
)
from paperbot.infrastructure.stores.memory_store import SqlAlchemyMemoryStore
from paperbot.infrastructure.stores.research_store import SqlAlchemyResearchStore

_TOKEN_RX = re.compile(r"[a-zA-Z0-9_+.-]+")


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RX.findall(text or "") if t.strip()}


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def _track_keyword_score(query: str, track: Dict[str, Any]) -> int:
    q = _tokenize(query)
    if not q:
        return 0
    score = 0
    for bucket, weight in [
        ("keywords", 3),
        ("methods", 2),
        ("venues", 1),
    ]:
        for term in track.get(bucket) or []:
            tt = str(term or "").strip().lower()
            if not tt:
                continue
            if _tokenize(tt) & q:
                score += weight
    # Soft match in name/description.
    name_desc = _tokenize(f"{track.get('name') or ''} {track.get('description') or ''}")
    score += 1 if (name_desc & q) else 0
    return score


def _track_profile_text(
    track: Dict[str, Any], *, memories: List[Dict[str, Any]], tasks: List[Dict[str, Any]]
) -> str:
    parts: List[str] = []
    parts.append(f"Track: {track.get('name') or ''}".strip())
    if track.get("description"):
        parts.append(f"Description: {track.get('description')}")
    for label, key in [("Keywords", "keywords"), ("Methods", "methods"), ("Venues", "venues")]:
        vals = [str(x).strip() for x in (track.get(key) or []) if str(x).strip()]
        if vals:
            parts.append(f"{label}: {', '.join(vals[:32])}")
    if tasks:
        titles = [
            str(t.get("title") or "").strip() for t in tasks if str(t.get("title") or "").strip()
        ]
        if titles:
            parts.append("Tasks: " + " | ".join(titles[:10]))
    if memories:
        mem_texts = [
            str(m.get("content") or "").strip()
            for m in memories
            if str(m.get("content") or "").strip()
        ]
        if mem_texts:
            parts.append("Notes: " + " | ".join(mem_texts[:12]))
    return "\n".join(parts).strip()


def _sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TrackRouterConfig:
    use_embeddings: bool = True
    embedding_model: str = "text-embedding-3-small"
    min_switch_score: float = 0.10  # combined score threshold to suggest switching
    min_margin: float = 0.03  # how much better than active to suggest
    per_track_memory_hits: int = 3
    per_track_task_titles: int = 5
    embedding_profile_memory_limit: int = 12
    embedding_profile_task_limit: int = 8


class TrackRouter:
    def __init__(
        self,
        *,
        research_store: SqlAlchemyResearchStore,
        memory_store: SqlAlchemyMemoryStore,
        embedding_provider: Optional[EmbeddingProvider] = None,
        config: Optional[TrackRouterConfig] = None,
    ):
        self.research_store = research_store
        self.memory_store = memory_store
        self.config = config or TrackRouterConfig()
        self.embedding_provider = embedding_provider or try_build_default_embedding_provider(
            config=EmbeddingConfig(model=self.config.embedding_model)
        )

    def _profile_context(
        self, *, user_id: str, track_id: int
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Stable per-track context used for profile embeddings.

        Intentionally does NOT depend on the query; query-dependent signals are separate features
        (keyword overlap, memory hits, task overlap).
        """
        tasks = self.research_store.list_tasks(
            user_id=user_id, track_id=track_id, limit=self.config.embedding_profile_task_limit
        )
        memories = self.memory_store.list_memories(
            user_id=user_id,
            limit=self.config.embedding_profile_memory_limit,
            scope_type="track",
            scope_id=str(track_id),
            include_pending=False,
            include_deleted=False,
        )
        return memories, tasks

    def ensure_track_embedding(
        self, *, user_id: str, track: Dict[str, Any]
    ) -> Optional[List[float]]:
        if not self.config.use_embeddings or self.embedding_provider is None:
            return None
        track_id = int(track.get("id") or track.get("track_id") or 0)
        if track_id <= 0:
            return None

        memories, tasks = self._profile_context(user_id=user_id, track_id=track_id)
        profile_text = _track_profile_text(track, memories=memories, tasks=tasks)
        profile_hash = _sha256_text(profile_text)

        cached = self.research_store.get_track_embedding(
            track_id=track_id, model=self.config.embedding_model
        )
        if (
            cached
            and profile_hash
            and cached.get("text_hash") == profile_hash
            and cached.get("embedding")
        ):
            return cached.get("embedding")

        try:
            vec = self.embedding_provider.embed(profile_text)
        except Exception:
            vec = None
        if not vec:
            return None

        self.research_store.upsert_track_embedding(
            track_id=track_id,
            model=self.config.embedding_model,
            profile_text=profile_text,
            embedding=vec,
        )
        return vec

    def precompute_track_embeddings(
        self, *, user_id: str, track_ids: Optional[List[int]] = None, limit: int = 200
    ) -> Dict[str, int]:
        """
        Best-effort embedding precompute, intended for background refresh.

        Returns counts: {"considered": n, "updated": n, "skipped": n, "failed": n}
        """
        if not self.config.use_embeddings or self.embedding_provider is None:
            return {"considered": 0, "updated": 0, "skipped": 0, "failed": 0}

        tracks = self.research_store.list_tracks(
            user_id=user_id, include_archived=False, limit=limit
        )
        if track_ids:
            allow = {int(x) for x in track_ids if int(x) > 0}
            tracks = [t for t in tracks if int(t.get("id") or 0) in allow]

        considered = 0
        updated = 0
        skipped = 0
        failed = 0

        for t in tracks:
            track_id = int(t.get("id") or 0)
            if track_id <= 0:
                continue
            considered += 1
            try:
                memories, tasks = self._profile_context(user_id=user_id, track_id=track_id)
                profile_text = _track_profile_text(t, memories=memories, tasks=tasks)
                profile_hash = _sha256_text(profile_text)
                cached = self.research_store.get_track_embedding(
                    track_id=track_id, model=self.config.embedding_model
                )
                if (
                    cached
                    and profile_hash
                    and cached.get("text_hash") == profile_hash
                    and cached.get("embedding")
                ):
                    skipped += 1
                    continue

                vec = self.embedding_provider.embed(profile_text)
                if not vec:
                    failed += 1
                    continue
                self.research_store.upsert_track_embedding(
                    track_id=track_id,
                    model=self.config.embedding_model,
                    profile_text=profile_text,
                    embedding=vec,
                )
                updated += 1
            except Exception:
                failed += 1

        return {"considered": considered, "updated": updated, "skipped": skipped, "failed": failed}

    def suggest_track(
        self,
        *,
        user_id: str,
        query: str,
        active_track_id: Optional[int],
        limit: int = 50,
    ) -> Optional[Dict[str, Any]]:
        tracks = self.research_store.list_tracks(
            user_id=user_id, include_archived=False, limit=limit
        )
        if not tracks:
            return None

        active = None
        for t in tracks:
            if active_track_id is not None and int(t.get("id") or 0) == int(active_track_id):
                active = t
                break

        scored: List[Tuple[float, Dict[str, Any]]] = []

        query_embedding: Optional[List[float]] = None
        if self.config.use_embeddings and self.embedding_provider is not None:
            try:
                query_embedding = self.embedding_provider.embed(query)
            except Exception:
                query_embedding = None

        for t in tracks:
            track_id = int(t.get("id") or 0)
            if track_id <= 0:
                continue

            # Feature 1: keyword overlap
            kw_score = float(_track_keyword_score(query, t))

            # Feature 2: memory hits within this track
            mem_hits = self.memory_store.search_memories(
                user_id=user_id,
                query=query,
                limit=self.config.per_track_memory_hits,
                scope_type="track",
                scope_id=str(track_id),
            )
            mem_score = float(sum(float(m.get("score") or 0) for m in mem_hits))

            # Feature 3: task overlap (titles)
            tasks = self.research_store.list_tasks(
                user_id=user_id, track_id=track_id, limit=self.config.per_track_task_titles
            )
            task_text = " ".join(str(x.get("title") or "") for x in tasks)
            task_score = float(len(_tokenize(task_text) & _tokenize(query)))

            # Feature 4: embedding similarity between query and stable track profile
            emb_sim = 0.0
            if query_embedding is not None:
                vec = self.ensure_track_embedding(user_id=user_id, track=t)
                if vec:
                    emb_sim = float(_cosine(query_embedding, vec))

            # Combine (simple normalized blend)
            combined = (
                0.45 * min(1.0, kw_score / 10.0)
                + 0.25 * min(1.0, mem_score / 10.0)
                + 0.10 * min(1.0, task_score / 10.0)
                + 0.20 * max(0.0, min(1.0, emb_sim))
            )

            scored.append(
                (
                    combined,
                    {
                        "track_id": track_id,
                        "track_name": t.get("name"),
                        "score": combined,
                        "features": {
                            "keyword_score": kw_score,
                            "memory_score": mem_score,
                            "task_score": task_score,
                            "embedding_similarity": emb_sim,
                        },
                        "top_memory_hits": mem_hits[:2],
                    },
                )
            )

        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            return None

        best_score, best = scored[0]
        active_best = None
        active_score = 0.0
        if active is not None:
            for s, d in scored:
                if d.get("track_id") == int(active.get("id") or 0):
                    active_best = d
                    active_score = s
                    break

        # Suggest only when switching is clearly better.
        if active_track_id is not None and int(best.get("track_id") or 0) == int(active_track_id):
            return None

        margin = best_score - float(active_score or 0.0)
        if best_score < self.config.min_switch_score or margin < self.config.min_margin:
            return None

        best["active_track_id"] = active_track_id
        best["active_score"] = float(active_score or 0.0)
        best["margin"] = float(margin)
        best["runner_up"] = scored[1][1] if len(scored) > 1 else None
        return best
