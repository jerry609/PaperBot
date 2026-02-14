from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import and_, desc, func, or_, select

from paperbot.infrastructure.stores.models import (
    AuthorModel,
    PaperAuthorModel,
    PaperFeedbackModel,
    PaperModel,
    ResearchTrackModel,
    UserAnchorActionModel,
    UserAnchorScoreModel,
)
from paperbot.infrastructure.stores.sqlalchemy_db import SessionProvider, get_db_url


@dataclass
class _AuthorAggregate:
    author: AuthorModel
    paper_count: int
    citation_sum: int


def _parse_keywords(track: ResearchTrackModel) -> list[str]:
    try:
        rows = json.loads(track.keywords_json or "[]")
        if isinstance(rows, list):
            return [str(x).strip().lower() for x in rows if str(x).strip()]
    except Exception:
        pass
    return []


def _paper_text(paper: PaperModel) -> str:
    parts: list[str] = [paper.title or "", paper.abstract or ""]
    try:
        keywords = paper.get_keywords()
        if isinstance(keywords, list):
            parts.extend(str(x) for x in keywords)
    except Exception:
        pass
    return " ".join(parts).lower()


def _anchor_level(score: float) -> str:
    if score >= 0.75:
        return "core"
    if score >= 0.55:
        return "active"
    if score >= 0.35:
        return "emerging"
    return "background"


def _safe_json_obj(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text or "{}")
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


class AnchorService:
    """Discover anchor authors with intrinsic + relevance + network scoring."""

    def __init__(self, db_url: Optional[str] = None):
        self._provider = SessionProvider(db_url or get_db_url())

    def discover(
        self,
        *,
        track_id: int,
        user_id: str = "default",
        limit: int = 20,
        window_years: int = 5,
        personalized: bool = True,
    ) -> list[dict]:
        now_year = datetime.utcnow().year
        year_from = max(now_year - max(int(window_years), 1) + 1, 1970)

        with self._provider.session() as session:
            track = session.execute(
                select(ResearchTrackModel).where(ResearchTrackModel.id == int(track_id))
            ).scalar_one_or_none()
            if track is None:
                raise ValueError(f"track not found: {track_id}")
            keywords = _parse_keywords(track)

            aggregates = self._collect_author_aggregates(
                session, year_from=year_from, now_year=now_year
            )
            if not aggregates:
                return []

            aggregates = aggregates[: max(int(limit), 1) * 4]
            citation_map = {int(item.author.id): int(item.citation_sum) for item in aggregates}
            max_paper_count = max(x.paper_count for x in aggregates) or 1
            max_citation_sum = max(x.citation_sum for x in aggregates) or 1
            network_map = self._build_network_map(session, year_from=year_from, now_year=now_year)
            action_map = self.list_user_anchor_actions(
                user_id=user_id,
                track_id=int(track_id),
                author_ids=[int(item.author.id) for item in aggregates],
                session=session,
            )

            payload: list[dict] = []
            # TODO: N+1 query — batch-fetch author_papers and feedback_rows
            #  outside the loop to reduce DB roundtrips (PR #112 review).
            for item in aggregates:
                author_papers = (
                    session.execute(
                        select(PaperModel)
                        .join(PaperAuthorModel, PaperAuthorModel.paper_id == PaperModel.id)
                        .where(PaperAuthorModel.author_id == item.author.id)
                        .where(
                            or_(
                                PaperModel.year.is_(None),
                                and_(PaperModel.year >= year_from, PaperModel.year <= now_year),
                            )
                        )
                        .order_by(PaperModel.citation_count.desc(), PaperModel.id.desc())
                        .limit(25)
                    )
                    .scalars()
                    .all()
                )
                if not author_papers:
                    continue

                keyword_matches = []
                if keywords:
                    for paper in author_papers:
                        text = _paper_text(paper)
                        if any(k in text for k in keywords):
                            keyword_matches.append(paper)

                paper_ids = [int(p.id) for p in author_papers if p.id is not None]
                feedback_rows = []
                if paper_ids:
                    feedback_rows = (
                        session.execute(
                            select(PaperFeedbackModel)
                            .where(PaperFeedbackModel.track_id == int(track_id))
                            .where(PaperFeedbackModel.user_id == user_id)
                            .where(
                                or_(
                                    PaperFeedbackModel.canonical_paper_id.in_(paper_ids),
                                    PaperFeedbackModel.paper_ref_id.in_(paper_ids),
                                )
                            )
                        )
                        .scalars()
                        .all()
                    )

                paper_volume_score = float(item.paper_count) / float(max_paper_count)
                citation_score = float(item.citation_sum) / float(max_citation_sum)
                intrinsic_score = 0.5 * paper_volume_score + 0.5 * citation_score

                keyword_match_rate = (
                    float(len(keyword_matches)) / float(len(author_papers))
                    if author_papers
                    else 0.0
                )

                action_weights = {
                    "like": 1.0,
                    "save": 1.0,
                    "cite": 1.2,
                    "dislike": -1.0,
                    "skip": -0.3,
                }
                raw_feedback = 0.0
                for row in feedback_rows:
                    raw_feedback += action_weights.get((row.action or "").lower(), 0.0)
                feedback_signal = (
                    (math.tanh(raw_feedback / 4.0) + 1.0) / 2.0 if feedback_rows else 0.0
                )

                relevance_score = keyword_match_rate
                network_score = self._compute_network_score(
                    author_id=int(item.author.id),
                    network_map=network_map,
                    citation_map=citation_map,
                    max_citation_sum=max_citation_sum,
                )
                personalization_score = feedback_signal if personalized else 0.0

                anchor_score = (
                    0.45 * intrinsic_score
                    + 0.3 * relevance_score
                    + 0.15 * network_score
                    + 0.1 * personalization_score
                )
                level = _anchor_level(anchor_score)

                item.author.anchor_score = float(round(anchor_score, 6))
                item.author.anchor_level = level
                item.author.paper_count = item.paper_count
                item.author.citation_count = item.citation_sum

                evidence = (keyword_matches or author_papers)[:3]
                evidence_rows = [
                    {
                        "paper_id": int(p.id),
                        "title": p.title,
                        "year": p.year,
                        "url": p.url,
                        "citation_count": int(p.citation_count or 0),
                    }
                    for p in evidence
                ]
                evidence_status = "ok" if evidence_rows else "missing"
                evidence_note = (
                    None
                    if evidence_rows
                    else "No direct evidence papers found in current window; consider broadening keywords or window_days."
                )

                score_breakdown = {
                    "intrinsic": float(round(intrinsic_score, 4)),
                    "relevance": float(round(relevance_score, 4)),
                    "network": float(round(network_score, 4)),
                    "personalization": float(round(personalization_score, 4)),
                    "total": float(round(anchor_score, 4)),
                }

                payload.append(
                    {
                        "author_id": int(item.author.id),
                        "author_ref": item.author.author_id,
                        "name": item.author.name,
                        "slug": item.author.slug,
                        "anchor_score": float(round(anchor_score, 4)),
                        "anchor_level": level,
                        "intrinsic_score": float(round(intrinsic_score, 4)),
                        "relevance_score": float(round(relevance_score, 4)),
                        "network_score": float(round(network_score, 4)),
                        "paper_count": int(item.paper_count),
                        "citation_sum": int(item.citation_sum),
                        "keyword_match_rate": float(round(keyword_match_rate, 4)),
                        "feedback_signal": float(round(feedback_signal, 4)),
                        "user_action": action_map.get(int(item.author.id)),
                        "score_breakdown": score_breakdown,
                        "evidence_status": evidence_status,
                        "evidence_note": evidence_note,
                        "evidence_papers": evidence_rows,
                    }
                )

                if personalized:
                    self._upsert_user_anchor_score(
                        session,
                        user_id=user_id,
                        track_id=int(track_id),
                        author_id=int(item.author.id),
                        score=float(round(anchor_score, 6)),
                        breakdown=score_breakdown,
                    )

            session.commit()

        payload.sort(
            key=lambda row: (
                row["anchor_score"],
                row["network_score"],
                row["relevance_score"],
                row["paper_count"],
            ),
            reverse=True,
        )
        return payload[: max(int(limit), 1)]

    def recompute_author_network_scores(self, *, window_years: int = 5) -> dict[str, int]:
        now_year = datetime.utcnow().year
        year_from = max(now_year - max(int(window_years), 1) + 1, 1970)

        with self._provider.session() as session:
            aggregates = self._collect_author_aggregates(
                session, year_from=year_from, now_year=now_year
            )
            if not aggregates:
                return {"authors": 0, "updated": 0}

            citation_map = {int(item.author.id): int(item.citation_sum) for item in aggregates}
            max_citation_sum = max((item.citation_sum for item in aggregates), default=1) or 1
            network_map = self._build_network_map(session, year_from=year_from, now_year=now_year)

            updated = 0
            for item in aggregates:
                author_id = int(item.author.id)
                score = self._compute_network_score(
                    author_id=author_id,
                    network_map=network_map,
                    citation_map=citation_map,
                    max_citation_sum=max_citation_sum,
                )
                metadata = _safe_json_obj(item.author.metadata_json)
                metadata["network_score"] = float(round(score, 6))
                metadata["network_score_window_years"] = int(window_years)
                metadata["network_score_updated_at"] = datetime.now(timezone.utc).isoformat()
                item.author.metadata_json = json.dumps(metadata, ensure_ascii=False)
                updated += 1

            session.commit()
            return {"authors": len(aggregates), "updated": updated}

    def set_user_anchor_action(
        self,
        *,
        user_id: str,
        track_id: int,
        author_id: int,
        action: str,
    ) -> dict[str, Any]:
        normalized_action = str(action or "").strip().lower()
        if normalized_action not in {"follow", "ignore"}:
            raise ValueError("action must be 'follow' or 'ignore'")

        now = datetime.now(timezone.utc)
        with self._provider.session() as session:
            row = session.execute(
                select(UserAnchorActionModel).where(
                    UserAnchorActionModel.user_id == str(user_id),
                    UserAnchorActionModel.track_id == int(track_id),
                    UserAnchorActionModel.author_id == int(author_id),
                )
            ).scalar_one_or_none()

            if row is None:
                row = UserAnchorActionModel(
                    user_id=str(user_id),
                    track_id=int(track_id),
                    author_id=int(author_id),
                    action=normalized_action,
                    created_at=now,
                    updated_at=now,
                )
                session.add(row)
            else:
                row.action = normalized_action
                row.updated_at = now

            session.commit()
            session.refresh(row)

            return {
                "user_id": row.user_id,
                "track_id": int(row.track_id),
                "author_id": int(row.author_id),
                "action": row.action,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            }

    def get_user_anchor_actions(self, *, user_id: str, track_id: int) -> list[dict[str, Any]]:
        with self._provider.session() as session:
            rows = (
                session.execute(
                    select(UserAnchorActionModel)
                    .where(UserAnchorActionModel.user_id == str(user_id))
                    .where(UserAnchorActionModel.track_id == int(track_id))
                    .order_by(
                        UserAnchorActionModel.updated_at.desc(), UserAnchorActionModel.id.desc()
                    )
                )
                .scalars()
                .all()
            )
            return [
                {
                    "user_id": row.user_id,
                    "track_id": int(row.track_id),
                    "author_id": int(row.author_id),
                    "action": row.action,
                    "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                }
                for row in rows
            ]

    def get_followed_paper_anchor_scores(
        self,
        *,
        user_id: str,
        track_id: int,
        paper_ids: list[int],
    ) -> dict[int, float]:
        """Return max personalized anchor score for followed authors per paper."""
        clean_ids = sorted({int(pid) for pid in paper_ids if int(pid) > 0})
        if not clean_ids:
            return {}

        with self._provider.session() as session:
            rows = session.execute(
                select(
                    PaperAuthorModel.paper_id,
                    func.max(func.coalesce(UserAnchorScoreModel.personalized_anchor_score, 0.0)),
                )
                .join(
                    UserAnchorActionModel,
                    and_(
                        UserAnchorActionModel.author_id == PaperAuthorModel.author_id,
                        UserAnchorActionModel.user_id == str(user_id),
                        UserAnchorActionModel.track_id == int(track_id),
                        UserAnchorActionModel.action == "follow",
                    ),
                )
                .outerjoin(
                    UserAnchorScoreModel,
                    and_(
                        UserAnchorScoreModel.user_id == UserAnchorActionModel.user_id,
                        UserAnchorScoreModel.track_id == UserAnchorActionModel.track_id,
                        UserAnchorScoreModel.author_id == UserAnchorActionModel.author_id,
                    ),
                )
                .where(PaperAuthorModel.paper_id.in_(clean_ids))
                .group_by(PaperAuthorModel.paper_id)
            ).all()

            out: dict[int, float] = {}
            for paper_id, score in rows:
                if paper_id is None:
                    continue
                out[int(paper_id)] = max(0.0, float(score or 0.0))
            return out

    @staticmethod
    def list_user_anchor_actions(
        *,
        user_id: str,
        track_id: int,
        author_ids: list[int],
        session,
    ) -> dict[int, str]:
        if not author_ids:
            return {}

        rows = (
            session.execute(
                select(UserAnchorActionModel)
                .where(UserAnchorActionModel.user_id == str(user_id))
                .where(UserAnchorActionModel.track_id == int(track_id))
                .where(UserAnchorActionModel.author_id.in_(author_ids))
            )
            .scalars()
            .all()
        )
        return {int(row.author_id): str(row.action) for row in rows if row.author_id is not None}

    @staticmethod
    def _upsert_user_anchor_score(
        session,
        *,
        user_id: str,
        track_id: int,
        author_id: int,
        score: float,
        breakdown: dict[str, Any],
    ) -> None:
        row = session.execute(
            select(UserAnchorScoreModel).where(
                UserAnchorScoreModel.user_id == str(user_id),
                UserAnchorScoreModel.track_id == int(track_id),
                UserAnchorScoreModel.author_id == int(author_id),
            )
        ).scalar_one_or_none()

        if row is None:
            row = UserAnchorScoreModel(
                user_id=str(user_id),
                track_id=int(track_id),
                author_id=int(author_id),
                personalized_anchor_score=float(score),
                breakdown_json=json.dumps(breakdown, ensure_ascii=False),
                computed_at=datetime.now(timezone.utc),
            )
            session.add(row)
            return

        row.personalized_anchor_score = float(score)
        row.breakdown_json = json.dumps(breakdown, ensure_ascii=False)
        row.computed_at = datetime.now(timezone.utc)

    @staticmethod
    def _collect_author_aggregates(
        session,
        *,
        year_from: int,
        now_year: int,
    ) -> list[_AuthorAggregate]:
        rows = session.execute(
            select(
                AuthorModel,
                func.count(PaperAuthorModel.paper_id).label("paper_count"),
                func.sum(func.coalesce(PaperModel.citation_count, 0)).label("citation_sum"),
            )
            .join(PaperAuthorModel, PaperAuthorModel.author_id == AuthorModel.id)
            .join(PaperModel, PaperModel.id == PaperAuthorModel.paper_id)
            .where(
                or_(
                    PaperModel.year.is_(None),
                    and_(PaperModel.year >= year_from, PaperModel.year <= now_year),
                )
            )
            .group_by(AuthorModel.id)
            .order_by(desc("citation_sum"), desc("paper_count"))
        ).all()

        aggregates: list[_AuthorAggregate] = []
        for author, paper_count, citation_sum in rows:
            aggregates.append(
                _AuthorAggregate(
                    author=author,
                    paper_count=max(int(paper_count or 0), 0),
                    citation_sum=max(int(citation_sum or 0), 0),
                )
            )
        return aggregates

    @staticmethod
    def _build_network_map(
        session,
        *,
        year_from: int,
        now_year: int,
    ) -> dict[int, list[tuple[int, int]]]:
        # Build coauthor graph via paper-level author co-occurrence.
        paper_rows = session.execute(
            select(PaperAuthorModel.paper_id, PaperAuthorModel.author_id)
            .join(PaperModel, PaperModel.id == PaperAuthorModel.paper_id)
            .where(
                or_(
                    PaperModel.year.is_(None),
                    and_(PaperModel.year >= year_from, PaperModel.year <= now_year),
                )
            )
        ).all()

        authors_by_paper: dict[int, set[int]] = defaultdict(set)
        all_authors: set[int] = set()
        for paper_id, author_id in paper_rows:
            if paper_id is None or author_id is None:
                continue
            pid = int(paper_id)
            aid = int(author_id)
            authors_by_paper[pid].add(aid)
            all_authors.add(aid)

        coauthor_counts: dict[tuple[int, int], int] = defaultdict(int)
        for authors in authors_by_paper.values():
            author_list = sorted(authors)
            for i, left in enumerate(author_list):
                for right in author_list[i + 1 :]:
                    coauthor_counts[(left, right)] += 1
                    coauthor_counts[(right, left)] += 1

        network_map: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for (author_id, coauthor_id), collab_count in coauthor_counts.items():
            network_map[int(author_id)].append((int(coauthor_id), int(collab_count)))

        for author_id in all_authors:
            network_map.setdefault(int(author_id), [])

        return dict(network_map)

    @staticmethod
    def _compute_network_score(
        *,
        author_id: int,
        network_map: dict[int, list[tuple[int, int]]],
        citation_map: dict[int, int],
        max_citation_sum: int,
    ) -> float:
        neighbors = network_map.get(int(author_id), [])
        if not neighbors:
            return 0.0

        scores: list[float] = []
        for neighbor_id, collab_count in neighbors:
            neighbor_citations = float(citation_map.get(int(neighbor_id), 0)) / float(
                max_citation_sum or 1
            )
            collab_weight = min(float(collab_count) / 3.0, 1.0)
            scores.append(0.7 * neighbor_citations + 0.3 * collab_weight)

        if not scores:
            return 0.0

        breadth_factor = 1.0 - math.exp(-float(len(neighbors)) / 3.0)
        return max(min(float(sum(scores) / len(scores)) * breadth_factor, 1.0), 0.0)
