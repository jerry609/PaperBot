from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sqlalchemy import Integer, String, cast, desc, func, or_, select

from paperbot.domain.harvest import HarvestedPaper, HarvestSource
from paperbot.domain.identity import PaperIdentity
from paperbot.domain.paper_identity import normalize_arxiv_id, normalize_doi
from paperbot.infrastructure.stores.models import (
    Base,
    HarvestRunModel,
    PaperFeedbackModel,
    PaperIdentifierModel,
    PaperJudgeScoreModel,
    PaperModel,
)
from paperbot.infrastructure.stores.sqlalchemy_db import SessionProvider, get_db_url
from paperbot.infrastructure.stores.author_store import AuthorStore
from paperbot.utils.logging_config import LogFiles, Logger

USE_CANONICAL_FK = os.getenv("PAPERBOT_USE_CANONICAL_FK", "false").lower() == "true"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _safe_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    return [str(v).strip() for v in values if str(v).strip()]


def _parse_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not value:
        return None

    text = str(value).strip()
    if not text:
        return None

    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _as_utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class SqlAlchemyPaperStore:
    """Canonical paper registry with idempotent upsert for daily workflows."""


@dataclass
class LibraryPaper:
    """Paper with library metadata (saved_at, track_id, action)."""

    paper: PaperModel
    saved_at: datetime
    track_id: Optional[int]
    action: str


class PaperStore:
    """
    Paper storage repository.

    Handles:
    - Batch upsert with DB-level deduplication
    - Filter-based search with pagination
    - Source tracking
    - User library (saved papers)
    """

    def __init__(self, db_url: Optional[str] = None, *, auto_create_schema: bool = True):
        self.db_url = db_url or get_db_url()
        self._provider = SessionProvider(self.db_url)
        if auto_create_schema:
            Base.metadata.create_all(self._provider.engine)
        self._author_store = AuthorStore(db_url=self.db_url, auto_create_schema=auto_create_schema)

    def upsert_paper(
        self,
        *,
        paper: Dict[str, Any],
        source_hint: Optional[str] = None,
        seen_at: Optional[datetime] = None,
        sync_authors: bool = True,
    ) -> Dict[str, Any]:
        now = _utcnow()

        title = str(paper.get("title") or "").strip()
        url = str(paper.get("url") or "").strip()
        external_url = str(paper.get("external_url") or "").strip()
        pdf_url = str(paper.get("pdf_url") or "").strip()
        abstract = str(paper.get("snippet") or paper.get("abstract") or "").strip()

        arxiv_id = (
            normalize_arxiv_id(paper.get("arxiv_id"))
            or normalize_arxiv_id(paper.get("paper_id"))
            or normalize_arxiv_id(url)
            or normalize_arxiv_id(external_url)
            or normalize_arxiv_id(pdf_url)
        )
        doi = (
            normalize_doi(paper.get("doi"))
            or normalize_doi(url)
            or normalize_doi(external_url)
            or normalize_doi(pdf_url)
        )

        # Extract S2/OpenAlex IDs from identities list (PaperCandidate format)
        semantic_scholar_id = paper.get("semantic_scholar_id")
        openalex_id = paper.get("openalex_id")
        for ident in paper.get("identities") or []:
            src = str(ident.get("source") or "").strip()
            ext_id = str(ident.get("external_id") or "").strip()
            if not ext_id:
                continue
            if src == "semantic_scholar" and not semantic_scholar_id:
                semantic_scholar_id = ext_id
            elif src == "openalex" and not openalex_id:
                openalex_id = ext_id
            elif src == "arxiv" and not arxiv_id:
                arxiv_id = normalize_arxiv_id(ext_id)
            elif src == "doi" and not doi:
                doi = normalize_doi(ext_id)

        source = (
            source_hint
            or (paper.get("sources") or [None])[0]
            or paper.get("source")
            or paper.get("primary_source")
            or "papers_cool"
        )
        venue = str(paper.get("subject_or_venue") or paper.get("venue") or "").strip()
        publication_date = str(
            paper.get("publication_date")
            or paper.get("published_at")
            or paper.get("published")
            or ""
        ).strip()

        year_raw = (
            paper.get("year") if paper.get("year") is not None else paper.get("published_year")
        )
        try:
            year = int(year_raw) if year_raw is not None else None
        except Exception:
            year = None

        citation_raw = paper.get("citation_count")
        try:
            citation_count = int(citation_raw) if citation_raw is not None else 0
        except Exception:
            citation_count = 0

        authors = _safe_list(paper.get("authors"))
        keywords = _safe_list(paper.get("keywords"))
        fields_of_study = _safe_list(paper.get("fields_of_study"))

        normalized_title = title.lower().strip() or "untitled"
        title_hash = hashlib.sha256(normalized_title.encode("utf-8")).hexdigest()

        with self._provider.session() as session:
            row = None
            if arxiv_id:
                row = session.execute(
                    select(PaperModel).where(PaperModel.arxiv_id == arxiv_id)
                ).scalar_one_or_none()
            if row is None and doi:
                row = session.execute(
                    select(PaperModel).where(PaperModel.doi == doi)
                ).scalar_one_or_none()
            if row is None and semantic_scholar_id:
                row = session.execute(
                    select(PaperModel).where(PaperModel.semantic_scholar_id == semantic_scholar_id)
                ).scalar_one_or_none()
            if row is None and openalex_id:
                row = session.execute(
                    select(PaperModel).where(PaperModel.openalex_id == openalex_id)
                ).scalar_one_or_none()
            if row is None and url:
                row = session.execute(
                    select(PaperModel).where(PaperModel.url == url)
                ).scalar_one_or_none()
            if row is None and title:
                row = (
                    session.execute(
                        select(PaperModel)
                        .where(func.lower(PaperModel.title) == title.lower())
                        .limit(1)
                    )
                    .scalars()
                    .first()
                )

            created = row is None
            if row is None:
                row = PaperModel(
                    title_hash=title_hash,
                    first_seen_at=seen_at or now,
                    created_at=now,
                    updated_at=now,
                )
                session.add(row)

            if arxiv_id:
                row.arxiv_id = arxiv_id
            if doi:
                row.doi = doi
            if semantic_scholar_id:
                row.semantic_scholar_id = semantic_scholar_id
            if openalex_id:
                row.openalex_id = openalex_id
            row.title_hash = title_hash
            row.title = title or row.title or ""
            row.abstract = abstract or row.abstract or ""
            row.url = url or row.url or None
            row.pdf_url = pdf_url or row.pdf_url or None
            row.venue = venue or row.venue or ""
            row.year = year if year is not None else row.year
            row.publication_date = publication_date or row.publication_date
            row.citation_count = max(citation_count, int(row.citation_count or 0))

            if authors:
                row.authors_json = json.dumps(authors, ensure_ascii=False)
            if keywords:
                row.keywords_json = json.dumps(keywords, ensure_ascii=False)
            if fields_of_study:
                row.fields_of_study_json = json.dumps(fields_of_study, ensure_ascii=False)

            source_text = str(source or "").strip() or "papers_cool"
            row.primary_source = source_text
            existing_sources = row.get_sources()
            merged_sources = (
                sorted({*existing_sources, source_text}) if source_text else existing_sources
            )
            row.set_sources(merged_sources)

            row.updated_at = now
            session.commit()
            session.refresh(row)

            payload = self._paper_to_dict(row)
            payload["_created"] = created

            # Dual-write: also populate paper_identifiers
            self._sync_identifiers(session, row)

            # Extract and link authors to authors/paper_authors tables
            if sync_authors and authors and row.id:
                try:
                    self._author_store.replace_paper_authors(
                        paper_id=int(row.id),
                        authors=authors,
                    )
                except Exception as e:
                    Logger.warning(
                        f"Failed to sync paper authors for paper {row.id}: {e}",
                        file=LogFiles.HARVEST,
                    )

            return payload

    @staticmethod
    def _sync_identifiers(session, row: PaperModel) -> None:
        """Write known external IDs to paper_identifiers (idempotent)."""
        pairs: list[tuple[str, str]] = []
        if row.semantic_scholar_id:
            pairs.append(("semantic_scholar", row.semantic_scholar_id))
        if row.arxiv_id:
            pairs.append(("arxiv", row.arxiv_id))
        if row.openalex_id:
            pairs.append(("openalex", row.openalex_id))
        if row.doi:
            pairs.append(("doi", row.doi))

        for source, eid in pairs:
            existing = session.execute(
                select(PaperIdentifierModel).where(
                    PaperIdentifierModel.source == source,
                    PaperIdentifierModel.external_id == eid,
                )
            ).scalar_one_or_none()
            if existing is None:
                session.add(
                    PaperIdentifierModel(
                        paper_id=row.id,
                        source=source,
                        external_id=eid,
                        created_at=_utcnow(),
                    )
                )
            elif existing.paper_id != row.id:
                existing.paper_id = row.id
        try:
            session.flush()
        except Exception:
            session.rollback()

    def upsert_many(
        self,
        *,
        papers: Iterable[Dict[str, Any]],
        source_hint: Optional[str] = None,
        seen_at: Optional[datetime] = None,
    ) -> Dict[str, int]:
        created = 0
        updated = 0
        total = 0

        for paper in papers:
            if not isinstance(paper, dict):
                continue
            result = self.upsert_paper(paper=paper, source_hint=source_hint, seen_at=seen_at)
            total += 1
            if result.get("_created"):
                created += 1
            else:
                updated += 1

        return {"total": total, "created": created, "updated": updated}

    def list_recent(self, *, limit: int = 50, source: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._provider.session() as session:
            stmt = select(PaperModel).where(PaperModel.deleted_at.is_(None))
            if source:
                stmt = stmt.where(PaperModel.primary_source == source)
            stmt = stmt.order_by(desc(PaperModel.updated_at), desc(PaperModel.id)).limit(
                max(1, int(limit))
            )
            rows = session.execute(stmt).scalars().all()
            return [self._paper_to_dict(row) for row in rows]

    def upsert_judge_scores_from_report(self, report: Dict[str, Any]) -> Dict[str, int]:
        now = _utcnow()
        scored_at = _parse_datetime(report.get("generated_at")) or now

        created = 0
        updated = 0
        total = 0

        for query in report.get("queries") or []:
            query_name = str(query.get("normalized_query") or query.get("raw_query") or "").strip()
            if not query_name:
                continue
            for item in query.get("top_items") or []:
                if not isinstance(item, dict):
                    continue
                judge = item.get("judge")
                if not isinstance(judge, dict):
                    continue

                paper_row = self.upsert_paper(
                    paper=item,
                    source_hint=(report.get("sources") or [report.get("source")])[0],
                    seen_at=scored_at,
                )
                paper_db_id = int(paper_row.get("id") or 0)
                if paper_db_id <= 0:
                    continue

                total += 1
                with self._provider.session() as session:
                    row = session.execute(
                        select(PaperJudgeScoreModel).where(
                            PaperJudgeScoreModel.paper_id == paper_db_id,
                            PaperJudgeScoreModel.query == query_name,
                        )
                    ).scalar_one_or_none()

                    was_created = row is None
                    if row is None:
                        row = PaperJudgeScoreModel(
                            paper_id=paper_db_id,
                            query=query_name,
                            scored_at=scored_at,
                        )
                        session.add(row)

                    row.overall = _safe_float(judge.get("overall"))
                    row.relevance = _safe_float((judge.get("relevance") or {}).get("score"))
                    row.novelty = _safe_float((judge.get("novelty") or {}).get("score"))
                    row.rigor = _safe_float((judge.get("rigor") or {}).get("score"))
                    row.impact = _safe_float((judge.get("impact") or {}).get("score"))
                    row.clarity = _safe_float((judge.get("clarity") or {}).get("score"))
                    row.recommendation = str(judge.get("recommendation") or "")
                    row.one_line_summary = str(judge.get("one_line_summary") or "")
                    row.judge_model = str(judge.get("judge_model") or "")
                    try:
                        row.judge_cost_tier = (
                            int(judge.get("judge_cost_tier"))
                            if judge.get("judge_cost_tier") is not None
                            else None
                        )
                    except (ValueError, TypeError):
                        row.judge_cost_tier = None
                    row.scored_at = scored_at
                    row.metadata_json = "{}"

                    session.commit()
                    if was_created:
                        created += 1
                    else:
                        updated += 1

        return {"total": total, "created": created, "updated": updated}

    @staticmethod
    def _paper_to_dict(row: PaperModel) -> Dict[str, Any]:
        publication_date = row.publication_date
        published_at = publication_date

        return {
            "id": int(row.id),
            "arxiv_id": row.arxiv_id,
            "doi": row.doi,
            "semantic_scholar_id": row.semantic_scholar_id,
            "openalex_id": row.openalex_id,
            "title": row.title,
            "authors": row.get_authors(),
            "abstract": row.abstract,
            "url": row.url,
            "external_url": row.url,
            "pdf_url": row.pdf_url,
            "source": row.primary_source,
            "primary_source": row.primary_source,
            "venue": row.venue,
            "year": row.year,
            "publication_date": publication_date,
            "published_at": published_at,
            "first_seen_at": (row.first_seen_at or row.created_at).isoformat() if (row.first_seen_at or row.created_at) else None,
            "keywords": row.get_keywords(),
            "fields_of_study": row.get_fields_of_study(),
            "sources": row.get_sources(),
            "citation_count": int(row.citation_count or 0),
            "metadata": {},
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }

    def upsert_papers_batch(
        self,
        papers: List[HarvestedPaper],
    ) -> Tuple[int, int]:
        """
        Upsert papers with deduplication.

        Returns:
            Tuple of (new_count, updated_count)
        """
        Logger.info(f"Starting batch upsert for {len(papers)} papers", file=LogFiles.HARVEST)
        new_count = 0
        updated_count = 0
        now = _utcnow()

        with self._provider.session() as session:
            for paper in papers:
                Logger.info("Checking for existing paper in database", file=LogFiles.HARVEST)
                existing = self._find_existing(session, paper)

                if existing:
                    Logger.info("Found existing paper, updating metadata", file=LogFiles.HARVEST)
                    self._update_paper(existing, paper, now)
                    updated_count += 1
                else:
                    Logger.info(
                        "No existing paper found, creating new record", file=LogFiles.HARVEST
                    )
                    model = self._create_model(paper, now)
                    session.add(model)
                    new_count += 1

            Logger.info("Committing transaction to database", file=LogFiles.HARVEST)
            session.flush()

            # Dual-write identifiers for all papers in this batch
            for paper in papers:
                existing = self._find_existing(session, paper)
                if existing:
                    self._sync_identifiers(session, existing)

            session.commit()

        Logger.info(
            f"Batch upsert complete: {new_count} new, {updated_count} updated",
            file=LogFiles.HARVEST,
        )
        return new_count, updated_count

    def _find_existing(self, session, paper: HarvestedPaper) -> Optional[PaperModel]:
        """Find existing paper by canonical identifiers."""
        # Try each identifier in priority order
        if paper.doi:
            result = session.execute(
                select(PaperModel).where(PaperModel.doi == paper.doi)
            ).scalar_one_or_none()
            if result:
                return result

        if paper.arxiv_id:
            result = session.execute(
                select(PaperModel).where(PaperModel.arxiv_id == paper.arxiv_id)
            ).scalar_one_or_none()
            if result:
                return result

        if paper.semantic_scholar_id:
            result = session.execute(
                select(PaperModel).where(
                    PaperModel.semantic_scholar_id == paper.semantic_scholar_id
                )
            ).scalar_one_or_none()
            if result:
                return result

        if paper.openalex_id:
            result = session.execute(
                select(PaperModel).where(PaperModel.openalex_id == paper.openalex_id)
            ).scalar_one_or_none()
            if result:
                return result

        # Fallback to title hash
        title_hash = paper.compute_title_hash()
        result = session.execute(
            select(PaperModel).where(PaperModel.title_hash == title_hash)
        ).scalar_one_or_none()
        return result

    def _create_model(self, paper: HarvestedPaper, now: datetime) -> PaperModel:
        """Create a new PaperModel from HarvestedPaper."""
        return PaperModel(
            doi=paper.doi,
            arxiv_id=paper.arxiv_id,
            semantic_scholar_id=paper.semantic_scholar_id,
            openalex_id=paper.openalex_id,
            title_hash=paper.compute_title_hash(),
            title=paper.title,
            abstract=paper.abstract,
            authors_json=json.dumps(paper.authors, ensure_ascii=False),
            year=paper.year,
            venue=paper.venue,
            publication_date=paper.publication_date,
            citation_count=paper.citation_count,
            url=paper.url,
            pdf_url=paper.pdf_url,
            keywords_json=json.dumps(paper.keywords, ensure_ascii=False),
            fields_of_study_json=json.dumps(paper.fields_of_study, ensure_ascii=False),
            primary_source=paper.source.value,
            sources_json=json.dumps([paper.source.value], ensure_ascii=False),
            created_at=now,
            updated_at=now,
        )

    def _update_paper(self, existing: PaperModel, paper: HarvestedPaper, now: datetime) -> None:
        """Update existing paper with new data."""
        # Fill in missing identifiers
        if not existing.doi and paper.doi:
            existing.doi = paper.doi
        if not existing.arxiv_id and paper.arxiv_id:
            existing.arxiv_id = paper.arxiv_id
        if not existing.semantic_scholar_id and paper.semantic_scholar_id:
            existing.semantic_scholar_id = paper.semantic_scholar_id
        if not existing.openalex_id and paper.openalex_id:
            existing.openalex_id = paper.openalex_id

        # Prefer longer abstract
        if len(paper.abstract) > len(existing.abstract or ""):
            existing.abstract = paper.abstract

        # Prefer higher citation count
        if paper.citation_count > (existing.citation_count or 0):
            existing.citation_count = paper.citation_count

        # Fill in missing metadata
        if not existing.year and paper.year:
            existing.year = paper.year
        if not existing.venue and paper.venue:
            existing.venue = paper.venue
        if not existing.publication_date and paper.publication_date:
            existing.publication_date = paper.publication_date
        if not existing.url and paper.url:
            existing.url = paper.url
        if not existing.pdf_url and paper.pdf_url:
            existing.pdf_url = paper.pdf_url

        # Merge sources
        sources = existing.get_sources()
        if paper.source.value not in sources:
            sources.append(paper.source.value)
            existing.set_sources(sources)

        # Merge keywords and fields
        keywords = set(existing.get_keywords() + paper.keywords)
        existing.set_keywords(list(keywords))

        fields = set(existing.get_fields_of_study() + paper.fields_of_study)
        existing.set_fields_of_study(list(fields))

        existing.updated_at = now

    def search_papers(
        self,
        *,
        query: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        venues: Optional[List[str]] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        min_citations: Optional[int] = None,
        sources: Optional[List[str]] = None,
        sort_by: str = "citation_count",
        sort_order: str = "desc",
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[PaperModel], int]:
        """
        Search papers with filters and pagination.

        Returns:
            Tuple of (papers, total_count)
        """
        # Whitelist of allowed sort columns for security
        allowed_sort_columns = {
            "citation_count": PaperModel.citation_count,
            "year": PaperModel.year,
            "created_at": PaperModel.created_at,
            "updated_at": PaperModel.updated_at,
            "title": PaperModel.title,
        }

        with self._provider.session() as session:
            stmt = select(PaperModel).where(PaperModel.deleted_at.is_(None))

            # Full-text search (LIKE for v1)
            if query:
                pattern = f"%{query}%"
                stmt = stmt.where(
                    or_(
                        PaperModel.title.ilike(pattern),
                        PaperModel.abstract.ilike(pattern),
                    )
                )

            # Keyword filter (search in keywords_json)
            if keywords:
                keyword_conditions = [PaperModel.keywords_json.ilike(f"%{kw}%") for kw in keywords]
                stmt = stmt.where(or_(*keyword_conditions))

            # Year filters (use explicit None check to allow year_from=0 if needed)
            if year_from is not None:
                stmt = stmt.where(PaperModel.year >= year_from)
            if year_to is not None:
                stmt = stmt.where(PaperModel.year <= year_to)

            # Citation filter (use explicit None check to allow min_citations=0)
            if min_citations is not None:
                stmt = stmt.where(PaperModel.citation_count >= min_citations)

            # Venue filter
            if venues:
                venue_conditions = [PaperModel.venue.ilike(f"%{v}%") for v in venues]
                stmt = stmt.where(or_(*venue_conditions))

            # Source filter
            if sources:
                stmt = stmt.where(PaperModel.primary_source.in_(sources))

            # Count total before pagination
            count_stmt = select(func.count()).select_from(stmt.subquery())
            total_count = session.execute(count_stmt).scalar() or 0

            # Sort (use whitelist for security)
            sort_col = allowed_sort_columns.get(sort_by, PaperModel.citation_count)
            if sort_order.lower() == "desc":
                stmt = stmt.order_by(sort_col.desc())
            else:
                stmt = stmt.order_by(sort_col.asc())

            # Pagination
            stmt = stmt.offset(offset).limit(limit)

            papers = session.execute(stmt).scalars().all()

            return list(papers), total_count

    def get_paper_by_id(self, paper_id: int) -> Optional[PaperModel]:
        """Get a paper by its ID."""
        with self._provider.session() as session:
            return session.execute(
                select(PaperModel).where(
                    PaperModel.id == paper_id,
                    PaperModel.deleted_at.is_(None),
                )
            ).scalar_one_or_none()

    def get_paper_by_source_id(self, source: HarvestSource, source_id: str) -> Optional[PaperModel]:
        """
        Get a paper by its source-specific ID.

        Args:
            source: The harvest source (ARXIV, SEMANTIC_SCHOLAR, OPENALEX)
            source_id: The ID from that source

        Returns:
            PaperModel if found, None otherwise
        """
        with self._provider.session() as session:
            if source == HarvestSource.ARXIV:
                condition = PaperModel.arxiv_id == source_id
            elif source == HarvestSource.OPENALEX:
                condition = PaperModel.openalex_id == source_id
            else:  # Default to SEMANTIC_SCHOLAR
                condition = PaperModel.semantic_scholar_id == source_id

            return session.execute(
                select(PaperModel).where(condition, PaperModel.deleted_at.is_(None))
            ).scalar_one_or_none()

    def get_user_library(
        self,
        user_id: str,
        *,
        track_id: Optional[int] = None,
        actions: Optional[List[str]] = None,
        sort_by: str = "saved_at",
        sort_order: str = "desc",
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[LibraryPaper], int]:
        """
        Get papers in user's library (saved papers).

        Joins papers table with paper_feedback where action in actions.
        """
        Logger.info("Starting to fetch user library", file=LogFiles.HARVEST)
        if actions is None:
            actions = ["save"]

        if USE_CANONICAL_FK:
            return self._get_user_library_canonical(
                user_id,
                session_provider=self._provider,
                track_id=track_id,
                actions=actions,
                sort_by=sort_by,
                sort_order=sort_order,
                limit=limit,
                offset=offset,
            )

        with self._provider.session() as session:
            # Join papers with feedback, then deduplicate by paper.id
            # paper_feedback.paper_id can be either:
            # 1. Integer ID as string (from harvest saves): "123" -> join on papers.id
            # 2. Semantic Scholar ID (from recommendation saves): "abc123" -> join on papers.semantic_scholar_id

            Logger.info(
                "Executing database query to join papers with feedback", file=LogFiles.HARVEST
            )
            # First, get all matching paper-feedback pairs
            # Join on external IDs (semantic_scholar_id, arxiv_id, openalex_id)
            # This avoids CAST errors on PostgreSQL for non-numeric paper_ids
            # Also check library_paper_id from metadata if available
            base_stmt = (
                select(
                    PaperModel,
                    PaperFeedbackModel.ts,
                    PaperFeedbackModel.track_id,
                    PaperFeedbackModel.action,
                )
                .join(
                    PaperFeedbackModel,
                    or_(
                        PaperModel.semantic_scholar_id == PaperFeedbackModel.paper_id,
                        PaperModel.arxiv_id == PaperFeedbackModel.paper_id,
                        PaperModel.openalex_id == PaperFeedbackModel.paper_id,
                        # For backwards compatibility with numeric IDs stored as strings
                        cast(PaperModel.id, String) == PaperFeedbackModel.paper_id,
                    ),
                )
                .where(
                    PaperFeedbackModel.user_id == user_id,
                    PaperFeedbackModel.action.in_(actions),
                    PaperModel.deleted_at.is_(None),
                )
            )

            if track_id is not None:
                base_stmt = base_stmt.where(PaperFeedbackModel.track_id == track_id)

            # Execute and deduplicate in Python by paper.id (keeping latest feedback)
            all_results = session.execute(base_stmt).all()
            Logger.info(
                f"Query returned {len(all_results)} results before deduplication",
                file=LogFiles.HARVEST,
            )

            # Deduplicate by paper.id, keeping the one with latest timestamp
            Logger.info("Deduplicating results by paper id", file=LogFiles.HARVEST)
            paper_map: Dict[int, Tuple[PaperModel, Optional[datetime], Optional[int], str]] = {}
            for row in all_results:
                paper = row[0]
                ts = row[1]
                fb_track_id = row[2]
                fb_action = row[3]
                current_ts = ts or datetime.min.replace(tzinfo=timezone.utc)
                existing_ts = (
                    paper_map[paper.id][1] or datetime.min.replace(tzinfo=timezone.utc)
                    if paper.id in paper_map
                    else datetime.min.replace(tzinfo=timezone.utc)
                )
                if paper.id not in paper_map or current_ts > existing_ts:
                    paper_map[paper.id] = (paper, ts, fb_track_id, fb_action)

            # Convert to list and sort
            unique_results = list(paper_map.values())
            Logger.info(
                f"After deduplication: {len(unique_results)} unique papers", file=LogFiles.HARVEST
            )

            # Sort
            min_ts = datetime.min.replace(tzinfo=timezone.utc)
            if sort_by == "saved_at":
                unique_results.sort(
                    key=lambda x: x[1] or min_ts, reverse=(sort_order.lower() == "desc")
                )
            elif sort_by == "title":
                unique_results.sort(
                    key=lambda x: x[0].title or "", reverse=(sort_order.lower() == "desc")
                )
            elif sort_by == "citation_count":
                unique_results.sort(
                    key=lambda x: x[0].citation_count or 0, reverse=(sort_order.lower() == "desc")
                )
            elif sort_by == "year":
                unique_results.sort(
                    key=lambda x: x[0].year or 0, reverse=(sort_order.lower() == "desc")
                )
            else:
                unique_results.sort(
                    key=lambda x: x[1] or min_ts, reverse=(sort_order.lower() == "desc")
                )

            # Get total count before pagination
            total = len(unique_results)

            # Apply pagination
            paginated_results = unique_results[offset : offset + limit]

            return [
                LibraryPaper(
                    paper=row[0],
                    saved_at=row[1],
                    track_id=row[2],
                    action=row[3],
                )
                for row in paginated_results
            ], total

    def _get_user_library_canonical(
        self,
        user_id: str,
        *,
        session_provider,
        track_id: Optional[int] = None,
        actions: List[str],
        sort_by: str,
        sort_order: str,
        limit: int,
        offset: int,
    ) -> Tuple[List[LibraryPaper], int]:
        """Single FK JOIN path using paper_feedback.canonical_paper_id."""
        with session_provider.session() as session:
            stmt = (
                select(PaperModel, PaperFeedbackModel)
                .join(
                    PaperFeedbackModel,
                    PaperModel.id == PaperFeedbackModel.canonical_paper_id,
                )
                .where(
                    PaperFeedbackModel.user_id == user_id,
                    PaperFeedbackModel.action.in_(actions),
                    PaperFeedbackModel.canonical_paper_id.is_not(None),
                    PaperModel.deleted_at.is_(None),
                )
            )
            if track_id is not None:
                stmt = stmt.where(PaperFeedbackModel.track_id == track_id)

            all_results = session.execute(stmt).all()

            paper_map: Dict[int, Tuple[PaperModel, PaperFeedbackModel]] = {}
            for row in all_results:
                paper, feedback = row[0], row[1]
                if paper.id not in paper_map or feedback.ts > paper_map[paper.id][1].ts:
                    paper_map[paper.id] = (paper, feedback)

            unique_results = list(paper_map.values())
            min_ts = datetime.min.replace(tzinfo=timezone.utc)
            if sort_by == "saved_at":
                unique_results.sort(
                    key=lambda x: x[1].ts or min_ts, reverse=(sort_order.lower() == "desc")
                )
            elif sort_by == "citation_count":
                unique_results.sort(
                    key=lambda x: x[0].citation_count or 0, reverse=(sort_order.lower() == "desc")
                )
            else:
                unique_results.sort(
                    key=lambda x: x[1].ts or min_ts, reverse=(sort_order.lower() == "desc")
                )

            total = len(unique_results)
            paginated = unique_results[offset : offset + limit]
            return [
                LibraryPaper(
                    paper=row[0], saved_at=row[1].ts, track_id=row[1].track_id, action=row[1].action
                )
                for row in paginated
            ], total

    def remove_from_library(self, user_id: str, paper_id: int) -> bool:
        """Remove paper from user's library by deleting 'save' feedback."""
        with self._provider.session() as session:
            stmt = PaperFeedbackModel.__table__.delete().where(
                PaperFeedbackModel.user_id == user_id,
                PaperFeedbackModel.paper_id == str(paper_id),
                PaperFeedbackModel.action == "save",
            )
            result = session.execute(stmt)
            session.commit()
            return result.rowcount > 0

    def get_latest_judge_scores(self, paper_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Fetch latest judge score per paper id."""
        ids = sorted({int(pid) for pid in paper_ids if int(pid) > 0})
        if not ids:
            return {}

        with self._provider.session() as session:
            rows = (
                session.execute(
                    select(PaperJudgeScoreModel)
                    .where(PaperJudgeScoreModel.paper_id.in_(ids))
                    .order_by(desc(PaperJudgeScoreModel.scored_at), desc(PaperJudgeScoreModel.id))
                )
                .scalars()
                .all()
            )

            latest: Dict[int, Dict[str, Any]] = {}
            for row in rows:
                pid = int(row.paper_id)
                if pid in latest:
                    continue
                latest[pid] = {
                    "overall": float(row.overall or 0.0),
                    "recommendation": str(row.recommendation or ""),
                    "one_line_summary": str(row.one_line_summary or ""),
                    "judge_model": str(row.judge_model or ""),
                    "scored_at": row.scored_at.isoformat() if row.scored_at else None,
                }
            return latest

    def create_harvest_run(
        self,
        run_id: str,
        keywords: List[str],
        venues: List[str],
        sources: List[str],
        max_results_per_source: int,
    ) -> HarvestRunModel:
        """Create a new harvest run record."""
        now = _utcnow()
        with self._provider.session() as session:
            run = HarvestRunModel(
                run_id=run_id,
                keywords_json=json.dumps(keywords, ensure_ascii=False),
                venues_json=json.dumps(venues, ensure_ascii=False),
                sources_json=json.dumps(sources, ensure_ascii=False),
                max_results_per_source=max_results_per_source,
                status="running",
                started_at=now,
            )
            session.add(run)
            session.commit()
            session.refresh(run)
            return run

    def update_harvest_run(
        self,
        run_id: str,
        *,
        status: Optional[str] = None,
        papers_found: Optional[int] = None,
        papers_new: Optional[int] = None,
        papers_deduplicated: Optional[int] = None,
        errors: Optional[Dict[str, Any]] = None,
    ) -> Optional[HarvestRunModel]:
        """Update a harvest run record."""
        now = _utcnow()
        with self._provider.session() as session:
            run = session.execute(
                select(HarvestRunModel).where(HarvestRunModel.run_id == run_id)
            ).scalar_one_or_none()

            if run is None:
                return None

            if status is not None:
                run.status = status
                if status in ("success", "partial", "failed"):
                    run.ended_at = now

            if papers_found is not None:
                run.papers_found = papers_found
            if papers_new is not None:
                run.papers_new = papers_new
            if papers_deduplicated is not None:
                run.papers_deduplicated = papers_deduplicated
            if errors is not None:
                run.set_errors(errors)

            session.commit()
            session.refresh(run)
            return run

    def get_harvest_run(self, run_id: str) -> Optional[HarvestRunModel]:
        """Get a harvest run by its ID."""
        with self._provider.session() as session:
            return session.execute(
                select(HarvestRunModel).where(HarvestRunModel.run_id == run_id)
            ).scalar_one_or_none()

    def list_harvest_runs(
        self,
        *,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[HarvestRunModel]:
        """List harvest runs with optional filtering."""
        with self._provider.session() as session:
            stmt = select(HarvestRunModel)

            if status:
                stmt = stmt.where(HarvestRunModel.status == status)

            stmt = stmt.order_by(HarvestRunModel.started_at.desc())
            stmt = stmt.offset(offset).limit(limit)

            return list(session.execute(stmt).scalars().all())

    def get_paper_count(self) -> int:
        """Get total count of papers in the store."""
        with self._provider.session() as session:
            return (
                session.execute(
                    select(func.count())
                    .select_from(PaperModel)
                    .where(PaperModel.deleted_at.is_(None))
                ).scalar()
                or 0
            )

    def close(self) -> None:
        """Close database connections."""
        try:
            self._provider.engine.dispose()
        except Exception:
            pass

    def get_paper_by_source_id_any(self, source_id: str) -> Optional[PaperModel]:
        """Look up a paper by any source ID (semantic_scholar, arxiv, openalex, or internal id)."""
        with self._provider.session() as session:
            # Try internal numeric ID first
            try:
                int_id = int(source_id)
                result = session.execute(
                    select(PaperModel).where(PaperModel.id == int_id, PaperModel.deleted_at.is_(None))
                ).scalar_one_or_none()
                if result:
                    return result
            except (ValueError, TypeError):
                pass

            # Try all source ID columns
            result = session.execute(
                select(PaperModel).where(
                    PaperModel.deleted_at.is_(None),
                    or_(
                        PaperModel.semantic_scholar_id == source_id,
                        PaperModel.arxiv_id == source_id,
                        PaperModel.openalex_id == source_id,
                    ),
                )
            ).scalar_one_or_none()
            return result

    def update_structured_card(self, paper_id: int, card_json: str) -> None:
        """Cache structured card JSON on a paper record."""
        with self._provider.session() as session:
            paper = session.get(PaperModel, paper_id)
            if paper:
                paper.structured_card_json = card_json
                paper.updated_at = _utcnow()
                session.commit()


# Backward-compatible alias: older workflows/tests import SqlAlchemyPaperStore.
SqlAlchemyPaperStore = PaperStore


def paper_to_dict(paper: PaperModel) -> Dict[str, Any]:
    """Convert PaperModel to dictionary for API response."""
    return {
        "id": paper.id,
        "doi": paper.doi,
        "arxiv_id": paper.arxiv_id,
        "semantic_scholar_id": paper.semantic_scholar_id,
        "openalex_id": paper.openalex_id,
        "title": paper.title,
        "abstract": paper.abstract,
        "authors": paper.get_authors(),
        "year": paper.year,
        "venue": paper.venue,
        "publication_date": paper.publication_date,
        "citation_count": paper.citation_count,
        "url": paper.url,
        "pdf_url": paper.pdf_url,
        "keywords": paper.get_keywords(),
        "fields_of_study": paper.get_fields_of_study(),
        "primary_source": paper.primary_source,
        "sources": paper.get_sources(),
        "created_at": paper.created_at.isoformat() if paper.created_at else None,
        "updated_at": paper.updated_at.isoformat() if paper.updated_at else None,
    }
