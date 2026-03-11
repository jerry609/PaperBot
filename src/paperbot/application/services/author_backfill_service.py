from __future__ import annotations

import re
from typing import Any, Optional

from sqlalchemy import func, select

from paperbot.infrastructure.stores.author_store import AuthorStore
from paperbot.infrastructure.stores.models import AuthorModel, PaperAuthorModel, PaperModel
from paperbot.infrastructure.stores.sqlalchemy_db import SessionProvider, get_db_url


def normalize_author_name(name: Any) -> str:
    text = str(name or "").replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip(" ,;\t\n\r")
    return text.strip()


def _normalize_author_key(name: str) -> str:
    return normalize_author_name(name).casefold()


def _clean_author_list(raw_authors: list[Any]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()

    for raw in raw_authors or []:
        name = normalize_author_name(raw)
        if not name:
            continue
        key = _normalize_author_key(name)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(name)

    return cleaned


def run_author_backfill(
    *,
    db_url: Optional[str] = None,
    limit: Optional[int] = None,
    paper_id: Optional[int] = None,
    provider: Optional[SessionProvider] = None,
) -> dict[str, int]:
    url = db_url or get_db_url()
    provider = provider or SessionProvider(url)
    author_store = AuthorStore(db_url=url)

    with provider.session() as session:
        initial_author_count = int(
            session.execute(select(func.count(AuthorModel.id))).scalar() or 0
        )
        initial_relation_count = int(
            session.execute(select(func.count(PaperAuthorModel.id))).scalar() or 0
        )

        query = select(PaperModel).order_by(PaperModel.id.asc())
        if paper_id is not None:
            query = query.where(PaperModel.id == int(paper_id))
        if limit is not None and int(limit) > 0:
            query = query.limit(int(limit))
        papers = session.execute(query).scalars().all()

    stats = {
        "scanned_papers": 0,
        "processed_papers": 0,
        "skipped_no_authors": 0,
        "skipped_unchanged": 0,
        "new_authors": 0,
        "new_relations": 0,
    }

    # TODO: N+1 query — batch-fetch existing paper_authors and process
    #  in bulk to reduce DB roundtrips for large backfills (PR #112 review).
    for paper in papers:
        stats["scanned_papers"] += 1

        raw_authors = paper.get_authors()
        cleaned_authors = _clean_author_list(raw_authors)
        if not cleaned_authors:
            stats["skipped_no_authors"] += 1
            continue

        existing_rows = author_store.get_paper_authors(paper_id=int(paper.id))
        existing_names = [normalize_author_name(row.get("name")) for row in existing_rows]
        if existing_names == cleaned_authors:
            stats["skipped_unchanged"] += 1
            continue

        author_store.replace_paper_authors(paper_id=int(paper.id), authors=cleaned_authors)
        stats["processed_papers"] += 1

    with provider.session() as session:
        final_author_count = int(session.execute(select(func.count(AuthorModel.id))).scalar() or 0)
        final_relation_count = int(
            session.execute(select(func.count(PaperAuthorModel.id))).scalar() or 0
        )

    stats["new_authors"] = max(final_author_count - initial_author_count, 0)
    stats["new_relations"] = max(final_relation_count - initial_relation_count, 0)
    return stats
