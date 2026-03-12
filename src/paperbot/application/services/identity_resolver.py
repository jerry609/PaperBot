"""IdentityResolver — external ID → canonical papers.id resolution.

Replaces the multi-path waterfall in research_store._resolve_paper_ref_id
with a clean two-step lookup:
  1. paper_identifiers table (O(1) exact match)
  2. Fallback: normalize + query papers table columns
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from sqlalchemy import func, or_, select

from paperbot.domain.paper_identity import normalize_arxiv_id, normalize_doi
from paperbot.infrastructure.stores.identity_store import IdentityStore
from paperbot.infrastructure.stores.models import PaperModel
from paperbot.infrastructure.stores.sqlalchemy_db import SessionProvider, get_db_url


class IdentityResolver:
    """Resolve an external paper ID to the canonical papers.id."""

    def __init__(
        self,
        identity_store: Optional[IdentityStore] = None,
        db_url: Optional[str] = None,
        *,
        provider: Optional[SessionProvider] = None,
    ):
        self._identity_store = identity_store or IdentityStore(db_url=db_url)
        self._db_url = db_url or get_db_url()
        self._provider = provider or SessionProvider(self._db_url)

    def resolve(
        self,
        external_id: str,
        hints: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        """Resolve *external_id* to papers.id.

        Steps:
          1. paper_identifiers exact match (any source)
          2. Normalize as arxiv_id → papers.arxiv_id
          3. Normalize as doi → papers.doi
          4. URL match from hints
          5. Title fuzzy match from hints
        """
        pid = (external_id or "").strip()
        hints = hints or {}

        # 0a. library_paper_id hint → already-resolved internal ID, use directly
        lib_id = hints.get("library_paper_id")
        if lib_id is not None:
            try:
                return int(lib_id)
            except (TypeError, ValueError):
                pass

        # 0b. Numeric ID → direct lookup
        if pid.isdigit():
            with self._provider.session() as session:
                row = session.execute(
                    select(PaperModel).where(PaperModel.id == int(pid))
                ).scalar_one_or_none()
                if row:
                    return int(row.id)

        # 1. paper_identifiers table
        resolved = self._identity_store.resolve_any(pid)
        if resolved is not None:
            return resolved

        # 2. Normalize as arxiv / doi and try papers table columns
        arxiv_id = normalize_arxiv_id(pid) if pid else None
        doi = normalize_doi(pid) if pid else None

        url_candidates = []
        for key in ("paper_url", "url", "external_url", "pdf_url"):
            value = hints.get(key)
            if isinstance(value, str) and value.strip():
                url_candidates.append(value.strip())
        if pid.startswith("http"):
            url_candidates.append(pid)

        if not arxiv_id:
            for candidate in url_candidates:
                arxiv_id = normalize_arxiv_id(candidate)
                if arxiv_id:
                    break
        if not doi:
            for candidate in url_candidates:
                doi = normalize_doi(candidate)
                if doi:
                    break

        with self._provider.session() as session:
            if arxiv_id:
                row = session.execute(
                    select(PaperModel).where(PaperModel.arxiv_id == arxiv_id)
                ).scalar_one_or_none()
                if row:
                    return int(row.id)

            if doi:
                row = session.execute(
                    select(PaperModel).where(PaperModel.doi == doi)
                ).scalar_one_or_none()
                if row:
                    return int(row.id)

            # 3. URL match
            if url_candidates:
                row = session.execute(
                    select(PaperModel).where(
                        or_(
                            PaperModel.url.in_(url_candidates),
                            PaperModel.pdf_url.in_(url_candidates),
                        )
                    )
                ).first()
                if row:
                    return int(row[0].id)

            # 4. Title match
            title = str(hints.get("title") or "").strip()
            if title:
                row = session.execute(
                    select(PaperModel).where(
                        func.lower(PaperModel.title) == title.lower()
                    )
                ).scalar_one_or_none()
                if row:
                    return int(row.id)

        return None
