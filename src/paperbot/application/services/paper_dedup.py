"""Three-tier paper deduplication: DOI -> arxiv_id -> rapidfuzz title."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover - fallback when optional dependency is unavailable.
    class _FuzzFallback:
        @staticmethod
        def ratio(left: str, right: str) -> float:
            return SequenceMatcher(None, left or "", right or "").ratio() * 100.0

    fuzz = _FuzzFallback()

from paperbot.domain.paper import PaperCandidate

_PUNCT_RX = re.compile(r"[^\w\s]")
_WHITESPACE_RX = re.compile(r"\s+")


def normalize_title(title: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = (title or "").lower()
    text = _PUNCT_RX.sub("", text)
    text = _WHITESPACE_RX.sub(" ", text).strip()
    return text


class PaperDeduplicator:
    """Three-tier paper deduplication: DOI -> arxiv_id -> rapidfuzz title.

    Tier 1: exact DOI match.
    Tier 2: exact arxiv_id match (version-stripped).
    Tier 3: fuzzy title similarity >= threshold via rapidfuzz.
    """

    def __init__(self, title_threshold: float = 0.85) -> None:
        self._title_threshold = title_threshold
        self._doi_index: Dict[str, PaperCandidate] = {}
        self._arxiv_index: Dict[str, PaperCandidate] = {}
        self._title_entries: List[Tuple[str, PaperCandidate]] = []
        self._canonical: Dict[int, PaperCandidate] = {}  # id(original) -> surviving paper
        self._ordered_results: List[PaperCandidate] = []

    def add(self, paper: PaperCandidate) -> bool:
        """Add a paper. Returns True if new (kept), False if duplicate (merged)."""
        # Tier 1: DOI exact match
        doi = self._extract_doi(paper)
        if doi:
            existing = self._doi_index.get(doi)
            if existing is not None:
                self._merge_into(existing, paper)
                self._canonical[id(paper)] = existing
                return False
            self._doi_index[doi] = paper

        # Tier 2: arxiv_id exact match (version-stripped)
        arxiv_id = self._extract_arxiv_id(paper)
        if arxiv_id:
            existing = self._arxiv_index.get(arxiv_id)
            if existing is not None:
                self._merge_into(existing, paper)
                self._canonical[id(paper)] = existing
                # Ensure DOI index also points to the winner
                if doi:
                    self._doi_index[doi] = existing
                return False
            self._arxiv_index[arxiv_id] = paper

        # Tier 3: rapidfuzz title similarity
        normalized = normalize_title(paper.title)
        if normalized:
            match = self._find_title_match(
                normalized,
                doi_values=self._extract_doi_values(paper),
                arxiv_values=self._extract_arxiv_values(paper),
            )
            if match is not None:
                self._merge_into(match, paper)
                self._canonical[id(paper)] = match
                # Keep DOI/arxiv indexes pointing to the winner
                if doi:
                    self._doi_index[doi] = match
                if arxiv_id:
                    self._arxiv_index[arxiv_id] = match
                return False

        # No duplicate found — register as new
        self._canonical[id(paper)] = paper
        self._ordered_results.append(paper)
        if normalized:
            self._title_entries.append((normalized, paper))
        return True

    def canonical_for(self, paper: PaperCandidate) -> Optional[PaperCandidate]:
        """Return the surviving canonical paper that *paper* was merged into (or itself)."""
        return self._canonical.get(id(paper))

    def results(self) -> List[PaperCandidate]:
        """Return deduplicated papers in insertion order."""
        return list(self._ordered_results)

    def _find_title_match(
        self,
        normalized: str,
        *,
        doi_values: set[str],
        arxiv_values: set[str],
    ) -> Optional[PaperCandidate]:
        """Linear scan for a fuzzy title match above threshold."""
        threshold = self._title_threshold * 100  # rapidfuzz uses 0-100 scale
        for existing_title, existing_paper in self._title_entries:
            if self._has_conflicting_identity(
                existing_paper,
                doi_values=doi_values,
                arxiv_values=arxiv_values,
            ):
                continue
            score = fuzz.ratio(normalized, existing_title)
            if score >= threshold:
                return existing_paper
        return None

    @classmethod
    def _has_conflicting_identity(
        cls,
        paper: PaperCandidate,
        *,
        doi_values: set[str],
        arxiv_values: set[str],
    ) -> bool:
        existing_dois = cls._extract_doi_values(paper)
        if doi_values and existing_dois and doi_values.isdisjoint(existing_dois):
            return True

        existing_arxiv_ids = cls._extract_arxiv_values(paper)
        if arxiv_values and existing_arxiv_ids and arxiv_values.isdisjoint(existing_arxiv_ids):
            return True

        return False

    @staticmethod
    def _merge_into(winner: PaperCandidate, donor: PaperCandidate) -> None:
        """Merge metadata from *donor* into *winner*, keeping the better value."""
        # Keep higher citation count
        if (donor.citation_count or 0) > (winner.citation_count or 0):
            winner.citation_count = donor.citation_count

        # Keep longer abstract
        if len(donor.abstract or "") > len(winner.abstract or ""):
            winner.abstract = donor.abstract

        if len(donor.authors or []) > len(winner.authors or []):
            winner.authors = list(donor.authors or [])

        if not winner.url and donor.url:
            winner.url = donor.url

        if not winner.pdf_url and donor.pdf_url:
            winner.pdf_url = donor.pdf_url

        # Merge identities (avoid duplicates)
        existing_pairs = {(i.source, i.external_id) for i in winner.identities}
        for ident in donor.identities:
            if (ident.source, ident.external_id) not in existing_pairs:
                winner.identities.append(ident)
                existing_pairs.add((ident.source, ident.external_id))

        # Keep year if winner is missing it
        if not winner.year and donor.year:
            winner.year = donor.year

        # Keep venue if winner is missing it
        if not winner.venue and donor.venue:
            winner.venue = donor.venue

        if not winner.publication_date and donor.publication_date:
            winner.publication_date = donor.publication_date

        winner.keywords = PaperDeduplicator._merge_text_list(winner.keywords, donor.keywords)
        winner.fields_of_study = PaperDeduplicator._merge_text_list(
            winner.fields_of_study,
            donor.fields_of_study,
        )

        # Merge retrieval sources
        winner.retrieval_sources = PaperDeduplicator._merge_text_list(
            winner.retrieval_sources,
            donor.retrieval_sources,
        )

    @staticmethod
    def _merge_text_list(current: List[str], incoming: List[str]) -> List[str]:
        merged: List[str] = []
        seen: set[str] = set()
        for value in list(current or []) + list(incoming or []):
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            merged.append(text)
        return merged

    @staticmethod
    def _extract_doi(paper: PaperCandidate) -> str:
        """Extract and normalize a DOI from the paper's identities."""
        values = sorted(PaperDeduplicator._extract_doi_values(paper))
        return values[0] if values else ""

    @staticmethod
    def _extract_arxiv_id(paper: PaperCandidate) -> str:
        """Extract and normalize an arxiv_id, stripping the version suffix."""
        values = sorted(PaperDeduplicator._extract_arxiv_values(paper))
        return values[0] if values else ""

    @staticmethod
    def _extract_doi_values(paper: PaperCandidate) -> set[str]:
        values: set[str] = set()
        for ident in paper.identities:
            if ident.source == "doi" and ident.external_id:
                values.add(ident.external_id.strip().lower())
        return values

    @staticmethod
    def _extract_arxiv_values(paper: PaperCandidate) -> set[str]:
        values: set[str] = set()
        for ident in paper.identities:
            if ident.source != "arxiv" or not ident.external_id:
                continue
            raw = ident.external_id.strip().lower().removeprefix("arxiv:")
            # Strip version suffix (e.g. 2301.12345v2 -> 2301.12345)
            if "v" in raw:
                head, tail = raw.rsplit("v", 1)
                if head and tail.isdigit():
                    values.add(head)
                    continue
            values.add(raw)
        return values
