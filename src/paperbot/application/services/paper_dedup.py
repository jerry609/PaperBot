"""Three-tier paper deduplication: DOI -> arxiv_id -> rapidfuzz title."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from rapidfuzz import fuzz

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
            match = self._find_title_match(normalized)
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
        if normalized:
            self._title_entries.append((normalized, paper))
        return True

    def canonical_for(self, paper: PaperCandidate) -> Optional[PaperCandidate]:
        """Return the surviving canonical paper that *paper* was merged into (or itself)."""
        return self._canonical.get(id(paper))

    def results(self) -> List[PaperCandidate]:
        """Return deduplicated papers in insertion order."""
        seen_ids: set = set()
        out: List[PaperCandidate] = []
        for _, paper in self._title_entries:
            obj_id = id(paper)
            if obj_id not in seen_ids:
                seen_ids.add(obj_id)
                out.append(paper)
        # Also include papers that had no normalizable title but were kept
        for paper in self._doi_index.values():
            obj_id = id(paper)
            if obj_id not in seen_ids:
                seen_ids.add(obj_id)
                out.append(paper)
        for paper in self._arxiv_index.values():
            obj_id = id(paper)
            if obj_id not in seen_ids:
                seen_ids.add(obj_id)
                out.append(paper)
        return out

    def _find_title_match(self, normalized: str) -> Optional[PaperCandidate]:
        """Linear scan for a fuzzy title match above threshold."""
        threshold = self._title_threshold * 100  # rapidfuzz uses 0-100 scale
        for existing_title, existing_paper in self._title_entries:
            score = fuzz.ratio(normalized, existing_title)
            if score >= threshold:
                return existing_paper
        return None

    @staticmethod
    def _merge_into(winner: PaperCandidate, donor: PaperCandidate) -> None:
        """Merge metadata from *donor* into *winner*, keeping the better value."""
        # Keep higher citation count
        if (donor.citation_count or 0) > (winner.citation_count or 0):
            winner.citation_count = donor.citation_count

        # Keep longer abstract
        if len(donor.abstract or "") > len(winner.abstract or ""):
            winner.abstract = donor.abstract

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

        # Merge retrieval sources
        existing_sources = set(winner.retrieval_sources or [])
        for src in donor.retrieval_sources or []:
            if src not in existing_sources:
                winner.retrieval_sources.append(src)
                existing_sources.add(src)

    @staticmethod
    def _extract_doi(paper: PaperCandidate) -> str:
        """Extract and normalize a DOI from the paper's identities."""
        for ident in paper.identities:
            if ident.source == "doi" and ident.external_id:
                return ident.external_id.strip().lower()
        return ""

    @staticmethod
    def _extract_arxiv_id(paper: PaperCandidate) -> str:
        """Extract and normalize an arxiv_id, stripping the version suffix."""
        for ident in paper.identities:
            if ident.source == "arxiv" and ident.external_id:
                raw = ident.external_id.strip().lower().removeprefix("arxiv:")
                # Strip version suffix (e.g. 2301.12345v2 -> 2301.12345)
                if "v" in raw:
                    head, tail = raw.rsplit("v", 1)
                    if head and tail.isdigit():
                        return head
                return raw
        return ""
