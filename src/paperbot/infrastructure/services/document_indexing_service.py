from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Sequence

from paperbot.application.ports.document_intelligence_port import (
    DocumentChunk,
    DocumentContent,
    DocumentFetcherPort,
    DocumentIndexerPort,
    DocumentSection,
)
from paperbot.context_engine.embeddings import (
    EmbeddingProvider,
    try_build_default_embedding_provider,
)
from paperbot.infrastructure.stores.document_index_store import DocumentIndexStore
from paperbot.infrastructure.stores.paper_store import PaperStore

_WORD_RX = re.compile(r"\S+")
_SENTENCE_BOUNDARY_RX = re.compile(r"(?<=[.!?])\s+")
_STRUCTURED_CARD_KEY_ORDER = (
    "summary",
    "problem",
    "motivation",
    "method",
    "approach",
    "finding",
    "findings",
    "results",
    "limitations",
    "datasets",
    "evaluation",
    "applications",
)


def _estimate_tokens(text_value: str) -> int:
    return len(_WORD_RX.findall(text_value or ""))


def _normalize_label(value: str) -> str:
    text_value = str(value or "").strip().replace("_", " ")
    if not text_value:
        return ""
    return text_value[:1].upper() + text_value[1:]


def _stringify_card_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [_stringify_card_value(item) for item in value]
        parts = [part for part in parts if part]
        if not parts:
            return ""
        return "\n".join(f"- {part}" for part in parts)
    if isinstance(value, dict):
        parts: List[str] = []
        for key, nested_value in value.items():
            rendered = _stringify_card_value(nested_value)
            if not rendered:
                continue
            parts.append(f"{_normalize_label(str(key))}: {rendered}")
        return "\n".join(parts)
    return ""


def _iter_structured_card_sections(card: Dict[str, Any]) -> List[DocumentSection]:
    sections: List[DocumentSection] = []
    seen_keys: set[str] = set()
    ordered_keys = [key for key in _STRUCTURED_CARD_KEY_ORDER if key in card]
    ordered_keys.extend(key for key in card.keys() if key not in seen_keys)

    for index, key in enumerate(ordered_keys, start=1):
        if key in seen_keys:
            continue
        seen_keys.add(key)
        content = _stringify_card_value(card.get(key))
        if not content:
            continue
        sections.append(
            DocumentSection(
                name=str(key),
                heading=_normalize_label(str(key)),
                content=content,
                order=index,
                metadata={"origin": "structured_card"},
            )
        )
    return sections


def _chunk_text(text_value: str, *, max_chars: int = 900, overlap_chars: int = 120) -> List[str]:
    content = " ".join((text_value or "").split())
    if not content:
        return []
    if len(content) <= max_chars:
        return [content]

    sentences = [part.strip() for part in _SENTENCE_BOUNDARY_RX.split(content) if part.strip()]
    if not sentences:
        return [content[:max_chars]]

    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = f"{current[-overlap_chars:].strip()} {sentence}".strip()
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


class MetadataDocumentFetcher(DocumentFetcherPort):
    """Build an indexable document from canonical paper metadata."""

    def __init__(self, *, paper_store: Optional[PaperStore] = None) -> None:
        self.paper_store = paper_store or PaperStore()

    def fetch_document(self, *, paper_id: int) -> Optional[DocumentContent]:
        paper = self.paper_store.get_paper_by_id(int(paper_id))
        if paper is None:
            return None

        overview_lines: List[str] = [f"Title: {paper.title}"]
        authors = [str(author).strip() for author in paper.get_authors() if str(author).strip()]
        if authors:
            overview_lines.append(f"Authors: {', '.join(authors)}")
        if paper.year:
            overview_lines.append(f"Year: {paper.year}")
        if paper.venue:
            overview_lines.append(f"Venue: {paper.venue}")
        keywords = [
            str(keyword).strip() for keyword in paper.get_keywords() if str(keyword).strip()
        ]
        if keywords:
            overview_lines.append(f"Keywords: {', '.join(keywords)}")
        fields_of_study = [
            str(field_value).strip()
            for field_value in paper.get_fields_of_study()
            if str(field_value).strip()
        ]
        if fields_of_study:
            overview_lines.append(f"Fields of study: {', '.join(fields_of_study)}")
        if paper.url:
            overview_lines.append(f"Paper URL: {paper.url}")
        if paper.pdf_url:
            overview_lines.append(f"PDF URL: {paper.pdf_url}")

        sections: List[DocumentSection] = [
            DocumentSection(
                name="overview",
                heading="Paper Overview",
                content="\n".join(overview_lines),
                order=0,
                metadata={"origin": "paper_metadata"},
            )
        ]

        abstract = str(paper.abstract or "").strip()
        if abstract:
            sections.append(
                DocumentSection(
                    name="abstract",
                    heading="Abstract",
                    content=abstract,
                    order=1,
                    metadata={"origin": "paper_metadata"},
                )
            )

        structured_card = paper.get_structured_card() or {}
        if isinstance(structured_card, dict) and structured_card:
            sections.extend(_iter_structured_card_sections(structured_card))

        if not sections:
            return None

        checksum_payload = {
            "title": paper.title,
            "abstract": abstract,
            "structured_card": structured_card,
            "keywords": keywords,
            "fields_of_study": fields_of_study,
        }
        checksum = hashlib.sha256(
            json.dumps(checksum_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

        locator_url = str(paper.pdf_url or paper.url or "").strip() or None
        return DocumentContent(
            paper_id=int(paper.id),
            title=str(paper.title or ""),
            source_type="paper_metadata",
            sections=sections,
            locator_url=locator_url,
            checksum=checksum,
            metadata={
                "primary_source": paper.primary_source,
                "sources": paper.get_sources(),
                "year": paper.year,
                "venue": paper.venue,
            },
        )


class MetadataDocumentIndexer(DocumentIndexerPort):
    """Chunk metadata-derived sections into retrieval units."""

    def __init__(self, *, embedding_provider: Optional[EmbeddingProvider] = None) -> None:
        self.embedding_provider = embedding_provider

    def index_document(self, *, document: DocumentContent) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        next_index = 0
        for section in sorted(document.sections, key=lambda section: int(section.order)):
            section_chunks = _chunk_text(section.content)
            for local_index, section_chunk in enumerate(section_chunks):
                embedding = None
                if self.embedding_provider is not None:
                    try:
                        embedding = self.embedding_provider.embed(section_chunk)
                    except Exception:
                        embedding = None
                chunks.append(
                    DocumentChunk(
                        paper_id=int(document.paper_id),
                        section=str(section.name or ""),
                        heading=str(section.heading or ""),
                        content=section_chunk,
                        chunk_index=next_index,
                        token_count=_estimate_tokens(section_chunk),
                        embedding=embedding,
                        metadata={
                            **dict(section.metadata or {}),
                            "section_order": int(section.order),
                            "section_chunk_index": int(local_index),
                        },
                    )
                )
                next_index += 1
        return chunks


class DocumentIndexingService:
    """Orchestrates explicit ingest-triggered document indexing."""

    def __init__(
        self,
        *,
        paper_store: Optional[PaperStore] = None,
        index_store: Optional[DocumentIndexStore] = None,
        fetcher: Optional[DocumentFetcherPort] = None,
        indexer: Optional[DocumentIndexerPort] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ) -> None:
        self.paper_store = paper_store or PaperStore()
        self.embedding_provider = embedding_provider or try_build_default_embedding_provider()
        self.index_store = index_store or DocumentIndexStore(
            db_url=self.paper_store.db_url,
            embedding_provider=self.embedding_provider,
        )
        self.fetcher = fetcher or MetadataDocumentFetcher(paper_store=self.paper_store)
        self.indexer = indexer or MetadataDocumentIndexer(
            embedding_provider=self.embedding_provider,
        )

    def close(self) -> None:
        close_index_store = getattr(self.index_store, "close", None)
        if callable(close_index_store):
            close_index_store()

        close_paper_store = getattr(self.paper_store, "close", None)
        if callable(close_paper_store):
            close_paper_store()

    def enqueue_papers(self, *, paper_ids: Sequence[int], trigger_source: str) -> Dict[str, int]:
        return self.index_store.enqueue_jobs(
            paper_ids=paper_ids,
            trigger_source=trigger_source,
        )

    def process_pending_jobs(self, *, limit: int = 10) -> Dict[str, int]:
        claimed_jobs = self.index_store.claim_pending_jobs(limit=limit)
        summary = {
            "claimed": len(claimed_jobs),
            "completed": 0,
            "failed": 0,
            "chunks_indexed": 0,
        }
        for job in claimed_jobs:
            job_id = int(job.get("id") or 0)
            paper_id = int(job.get("paper_id") or 0)
            try:
                document = self.fetcher.fetch_document(paper_id=paper_id)
                if document is None:
                    raise RuntimeError(f"paper {paper_id} not found for indexing")

                asset = self.index_store.upsert_asset(
                    paper_id=paper_id,
                    source_type=document.source_type,
                    title=document.title,
                    locator_url=document.locator_url,
                    checksum=document.checksum,
                    metadata=document.metadata,
                )
                chunks = self.indexer.index_document(document=document)
                chunk_count = self.index_store.replace_chunks(
                    asset_id=int(asset["id"]),
                    paper_id=paper_id,
                    chunks=chunks,
                )
                self.index_store.complete_job(
                    job_id=job_id,
                    asset_id=int(asset["id"]),
                    chunk_count=chunk_count,
                )
                summary["completed"] += 1
                summary["chunks_indexed"] += chunk_count
            except Exception as exc:
                self.index_store.fail_job(job_id=job_id, error=str(exc))
                summary["failed"] += 1
        return summary
