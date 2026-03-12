from __future__ import annotations

import json
from pathlib import Path

from paperbot.context_engine.embeddings import HashEmbeddingProvider
from paperbot.infrastructure.services.document_indexing_service import DocumentIndexingService
from paperbot.infrastructure.stores.document_index_store import DocumentIndexStore
from paperbot.infrastructure.stores.paper_store import PaperStore


def test_document_indexing_service_indexes_metadata_and_retrieves_evidence(tmp_path: Path):
    db_url = f"sqlite:///{tmp_path / 'document-index.db'}"
    embedding_provider = HashEmbeddingProvider(dim=128)
    paper_store = PaperStore(db_url=db_url, auto_create_schema=True)

    upserted = paper_store.upsert_paper(
        paper={
            "title": "Sparse Retrieval for Transformer Agents",
            "abstract": (
                "We study sparse retrieval for transformer agents and show that "
                "retrieval-aware memory routing improves efficiency."
            ),
            "authors": ["Ada Lovelace", "Grace Hopper"],
            "year": 2026,
            "venue": "arXiv",
            "keywords": ["transformer", "retrieval", "agents"],
            "fields_of_study": ["Machine Learning"],
            "url": "https://example.com/paper",
            "pdf_url": "https://example.com/paper.pdf",
        },
        source_hint="papers_cool",
    )
    paper_id = int(upserted["id"])
    paper_store.update_structured_card(
        paper_id,
        json.dumps(
            {
                "method": "Sparse retrieval memory routing for transformer agents.",
                "findings": [
                    "Improves retrieval latency.",
                    "Maintains answer quality on research tasks.",
                ],
                "limitations": "Still depends on metadata coverage in v1 indexing.",
            },
            ensure_ascii=False,
        ),
    )

    index_store = DocumentIndexStore(
        db_url=db_url,
        auto_create_schema=True,
        embedding_provider=embedding_provider,
    )
    service = DocumentIndexingService(
        paper_store=paper_store,
        index_store=index_store,
        embedding_provider=embedding_provider,
    )

    enqueue_summary = service.enqueue_papers(
        paper_ids=[paper_id],
        trigger_source="unit_test",
    )
    assert enqueue_summary["queued"] == 1

    processed = service.process_pending_jobs(limit=5)
    assert processed["completed"] == 1
    assert processed["failed"] == 0
    assert processed["chunks_indexed"] >= 2

    hits = index_store.retrieve_evidence(
        query="sparse retrieval transformer memory routing",
        paper_ids=[paper_id],
        limit=3,
    )

    assert hits
    assert hits[0].paper_id == paper_id
    assert "retrieval" in hits[0].snippet.lower()
    assert hits[0].source_type == "paper_metadata"

    service.close()
