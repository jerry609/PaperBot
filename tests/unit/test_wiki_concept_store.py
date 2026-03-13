from __future__ import annotations

from pathlib import Path

from paperbot.infrastructure.stores.paper_store import PaperStore
from paperbot.infrastructure.stores.research_store import SqlAlchemyResearchStore
from paperbot.infrastructure.stores.wiki_concept_store import WikiConceptStore


def test_wiki_concept_store_loads_papers_and_tracks(tmp_path: Path):
    db_url = f"sqlite:///{tmp_path / 'wiki-grounding.db'}"
    paper_store = PaperStore(db_url=db_url)
    research_store = SqlAlchemyResearchStore(db_url=db_url)

    saved_paper = paper_store.upsert_paper(
        paper={
            "title": "Attention Is All You Need",
            "abstract": "Transformer models use self-attention for sequence modeling.",
            "keywords": ["transformer", "attention"],
            "fields_of_study": ["Natural Language Processing", "Architecture"],
            "citation_count": 1000,
            "year": 2017,
        }
    )
    paper_store.upsert_paper(
        paper={
            "title": "Global Unsaved Paper",
            "abstract": "This row should not appear in user-scoped wiki grounding.",
            "keywords": ["diffusion"],
            "fields_of_study": ["Method"],
            "citation_count": 5,
            "year": 2026,
        }
    )
    track = research_store.create_track(
        user_id="default",
        name="LLM Agents",
        description="Track the architecture and alignment stack for agents.",
        keywords=["transformer", "agents"],
        methods=["rlhf"],
        activate=True,
    )
    research_store.add_paper_feedback(
        user_id="default",
        track_id=int(track["id"]),
        paper_id=str(saved_paper["id"]),
        action="save",
    )

    store = WikiConceptStore(db_url=db_url)
    snapshot = store.load_grounding_snapshot(user_id="default")

    assert len(snapshot["papers"]) == 1
    assert snapshot["papers"][0]["title"] == "Attention Is All You Need"
    assert snapshot["papers"][0]["keywords"] == ["transformer", "attention"]
    assert len(snapshot["tracks"]) == 1
    assert snapshot["tracks"][0]["name"] == "LLM Agents"
    assert snapshot["tracks"][0]["methods"] == ["rlhf"]
