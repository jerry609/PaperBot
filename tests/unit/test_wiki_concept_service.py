from __future__ import annotations

from paperbot.application.ports.wiki_concept_port import GroundingSnapshot
from paperbot.application.services.wiki_concept_service import WikiConceptService


class _FakeWikiConceptStore:
    def load_grounding_snapshot(
        self,
        *,
        user_id: str,
        paper_limit: int = 250,
        track_limit: int = 100,
    ) -> GroundingSnapshot:
        return {
            "papers": [
                {
                    "title": "Attention Is All You Need",
                    "abstract": "The transformer architecture relies on self-attention.",
                    "keywords": ["transformer", "attention"],
                    "fields_of_study": ["Architecture"],
                    "citation_count": 1000,
                    "year": 2017,
                },
                {
                    "title": "Learning to summarize with BLEU-aware rewards",
                    "abstract": "We optimize a model with BLEU score as a target metric.",
                    "keywords": ["bleu", "summarization"],
                    "fields_of_study": ["Metric"],
                    "citation_count": 20,
                    "year": 2024,
                },
            ],
            "tracks": [
                {
                    "name": "Alignment Stack",
                    "description": "Preference optimization and RLHF",
                    "keywords": ["alignment"],
                    "methods": ["rlhf"],
                }
            ],
        }


def test_wiki_concept_service_enriches_catalog_with_live_grounding():
    service = WikiConceptService(_FakeWikiConceptStore())

    items = service.list_concepts(user_id="default", query="transformer")

    assert items
    top = items[0]
    assert top.id == "transformer"
    assert top.paper_count == 1
    assert top.related_papers == ["Attention Is All You Need"]


def test_wiki_concept_service_filters_by_category():
    service = WikiConceptService(_FakeWikiConceptStore())

    items = service.list_concepts(user_id="default", category="Metric")

    assert items
    assert all(item.category == "Metric" for item in items)
    assert any(item.id == "bleu" for item in items)
