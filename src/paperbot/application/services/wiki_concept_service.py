"""Build a wiki-oriented concept view grounded in stored papers and tracks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from paperbot.application.ports.wiki_concept_port import (
    GroundingSnapshot,
    PaperGroundingRecord,
    TrackGroundingRecord,
    WikiConceptPort,
)


@dataclass(frozen=True)
class WikiConceptSeed:
    id: str
    name: str
    description: str
    definition: str
    category: str
    icon: str
    related_concepts: Tuple[str, ...]
    examples: Tuple[str, ...]
    aliases: Tuple[str, ...] = ()

    @property
    def terms(self) -> Tuple[str, ...]:
        values = {self.name.lower(), self.id.lower()}
        values.update(alias.lower() for alias in self.aliases if alias.strip())
        return tuple(sorted(values))


@dataclass(frozen=True)
class WikiConceptView:
    id: str
    name: str
    description: str
    definition: str
    related_papers: List[str]
    related_concepts: List[str]
    examples: List[str]
    category: str
    icon: str
    paper_count: int
    track_count: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


_CATALOG: Tuple[WikiConceptSeed, ...] = (
    WikiConceptSeed(
        id="transformer",
        name="Transformer",
        description="A deep learning model architecture relying on self-attention mechanisms.",
        definition=(
            "The Transformer architecture processes input sequences in parallel using "
            "self-attention, allowing it to capture long-range dependencies without recurrent "
            "layers."
        ),
        category="Architecture",
        icon="layers",
        related_concepts=("Self-Attention", "Positional Encoding", "Multi-Head Attention"),
        examples=("GPT-4", "Claude", "LLaMA"),
        aliases=("transformer", "self-attention", "attention"),
    ),
    WikiConceptSeed(
        id="rlhf",
        name="RLHF",
        description="Reinforcement Learning from Human Feedback for preference alignment.",
        definition=(
            "RLHF trains a reward model on preference data and fine-tunes a base model to "
            "optimize that reward, improving alignment with human expectations."
        ),
        category="Method",
        icon="target",
        related_concepts=("Alignment", "Reward Model", "PPO"),
        examples=("ChatGPT alignment", "Claude training"),
        aliases=("rlhf", "reinforcement learning from human feedback", "alignment"),
    ),
    WikiConceptSeed(
        id="rag",
        name="Retrieval-Augmented Generation",
        description="A retrieval-plus-generation pattern that grounds responses with external context.",
        definition=(
            "RAG combines a retriever with a generator so the model can answer using retrieved "
            "documents, reducing hallucination and improving freshness."
        ),
        category="Method",
        icon="book-open",
        related_concepts=("Dense Retrieval", "Context Window", "Vector Index"),
        examples=("Paper QA", "Enterprise search copilots"),
        aliases=("rag", "retrieval-augmented generation", "retrieval augmented generation"),
    ),
    WikiConceptSeed(
        id="bleu",
        name="BLEU Score",
        description="A metric for evaluating overlap between generated and reference text.",
        definition=(
            "BLEU compares n-gram overlap between generated output and reference text. It is "
            "widely used in machine translation and still appears in summarization baselines."
        ),
        category="Metric",
        icon="bar-chart",
        related_concepts=("ROUGE", "METEOR", "BERTScore"),
        examples=("MT evaluation", "Summarization scoring"),
        aliases=("bleu", "bleu score"),
    ),
    WikiConceptSeed(
        id="diffusion",
        name="Diffusion Models",
        description="Generative models that learn to reverse a gradual noising process.",
        definition=(
            "Diffusion models corrupt data with noise over many steps and learn the reverse "
            "denoising process to synthesize new samples."
        ),
        category="Method",
        icon="waves",
        related_concepts=("Denoising", "Score Matching", "Latent Diffusion"),
        examples=("Stable Diffusion", "Imagen"),
        aliases=("diffusion", "diffusion model", "latent diffusion"),
    ),
    WikiConceptSeed(
        id="imagenet",
        name="ImageNet",
        description="A large-scale visual dataset that became a standard benchmark for vision models.",
        definition=(
            "ImageNet contains millions of labeled images and helped define the benchmark culture "
            "for large-scale visual recognition."
        ),
        category="Dataset",
        icon="image",
        related_concepts=("Transfer Learning", "Pretraining", "Fine-tuning"),
        examples=("ResNet-50 on ImageNet", "ViT benchmarks"),
        aliases=("imagenet",),
    ),
)


def _normalize_text(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    return text


def _contains_term(text: str, term: str) -> bool:
    if not text or not term:
        return False
    normalized_term = _normalize_text(term)
    if not normalized_term:
        return False
    if " " in normalized_term or "-" in normalized_term:
        return normalized_term in text
    if len(normalized_term) <= 4:
        return re.search(rf"\b{re.escape(normalized_term)}\b", text) is not None
    return normalized_term in text


def _paper_text(record: PaperGroundingRecord) -> str:
    values: List[str] = [
        record["title"],
        record["abstract"],
        " ".join(record["keywords"]),
        " ".join(record["fields_of_study"]),
    ]
    return _normalize_text(" ".join(values))


def _track_text(record: TrackGroundingRecord) -> str:
    values: List[str] = [
        record["name"],
        record["description"],
        " ".join(record["keywords"]),
        " ".join(record["methods"]),
    ]
    return _normalize_text(" ".join(values))


def _concept_query_matches(seed: WikiConceptSeed, item: WikiConceptView, query: str) -> bool:
    normalized_query = _normalize_text(query)
    if not normalized_query:
        return True
    haystack = _normalize_text(
        " ".join(
            [
                seed.name,
                seed.description,
                seed.definition,
                seed.category,
                " ".join(seed.terms),
                " ".join(item.related_papers),
                " ".join(item.related_concepts),
                " ".join(item.examples),
            ]
        )
    )
    return _contains_term(haystack, normalized_query)


class WikiConceptService:
    """Compose curated concept definitions with live paper and track grounding."""

    def __init__(self, concept_store: WikiConceptPort):
        self._concept_store = concept_store

    def list_concepts(
        self,
        *,
        user_id: str,
        query: str = "",
        category: Optional[str] = None,
        limit: int = 100,
    ) -> List[WikiConceptView]:
        snapshot = self._concept_store.load_grounding_snapshot(user_id=user_id)
        views = [self._build_view(seed, snapshot) for seed in _CATALOG]
        filtered = [
            item
            for seed, item in zip(_CATALOG, views)
            if self._category_matches(item.category, category)
            and _concept_query_matches(seed, item, query)
        ]
        filtered.sort(
            key=lambda item: (
                -self._query_score(item, query),
                -item.paper_count,
                -item.track_count,
                item.name.lower(),
            ),
        )
        return filtered[: max(1, limit)]

    @staticmethod
    def categories() -> List[str]:
        categories = sorted({seed.category for seed in _CATALOG})
        return ["All", *categories]

    def _build_view(self, seed: WikiConceptSeed, snapshot: GroundingSnapshot) -> WikiConceptView:
        paper_matches = self._matching_papers(seed, snapshot["papers"])
        track_matches = self._matching_tracks(seed, snapshot["tracks"])
        return WikiConceptView(
            id=seed.id,
            name=seed.name,
            description=seed.description,
            definition=seed.definition,
            related_papers=[paper["title"] for paper in paper_matches[:3]],
            related_concepts=list(seed.related_concepts),
            examples=list(seed.examples),
            category=seed.category,
            icon=seed.icon,
            paper_count=len(paper_matches),
            track_count=len(track_matches),
        )

    @staticmethod
    def _category_matches(actual: str, expected: Optional[str]) -> bool:
        if not expected or expected == "All":
            return True
        return actual.lower() == expected.strip().lower()

    @staticmethod
    def _query_score(item: WikiConceptView, query: str) -> int:
        normalized_query = _normalize_text(query)
        if not normalized_query:
            return 0
        score = 0
        if _contains_term(_normalize_text(item.name), normalized_query):
            score += 10
        if _contains_term(_normalize_text(item.category), normalized_query):
            score += 3
        if any(_contains_term(_normalize_text(paper), normalized_query) for paper in item.related_papers):
            score += 2
        return score

    @staticmethod
    def _matching_papers(
        seed: WikiConceptSeed, records: Sequence[PaperGroundingRecord]
    ) -> List[PaperGroundingRecord]:
        matches: List[PaperGroundingRecord] = []
        for record in records:
            text = _paper_text(record)
            if any(_contains_term(text, term) for term in seed.terms):
                matches.append(record)
        matches.sort(
            key=lambda row: (int(row["citation_count"]), int(row["year"] or 0), row["title"].lower()),
            reverse=True,
        )
        return matches

    @staticmethod
    def _matching_tracks(
        seed: WikiConceptSeed, records: Sequence[TrackGroundingRecord]
    ) -> List[TrackGroundingRecord]:
        matches: List[TrackGroundingRecord] = []
        for record in records:
            text = _track_text(record)
            if any(_contains_term(text, term) for term in seed.terms):
                matches.append(record)
        matches.sort(key=lambda row: row["name"].lower())
        return matches
