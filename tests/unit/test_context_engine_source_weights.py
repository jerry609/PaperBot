from __future__ import annotations

from paperbot.context_engine.engine import _learn_source_weights_from_feedback


def test_learn_source_weights_returns_none_with_insufficient_samples() -> None:
    rows = [
        {"action": "like", "metadata": {"retrieval_sources": ["semantic_scholar"]}},
        {"action": "save", "metadata": {"retrieval_sources": ["semantic_scholar"]}},
    ]

    result = _learn_source_weights_from_feedback(
        feedback_rows=rows,
        selected_sources=["semantic_scholar", "arxiv"],
        default_weights={"semantic_scholar": 1.0, "arxiv": 0.8},
        min_samples=8,
    )

    assert result is None


def test_learn_source_weights_biases_toward_positive_feedback_sources() -> None:
    rows = [
        {"action": "save", "metadata": {"retrieval_sources": ["arxiv"]}},
        {"action": "like", "metadata": {"retrieval_sources": ["arxiv"]}},
        {"action": "save", "metadata": {"retrieval_sources": ["arxiv", "openalex"]}},
        {"action": "like", "metadata": {"retrieval_sources": ["arxiv"]}},
        {"action": "save", "metadata": {"retrieval_sources": ["arxiv"]}},
        {"action": "dislike", "metadata": {"retrieval_sources": ["semantic_scholar"]}},
        {"action": "not_relevant", "metadata": {"retrieval_sources": ["semantic_scholar"]}},
        {"action": "dislike", "metadata": {"retrieval_sources": ["semantic_scholar"]}},
    ]

    result = _learn_source_weights_from_feedback(
        feedback_rows=rows,
        selected_sources=["semantic_scholar", "arxiv", "openalex"],
        default_weights={"semantic_scholar": 1.0, "arxiv": 0.8, "openalex": 0.9},
        min_samples=4,
    )

    assert result is not None
    assert result["arxiv"] > result["semantic_scholar"]
    assert 0.3 <= result["semantic_scholar"] <= 1.8
    assert 0.3 <= result["arxiv"] <= 1.8
