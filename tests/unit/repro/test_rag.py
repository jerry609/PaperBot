"""Pytest coverage for repro RAG knowledge base."""

from __future__ import annotations

from repro.rag.knowledge_base import BUILTIN_PATTERNS, CodeKnowledgeBase, CodePattern


def test_code_pattern_to_context_renders_metadata_and_code():
    pattern = CodePattern(
        name="training_loop",
        description="Training loop with validation",
        code="for batch in loader:\n    pass",
        tags=["training", "pytorch"],
        source="unit test",
    )

    context = pattern.to_context()

    assert "# Pattern: training_loop" in context
    assert "# Training loop with validation" in context
    assert "for batch in loader" in context


def test_knowledge_base_add_search_and_get_pattern():
    kb = CodeKnowledgeBase()
    pattern = CodePattern(
        name="transformer_encoder",
        description="Encoder block",
        code="# encoder",
        tags=["attention", "transformer"],
    )
    kb.add_pattern(pattern)

    results = kb.search("transformer encoder", k=3)

    assert len(results) == 1
    assert results[0].name == "transformer_encoder"
    assert kb.get_pattern("transformer_encoder") is pattern


def test_knowledge_base_search_is_case_insensitive_and_prefers_tag_matches():
    kb = CodeKnowledgeBase()
    kb.add_pattern(
        CodePattern(
            name="high_score",
            description="Other description",
            code="# code",
            tags=["pytorch", "training"],
        )
    )
    kb.add_pattern(
        CodePattern(
            name="low_score",
            description="Something about pytorch training",
            code="# code",
            tags=["other"],
        )
    )

    upper_results = kb.search("PYTORCH", k=3)
    ranked_results = kb.search("pytorch training", k=2)

    assert len(upper_results) == 2
    assert ranked_results[0].name == "high_score"


def test_knowledge_base_lists_sorted_tags_and_patterns():
    kb = CodeKnowledgeBase()
    kb.add_pattern(CodePattern(name="p1", description="", code="", tags=["beta", "alpha"]))
    kb.add_pattern(CodePattern(name="p2", description="", code="", tags=["gamma", "alpha"]))

    assert kb.list_tags() == ["alpha", "beta", "gamma"]
    assert kb.list_patterns() == ["p1", "p2"]


def test_knowledge_base_serialization_keeps_summary_fields():
    kb = CodeKnowledgeBase()
    kb.add_pattern(
        CodePattern(
            name="test",
            description="Test pattern",
            code="x = 1",
            tags=["tag1", "tag2"],
            source="unit test",
        )
    )

    payload = kb.to_dict()

    assert payload["patterns"]["test"]["description"] == "Test pattern"
    assert payload["patterns"]["test"]["code_length"] == 5
    assert "tag_index" in payload


def test_builtin_patterns_are_registered_and_compile():
    kb = CodeKnowledgeBase.from_builtin()

    assert len(kb.patterns) > 0
    assert "pytorch_training_loop" in [pattern.name for pattern in kb.search("training loop", k=5)]
    assert "transformer_encoder_block" in [
        pattern.name for pattern in kb.search("transformer attention encoder", k=5)
    ]

    for pattern in BUILTIN_PATTERNS:
        compile(pattern.code, f"<{pattern.name}>", "exec")
