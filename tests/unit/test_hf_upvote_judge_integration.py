"""Tests for HF upvotes integration into Judge scoring prompt."""
from paperbot.application.workflows.analysis.judge_prompts import (
    build_paper_judge_user_prompt,
)
from paperbot.application.workflows.analysis.judge_rubrics import default_judge_rubric


def test_upvotes_included_in_judge_prompt_when_present():
    paper = {
        "title": "FlashAttention-3",
        "snippet": "Faster attention with IO-awareness",
        "authors": ["Alice"],
        "venue": "NeurIPS 2026",
        "keywords": ["attention", "efficiency"],
        "upvotes": 42,
    }
    rubric = default_judge_rubric()
    prompt = build_paper_judge_user_prompt(query="efficient attention", paper=paper, rubric=rubric)

    assert "Community Upvotes (HuggingFace): 42" in prompt


def test_upvotes_omitted_from_judge_prompt_when_absent():
    paper = {
        "title": "FlashAttention-3",
        "snippet": "Faster attention with IO-awareness",
        "authors": ["Alice"],
        "venue": "NeurIPS 2026",
        "keywords": ["attention", "efficiency"],
    }
    rubric = default_judge_rubric()
    prompt = build_paper_judge_user_prompt(query="efficient attention", paper=paper, rubric=rubric)

    assert "Community Upvotes" not in prompt


def test_upvotes_zero_still_included():
    paper = {
        "title": "Test Paper",
        "snippet": "Abstract",
        "authors": [],
        "keywords": [],
        "upvotes": 0,
    }
    rubric = default_judge_rubric()
    prompt = build_paper_judge_user_prompt(query="test", paper=paper, rubric=rubric)

    assert "Community Upvotes (HuggingFace): 0" in prompt
