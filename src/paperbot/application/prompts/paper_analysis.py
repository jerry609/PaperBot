from __future__ import annotations

PAPER_SUMMARY_SYSTEM = """You are a concise research paper analyst.
Write a short summary in 2-3 sentences. Focus on:
1) core problem,
2) method,
3) key contribution or result.
Use plain language and avoid hype."""

PAPER_SUMMARY_USER = """Title: {title}
Abstract/Snippet: {abstract}

Return only the summary text."""

RELEVANCE_ASSESS_SYSTEM = """You assess how relevant a paper is to a user query.
Return strict JSON with fields:
- score: integer 0-100
- reason: short string (<= 40 words)
No markdown."""

RELEVANCE_ASSESS_USER = """Query: {query}
Title: {title}
Abstract/Snippet: {abstract}
Keywords: {keywords}"""
