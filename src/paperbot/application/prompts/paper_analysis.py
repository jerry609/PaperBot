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

STRUCTURED_CARD_SYSTEM = """You are a research paper analyst. Extract structured information from the paper.
Return ONLY valid JSON, no markdown."""

STRUCTURED_CARD_USER = """Title: {title}
Abstract: {abstract}

Extract JSON with these fields:
- method: Core methodology or approach (1-2 sentences)
- dataset: Datasets or benchmarks used (1 sentence, "N/A" if not mentioned)
- conclusion: Main finding or contribution (1-2 sentences)
- limitations: Key limitations or caveats (1 sentence, "Not stated" if unclear)

Return ONLY valid JSON, no markdown."""

RELATED_WORK_SYSTEM = """You are an academic writing assistant. Generate a Related Work section draft.
Every claim MUST cite a paper using [AuthorYear] format. Do NOT invent citations."""

RELATED_WORK_USER = """Topic: {topic}

Papers:
{papers_formatted}

Write a coherent Related Work section (3-5 paragraphs) in academic English.
Group papers by sub-theme. Each paragraph should cover one sub-theme.
End with a brief summary of the research gap.

At the end, list all citations used in the format:
## References
- [Author2025] Full Title. Venue, Year."""

DAILY_DIGEST_CARD_SYSTEM = """You are a research paper analyst writing a daily digest card.
Extract structured information for a newsletter audience.
Return ONLY valid JSON, no markdown."""

DAILY_DIGEST_CARD_USER = """Title: {title}
Abstract: {abstract}

Extract JSON with these fields:
- highlight: One-sentence key takeaway for a newsletter reader (max 30 words)
- method: Core methodology or approach (1 sentence)
- finding: Main result or contribution (1 sentence)
- tags: Array of 2-4 short topic tags (e.g. ["LLM", "efficiency", "KV cache"])

Return ONLY valid JSON, no markdown."""
