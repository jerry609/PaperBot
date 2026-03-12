import { describe, expect, it } from "vitest"

import { buildDashboardDailyBrief } from "./dashboard-brief"

describe("dashboard-brief", () => {
  it("builds dashboard highlights, query pulse, and trend rows from a daily report", () => {
    const brief = buildDashboardDailyBrief({
      title: "DailyPaper Digest",
      date: "2026-03-12",
      source: "papers_cool",
      sources: ["papers_cool", "hf_daily"],
      stats: {
        unique_items: 8,
        total_query_hits: 21,
        query_count: 2,
      },
      queries: [
        {
          normalized_query: "agentic retrieval",
          top_items: [
            {
              paper_id: "paper-1",
              title: "Agentic Retrieval for RAG",
              url: "https://example.com/paper-1",
              score: 8.6,
              subject_or_venue: "ICLR 2026",
              matched_keywords: ["agentic retrieval", "graph rag"],
              sources: ["papers_cool"],
              branches: ["arxiv"],
              judge: {
                overall: 4.7,
                recommendation: "must_read",
              },
              digest_card: {
                highlight: "A strong retrieval policy with graph memory.",
                tags: ["retrieval", "memory"],
              },
            },
          ],
        },
        {
          normalized_query: "test-time scaling",
          top_items: [
            {
              paper_id: "paper-2",
              title: "Test-Time Scaling with Verifiers",
              external_url: "https://example.com/paper-2",
              score: 8.1,
              venue: "arXiv",
              matched_keywords: ["test-time scaling"],
            },
          ],
        },
      ],
      llm_analysis: {
        query_trends: [
          {
            query: "agentic retrieval",
            analysis: "Repository activity and new candidates are rising together.",
          },
        ],
      },
    })

    expect(brief.sourceLabel).toBe("papers.cool")
    expect(brief.sourceBadges).toEqual(["papers.cool", "HF Daily"])
    expect(brief.stats).toEqual({
      uniqueItems: 8,
      totalQueryHits: 21,
      queryCount: 2,
    })
    expect(brief.highlights[0]).toMatchObject({
      id: "paper-1",
      title: "Agentic Retrieval for RAG",
      href: "https://example.com/paper-1",
      queryLabel: "agentic retrieval",
      venueLabel: "ICLR 2026",
      metricLabel: "Judge 4.7",
      recommendation: "Must Read",
    })
    expect(brief.queryPulse).toEqual([
      { query: "agentic retrieval", hits: 1 },
      { query: "test-time scaling", hits: 1 },
    ])
    expect(brief.trendRows).toEqual([
      {
        query: "agentic retrieval",
        analysis: "Repository activity and new candidates are rising together.",
      },
    ])
  })
})
