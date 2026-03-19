import { describe, expect, it } from "vitest"

import {
  buildDashboardIntelligenceCards,
  summarizeDashboardIntelligence,
} from "./dashboard-intelligence"
import type { IntelligenceFeedItem } from "./types"

describe("dashboard-intelligence", () => {
  it("maps external radar signals into dashboard cards", () => {
    const items: IntelligenceFeedItem[] = [
      {
        id: "reddit:keyword:rag",
        source: "reddit",
        source_label: "Reddit Search",
        kind: "keyword_spike",
        title: "Reddit spike: rag",
        summary: "24h mentions: 12 across r/MachineLearning. Top post: RAG agents are back.",
        url: "https://reddit.example/rag",
        keyword_hits: ["rag"],
        author_matches: ["Alice Zhang"],
        repo_matches: ["org/rag-agent"],
        match_reasons: ["keyword: rag", "delta: +5", "author: Alice Zhang", "repo: org/rag-agent"],
        score: 92,
        metric: {
          name: "mentions/24h",
          value: 12,
          delta: 5,
        },
        published_at: "2026-03-10T10:00:00+00:00",
        detected_at: "2026-03-10T11:00:00+00:00",
        matched_tracks: [
          {
            track_id: 7,
            track_name: "RAG Agents",
            matched_keywords: ["rag"],
          },
        ],
        research_query: "rag, org/rag-agent",
        payload: {},
      },
    ]

    const [card] = buildDashboardIntelligenceCards(items)

    expect(card.title).toBe("Reddit spike: rag")
    expect(card.metricLabel).toBe("mentions/24h 12 (+5)")
    expect(card.reasonChips).toEqual([
      "keyword: rag",
      "delta: +5",
      "author: Alice Zhang",
      "repo: org/rag-agent",
    ])
    expect(card.href).toBe("https://reddit.example/rag")
    expect(card.isExternal).toBe(true)
    expect(card.researchHref).toBe("/research?track_id=7&query=rag%2C+org%2Frag-agent")
    expect(card.matchedTrackNames).toEqual(["RAG Agents"])
  })

  it("falls back to synthesized reason chips and empty state copy", () => {
    const items: IntelligenceFeedItem[] = [
      {
        id: "hf:1",
        source: "huggingface",
        source_label: "HF Daily Papers",
        kind: "paper_buzz",
        title: "A paper",
        summary: "Signal summary",
        keyword_hits: ["agents"],
        author_matches: ["Jane Doe"],
        repo_matches: ["paperbot/paperbot"],
        match_reasons: [],
        score: 70,
        metric: {
          name: "upvotes",
          value: 9,
          delta: 0,
        },
        matched_tracks: [],
        research_query: "",
        payload: {},
      },
    ]

    const [card] = buildDashboardIntelligenceCards(items)
    const [emptyCard] = buildDashboardIntelligenceCards([])

    expect(card.reasonChips).toEqual([
      "keyword: agents",
      "author: Jane Doe",
      "repo: paperbot/paperbot",
    ])
    expect(emptyCard.title).toBe("No urgent community signals")
    expect(emptyCard.metricLabel).toBe("stable")
    expect(emptyCard.researchHref).toBe("/research")
  })

  it("keeps the full mapped signal list and summarizes matching state", () => {
    const items: IntelligenceFeedItem[] = [
      {
        id: "reddit:1",
        source: "reddit",
        source_label: "Reddit Search",
        kind: "keyword_spike",
        title: "Signal A",
        summary: "Summary A",
        keyword_hits: ["rag"],
        author_matches: [],
        repo_matches: [],
        match_reasons: [],
        score: 88,
        metric: {
          name: "mentions/24h",
          value: 10,
          delta: 4,
        },
        matched_tracks: [
          {
            track_id: 1,
            track_name: "RAG",
            matched_keywords: ["rag"],
          },
        ],
        research_query: "rag",
        payload: {},
      },
      {
        id: "github:1",
        source: "github",
        source_label: "GitHub Watch",
        kind: "repo_spike",
        title: "Signal B",
        summary: "Summary B",
        keyword_hits: [],
        author_matches: [],
        repo_matches: ["paperbot/paperbot"],
        match_reasons: [],
        score: 64,
        metric: {
          name: "stars/day",
          value: 2,
          delta: 0,
        },
        matched_tracks: [],
        research_query: "",
        payload: {},
      },
    ]

    const cards = buildDashboardIntelligenceCards(items)
    const summary = summarizeDashboardIntelligence(items)

    expect(cards).toHaveLength(2)
    expect(cards[1]?.title).toBe("Signal B")
    expect(summary).toEqual({
      totalCount: 2,
      matchedCount: 1,
      risingCount: 1,
      sourceCount: 2,
    })
  })
})
