import { mkdtemp, rm, utimes, writeFile } from "node:fs/promises"
import os from "node:os"
import path from "node:path"

import { afterEach, describe, expect, it } from "vitest"

import { buildDashboardBrief, fetchLatestDashboardBrief } from "./dashboard-brief"

describe("dashboard-brief", () => {
  const originalOutputDir = process.env.PAPERBOT_DAILYPAPER_OUTPUT_DIR
  const tempDirs: string[] = []

  afterEach(async () => {
    if (originalOutputDir === undefined) {
      delete process.env.PAPERBOT_DAILYPAPER_OUTPUT_DIR
    } else {
      process.env.PAPERBOT_DAILYPAPER_OUTPUT_DIR = originalOutputDir
    }

    await Promise.all(tempDirs.splice(0).map((dir) => rm(dir, { force: true, recursive: true })))
  })

  it("builds deduped recommendations from a DailyPaper report", () => {
    const snapshot = buildDashboardBrief({
      title: "Daily Picks",
      generated_at: "2026-03-12T08:00:00Z",
      source: "hf_daily",
      global_top: [
        {
          paper_id: "2503.12345",
          title: "Scaling Laws for Retrieval-Augmented Generation",
          external_url: "https://arxiv.org/abs/2503.12345",
          score: 9.1,
          snippet: "A fallback summary",
          matched_keywords: ["RAG", "Context"],
          digest_card: {
            highlight: "Shows log-linear gains for larger retrieval pools.",
            tags: ["RAG", "Scaling Laws"],
          },
          judge: {
            recommendation: "must_read",
          },
          authors: ["Alice Smith", "Bob Jones"],
          year: 2026,
        },
        {
          paper_id: "2503.12345",
          title: "Duplicate entry",
          external_url: "https://arxiv.org/abs/2503.12345",
        },
      ],
      llm_analysis: {
        query_trends: [
          { query: "rag", analysis: "Momentum is accelerating across benchmarks." },
          { query: "agents", analysis: "Agent evaluation remains a top thread." },
          { query: "", analysis: "Should be filtered." },
        ],
      },
    })

    expect(snapshot.title).toBe("Daily Picks")
    expect(snapshot.sourceLabel).toBe("HF Daily")
    expect(snapshot.recommendations).toHaveLength(1)
    expect(snapshot.recommendations[0]).toMatchObject({
      id: "2503.12345",
      paperId: "2503.12345",
      title: "Scaling Laws for Retrieval-Augmented Generation",
      href: "https://arxiv.org/abs/2503.12345",
      summary: "Shows log-linear gains for larger retrieval pools.",
      tags: ["RAG", "Scaling Laws", "Context"],
      metric: "Score 9.1",
      recommendation: "Must Read",
      authors: ["Alice Smith", "Bob Jones"],
      year: 2026,
      paperSource: "arxiv",
    })
    expect(snapshot.trendRows).toEqual([
      { query: "rag", analysis: "Momentum is accelerating across benchmarks." },
      { query: "agents", analysis: "Agent evaluation remains a top thread." },
    ])
  })

  it("reads the newest report from PAPERBOT_DAILYPAPER_OUTPUT_DIR", async () => {
    const tempDir = await mkdtemp(path.join(os.tmpdir(), "paperbot-dashboard-brief-"))
    tempDirs.push(tempDir)
    process.env.PAPERBOT_DAILYPAPER_OUTPUT_DIR = tempDir

    const olderPath = path.join(tempDir, "older.json")
    const newerPath = path.join(tempDir, "newer.json")

    await writeFile(
      olderPath,
      JSON.stringify({
        title: "Older Brief",
        queries: [
          {
            normalized_query: "agents",
            top_items: [
              {
                paper_id: "older-paper",
                title: "Older paper",
                url: "https://example.com/older",
                judge: { one_line_summary: "Older summary" },
              },
            ],
          },
        ],
      }),
      "utf-8",
    )
    await writeFile(
      newerPath,
      JSON.stringify({
        title: "Newer Brief",
        queries: [
          {
            normalized_query: "memory",
            top_items: [
              {
                paper_id: "fresh-paper",
                title: "Fresh paper",
                external_url: "https://openalex.org/W123",
                matched_keywords: ["Memory"],
                judge: {
                  recommendation: "worth_reading",
                  one_line_summary: "Fresh summary",
                },
              },
            ],
          },
        ],
      }),
      "utf-8",
    )

    const olderTime = new Date("2026-03-12T08:00:00Z")
    const newerTime = new Date("2026-03-12T09:00:00Z")
    await utimes(olderPath, olderTime, olderTime)
    await utimes(newerPath, newerTime, newerTime)

    const snapshot = await fetchLatestDashboardBrief()

    expect(snapshot).not.toBeNull()
    expect(snapshot?.title).toBe("Newer Brief")
    expect(snapshot?.recommendations[0]).toMatchObject({
      id: "fresh-paper",
      title: "Fresh paper",
      summary: "Fresh summary",
      recommendation: "Worth Reading",
      paperSource: "openalex",
    })
  })
})
