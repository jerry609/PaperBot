import { describe, expect, it } from "vitest"
import { renderToStaticMarkup } from "react-dom/server"

import type { ResearchTrackContextResponse } from "@/lib/types"

import { ResearchTrackContextPanel, buildStatItems } from "./ResearchTrackContextPanel"

const TRACK_CONTEXT_FIXTURE: ResearchTrackContextResponse = {
  user_id: "default",
  track_id: 7,
  track: {
    id: 7,
    name: "Agentic Retrieval",
    description: "Keep retrieval quality and latency balanced.",
    keywords: ["rag", "retrieval", "reranking"],
    is_active: true,
  },
  tasks: [
    { id: 1, track_id: 7, title: "Validate reranker", status: "todo" },
    { id: 2, track_id: 7, title: "Compare OpenAlex recall", status: "doing" },
  ],
  milestones: [
    { id: 9, track_id: 7, name: "Freeze benchmark slice", status: "doing" },
  ],
  memory: {
    total_items: 6,
    approved_items: 4,
    pending_items: 2,
    top_tags: ["retrieval", "latency"],
    latest_memory_at: "2026-03-12T08:00:00+00:00",
  },
  feedback: {
    total_items: 5,
    actions: { save: 3, like: 2 },
    latest_feedback_at: "2026-03-12T08:30:00+00:00",
    recent_items: [],
  },
  saved_papers: {
    total_items: 3,
    latest_saved_at: "2026-03-12T08:40:00+00:00",
    recent_items: [],
  },
  eval_summary: {
    feedback_coverage: 0.75,
  },
}

describe("ResearchTrackContextPanel", () => {
  it("builds compact stat cards from the track snapshot", () => {
    const stats = buildStatItems(TRACK_CONTEXT_FIXTURE)

    expect(stats).toHaveLength(3)
    expect(stats.map((item) => ({ label: item.label, value: item.value }))).toEqual([
      { label: "Pending Memory", value: "2" },
      { label: "Saved Papers", value: "3" },
      { label: "Feedback", value: "5" },
    ])
  })

  it("renders the consolidated track snapshot details", () => {
    const html = renderToStaticMarkup(
      <ResearchTrackContextPanel
        context={TRACK_CONTEXT_FIXTURE}
        onOpenMemory={() => undefined}
      />
    )

    expect(html).toContain("Track Snapshot")
    expect(html).toContain("Agentic Retrieval")
    expect(html).toContain("Pending Memory")
    expect(html).toContain("Validate reranker")
    expect(html).toContain("Freeze benchmark slice")
    expect(html).toContain("save: 3")
    expect(html).toContain("75%")
  })
})
