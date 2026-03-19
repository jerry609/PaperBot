import { describe, expect, it, vi } from "vitest"

import type { SavedPaperItem, ReadingStatus } from "./SavedPapersList"

// We import the default component only to ensure the module compiles,
// but these tests focus on the pure helper logic.
import SavedPapersList, {
  formatReadingStatusLabel,
  matchesSavedPaperView,
} from "./SavedPapersList"

describe("formatReadingStatusLabel", () => {
  it("maps internal reading status values to user-facing labels", () => {
    const cases: Array<[ReadingStatus, string]> = [
      ["unread", "To read"],
      ["reading", "Reading"],
      ["read", "Read"],
      ["archived", "Archived"],
    ]

    for (const [status, expected] of cases) {
      expect(formatReadingStatusLabel(status)).toBe(expected)
    }
  })
})

describe("SavedPapersList module", () => {
  it("defines a SavedPaperItem shape compatible with reading status updates", () => {
    const item: SavedPaperItem = {
      paper: {
        id: 1,
        title: "Test Paper",
      },
      saved_at: null,
      track_id: null,
      action: "save",
      reading_status: {
        status: "unread",
        updated_at: null,
      },
      latest_judge: null,
      provenance: {
        primary: "manual_save",
        labels: ["Saved manually"],
        is_manual: true,
        is_workflow: false,
      },
    }

    expect(item.paper.id).toBe(1)
    expect(item.reading_status?.status).toBe("unread")
  })

  it("matches saved paper view filters using provenance and workflow review", () => {
    const manualItem: SavedPaperItem = {
      paper: {
        id: 1,
        title: "Manual Paper",
      },
      provenance: {
        primary: "manual_save",
        labels: ["Saved manually"],
        is_manual: true,
        is_workflow: false,
      },
      latest_judge: null,
    }

    const workflowItem: SavedPaperItem = {
      paper: {
        id: 2,
        title: "Workflow Paper",
      },
      provenance: {
        primary: "daily_brief",
        labels: ["Daily Brief", "Workflow reviewed"],
        is_manual: false,
        is_workflow: true,
      },
      latest_judge: {
        overall: 4.5,
        recommendation: "must_read",
        one_line_summary: "Worth routing into the active track.",
      },
    }

    expect(matchesSavedPaperView(manualItem, "manual")).toBe(true)
    expect(matchesSavedPaperView(manualItem, "workflow")).toBe(false)
    expect(matchesSavedPaperView(workflowItem, "workflow")).toBe(true)
    expect(matchesSavedPaperView(workflowItem, "manual")).toBe(false)
  })
})
