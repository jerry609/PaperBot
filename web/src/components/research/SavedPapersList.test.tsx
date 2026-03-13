import { describe, expect, it, vi } from "vitest"

import type { SavedPaperItem, ReadingStatus } from "./SavedPapersList"

// We import the default component only to ensure the module compiles,
// but these tests focus on the pure helper logic.
import SavedPapersList, { formatReadingStatusLabel } from "./SavedPapersList"

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
    }

    expect(item.paper.id).toBe(1)
    expect(item.reading_status?.status).toBe("unread")
  })
})

