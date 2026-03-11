import { describe, expect, it } from "vitest"

import {
  currentFeedbackFromRequestAction,
  normalizePaperFeedbackAction,
  normalizePaperPreferenceAction,
  togglePaperPreferenceAction,
  toggleSaveFeedbackAction,
} from "./paper-feedback"

describe("paper-feedback", () => {
  it("normalizes persisted feedback actions into active UI state", () => {
    expect(normalizePaperFeedbackAction("save")).toBe("save")
    expect(normalizePaperFeedbackAction("not_relevant")).toBe("dislike")
    expect(normalizePaperFeedbackAction("unlike")).toBeNull()
    expect(normalizePaperPreferenceAction("save")).toBeNull()
    expect(normalizePaperPreferenceAction("not_relevant")).toBe("dislike")
  })

  it("derives save toggle request actions from the current state", () => {
    expect(toggleSaveFeedbackAction(false)).toBe("save")
    expect(toggleSaveFeedbackAction(true)).toBe("unsave")
  })

  it("derives preference toggle request actions from the current state", () => {
    expect(togglePaperPreferenceAction(null, "like")).toBe("like")
    expect(togglePaperPreferenceAction("like", "like")).toBe("unlike")
    expect(togglePaperPreferenceAction("dislike", "dislike")).toBe("undislike")
  })

  it("maps request actions back into active state after a successful mutation", () => {
    expect(currentFeedbackFromRequestAction("save")).toBe("save")
    expect(currentFeedbackFromRequestAction("unsave")).toBeNull()
    expect(currentFeedbackFromRequestAction("undislike")).toBeNull()
  })
})
