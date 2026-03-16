import { describe, expect, it } from "vitest"

import { buildStudioApprovalContinuePrompt, parseStudioApprovalRequest } from "./studio-approval"

describe("parseStudioApprovalRequest", () => {
  it("extracts command and worker agent id from Claude approval text", () => {
    const parsed = parseStudioApprovalRequest(
      [
        "The command requires approval from you to run. Could you approve the `git -C /home/master1/PaperBot branch --show-current` command, or run it yourself and share the output?",
        "agentId: afec8e10340629da4 (for resuming to continue this agent's work if needed)",
      ].join("\n"),
    )

    expect(parsed).toEqual({
      message: [
        "The command requires approval from you to run. Could you approve the `git -C /home/master1/PaperBot branch --show-current` command, or run it yourself and share the output?",
        "agentId: afec8e10340629da4 (for resuming to continue this agent's work if needed)",
      ].join("\n"),
      command: "git -C /home/master1/PaperBot branch --show-current",
      workerAgentId: "afec8e10340629da4",
    })
  })

  it("returns null for plain tool output", () => {
    expect(parseStudioApprovalRequest("test/milestone-v1.2")).toBeNull()
  })
})

describe("buildStudioApprovalContinuePrompt", () => {
  it("includes command and worker resume hint", () => {
    const prompt = buildStudioApprovalContinuePrompt({
      command: "git -C /home/master1/PaperBot branch --show-current",
      workerAgentId: "afec8e10340629da4",
    })

    expect(prompt).toContain("Approved command: `git -C /home/master1/PaperBot branch --show-current`")
    expect(prompt).toContain("resume worker agentId: afec8e10340629da4")
    expect(prompt).toContain("Finish the task")
  })
})
