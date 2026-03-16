import { describe, expect, it } from "vitest"

import { parseStudioSlashCommand } from "./studio-slash"

describe("parseStudioSlashCommand", () => {
  const knownModelAliases = ["sonnet", "opus"]

  it("parses mode commands with inline remainder", () => {
    expect(parseStudioSlashCommand("/plan map the architecture", knownModelAliases)).toEqual({
      kind: "mode",
      mode: "Plan",
      remainder: "map the architecture",
    })
  })

  it("parses generic model commands", () => {
    expect(parseStudioSlashCommand("/model opus compare tradeoffs", knownModelAliases)).toEqual({
      kind: "model",
      modelOption: "opus",
      customModel: "",
      remainder: "compare tradeoffs",
    })
  })

  it("parses custom model commands", () => {
    expect(parseStudioSlashCommand("/model claude-sonnet-4-6", knownModelAliases)).toEqual({
      kind: "model",
      modelOption: "custom",
      customModel: "claude-sonnet-4-6",
      remainder: "",
    })
  })

  it("parses effort aliases", () => {
    expect(parseStudioSlashCommand("/effort-high fix the flaky test", knownModelAliases)).toEqual({
      kind: "effort",
      effort: "high",
      remainder: "fix the flaky test",
    })
  })

  it("parses advanced option setters", () => {
    expect(parseStudioSlashCommand("/resume session-123", knownModelAliases)).toEqual({
      kind: "resume",
      value: "session-123",
    })
    expect(parseStudioSlashCommand("/agent reviewer", knownModelAliases)).toEqual({
      kind: "agent",
      value: "reviewer",
    })
    expect(parseStudioSlashCommand("/tools Bash,Read", knownModelAliases)).toEqual({
      kind: "tools",
      value: "Bash,Read",
    })
  })

  it("parses quick commands", () => {
    expect(parseStudioSlashCommand("/claude-mcp list", knownModelAliases)).toEqual({
      kind: "quick_command",
      runtime: "claude",
      presetId: "claude-mcp",
      args: "list",
    })
  })

  it("parses monitor navigation slash commands", () => {
    expect(parseStudioSlashCommand("/monitor", knownModelAliases)).toEqual({
      kind: "view",
      view: "board",
    })
  })

  it("no longer treats /codex as a direct delegation shortcut", () => {
    expect(parseStudioSlashCommand("/codex implement the loader", knownModelAliases)).toBeNull()
  })

  it("returns null for unrecognized slash commands", () => {
    expect(parseStudioSlashCommand("/not-a-real-command", knownModelAliases)).toBeNull()
  })
})
