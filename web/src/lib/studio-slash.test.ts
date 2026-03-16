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

  it("parses claude command palette entries", () => {
    expect(parseStudioSlashCommand("/mcp list", knownModelAliases)).toEqual({
      kind: "quick_command",
      runtime: "claude",
      presetId: "claude-mcp",
      args: "list",
    })
    expect(parseStudioSlashCommand("/doctor", knownModelAliases)).toEqual({
      kind: "quick_command",
      runtime: "claude",
      presetId: "claude-doctor",
      args: "",
    })
  })

  it("parses local session slash commands", () => {
    expect(parseStudioSlashCommand("/help", knownModelAliases)).toEqual({ kind: "help" })
    expect(parseStudioSlashCommand("/status", knownModelAliases)).toEqual({ kind: "status" })
    expect(parseStudioSlashCommand("/clear", knownModelAliases)).toEqual({ kind: "clear" })
    expect(parseStudioSlashCommand("/new", knownModelAliases)).toEqual({ kind: "new_thread" })
  })

  it("rejects legacy studio-only slash commands", () => {
    expect(parseStudioSlashCommand("/code fix the loader", knownModelAliases)).toBeNull()
    expect(parseStudioSlashCommand("/monitor", knownModelAliases)).toBeNull()
    expect(parseStudioSlashCommand("/resume session-123", knownModelAliases)).toBeNull()
    expect(parseStudioSlashCommand("/effort-high fix the flaky test", knownModelAliases)).toBeNull()
  })

  it("no longer treats /codex as a direct delegation shortcut", () => {
    expect(parseStudioSlashCommand("/codex implement the loader", knownModelAliases)).toBeNull()
  })

  it("returns null for unrecognized slash commands", () => {
    expect(parseStudioSlashCommand("/not-a-real-command", knownModelAliases)).toBeNull()
  })
})
