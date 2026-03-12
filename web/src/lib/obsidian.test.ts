import { describe, expect, it } from "vitest"

import { buildObsidianExportCommand, describeObsidianScope } from "./obsidian"

describe("buildObsidianExportCommand", () => {
  it("builds a track-scoped command with quoted values", () => {
    expect(
      buildObsidianExportCommand({
        vaultPath: "~/vaults/research",
        rootDir: "Research Notes",
        userId: "default",
        trackId: 7,
      })
    ).toBe(
      "paperbot export obsidian --user-id 'default' --track-id 7 --root-dir 'Research Notes' --vault '~/vaults/research'"
    )
  })

  it("falls back to a placeholder path when the vault is empty", () => {
    expect(
      buildObsidianExportCommand({
        vaultPath: "   ",
      })
    ).toBe("paperbot export obsidian --user-id 'default' --vault '/path/to/your/vault'")
  })
})

describe("describeObsidianScope", () => {
  it("describes track scope when a track is selected", () => {
    expect(describeObsidianScope(12, "Agent Memory")).toBe("Track: Agent Memory")
  })

  it("falls back to global scope text without a track", () => {
    expect(describeObsidianScope(null, null)).toBe("Global saved library")
  })
})
