import { describe, expect, it } from "vitest"

import { buildStudioRuntimeInfo, formatRuntimePath } from "./studio-runtime"

describe("formatRuntimePath", () => {
  it("keeps short paths readable", () => {
    expect(formatRuntimePath("/tmp/work")).toBe("/tmp/work")
  })

  it("compresses deep paths to the trailing segments", () => {
    expect(formatRuntimePath("/home/master1/Projects/PaperBot")).toBe(".../Projects/PaperBot")
  })
})

describe("buildStudioRuntimeInfo", () => {
  it("returns a Claude Code runtime summary when CLI is available", () => {
    const info = buildStudioRuntimeInfo(
      {
        claude_cli: true,
        claude_path: "/usr/local/bin/claude",
        claude_version: "2.1.76",
      },
      {
        cwd: "/home/master1/Projects/PaperBot",
      },
    )

    expect(info.source).toBe("claude_code")
    expect(info.label).toBe("Claude Code")
    expect(info.statusLabel).toBe("CLI 2.1.76")
    expect(info.workspaceLabel).toBe(".../Projects/PaperBot")
    expect(info.detail).toContain("Running in")
  })

  it("returns fallback runtime information when Claude Code is unavailable", () => {
    const info = buildStudioRuntimeInfo(
      {
        claude_cli: false,
        fallback: "anthropic_api",
        error: "Failed to check Claude CLI status",
      },
      {
        cwd: "/tmp",
      },
    )

    expect(info.source).toBe("anthropic_api")
    expect(info.label).toBe("Anthropic API fallback")
    expect(info.statusLabel).toBe("Claude Code unavailable")
    expect(info.detail).toBe("Failed to check Claude CLI status")
    expect(info.workspaceLabel).toBe("/tmp")
  })
})
