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
        claude_agent_sdk: false,
        claude_path: "/usr/local/bin/claude",
        claude_version: "2.1.76",
        chat_surface: "managed_session",
        chat_transport: "claude_cli_print",
        preferred_chat_transport: "claude_agent_sdk",
        slash_commands: ["help", "status", "plan", "model", "agents", "mcp", "auth", "doctor"],
        permission_profiles: ["default", "full_access"],
        runtime_commands: ["agents", "auth", "doctor", "mcp"],
        code_mode_enabled: true,
        known_model_aliases: ["sonnet", "opus"],
        opencode_cli: true,
        opencode_path: "/usr/local/bin/opencode",
        opencode_version: "1.2.26",
      },
      {
        cwd: "/home/master1/Projects/PaperBot",
      },
    )

    expect(info.source).toBe("claude_code")
    expect(info.label).toBe("Claude Code")
    expect(info.chatSurface).toBe("managed_session")
    expect(info.chatTransport).toBe("claude_cli_print")
    expect(info.preferredChatTransport).toBe("claude_agent_sdk")
    expect(info.claudeAgentSdkAvailable).toBe(false)
    expect(info.supportedSlashCommands).toContain("doctor")
    expect(info.supportedPermissionProfiles).toEqual(["default", "full_access"])
    expect(info.runtimeCommands).toContain("mcp")
    expect(info.statusLabel).toBe("Managed chat · CLI print · CLI 2.1.76")
    expect(info.codeModeEnabled).toBe(true)
    expect(info.knownModelAliases).toEqual(["sonnet", "opus"])
    expect(info.opencodeAvailable).toBe(true)
    expect(info.opencodeVersion).toBe("1.2.26")
    expect(info.workspaceLabel).toBe(".../Projects/PaperBot")
    expect(info.detail).toContain("Install claude_agent_sdk")
  })

  it("returns fallback runtime information when Claude Code is unavailable", () => {
    const info = buildStudioRuntimeInfo(
      {
        claude_cli: false,
        claude_agent_sdk: false,
        chat_surface: "managed_session",
        chat_transport: "anthropic_api",
        preferred_chat_transport: "claude_agent_sdk",
        slash_commands: ["help", "status", "plan", "model", "agents", "mcp", "auth", "doctor"],
        permission_profiles: ["default", "full_access"],
        runtime_commands: ["agents", "auth", "doctor", "mcp"],
        fallback: "anthropic_api",
        error: "Failed to check Claude CLI status",
        code_mode_enabled: false,
        known_model_aliases: ["sonnet", "opus"],
      },
      {
        cwd: "/tmp",
      },
    )

    expect(info.source).toBe("anthropic_api")
    expect(info.label).toBe("Managed chat fallback")
    expect(info.chatTransport).toBe("anthropic_api")
    expect(info.supportedSlashCommands).toContain("plan")
    expect(info.supportedPermissionProfiles).toEqual(["default", "full_access"])
    expect(info.statusLabel).toBe("Managed chat · API fallback")
    expect(info.codeModeEnabled).toBe(false)
    expect(info.knownModelAliases).toEqual(["sonnet", "opus"])
    expect(info.detail).toBe("Failed to check Claude CLI status")
    expect(info.workspaceLabel).toBe("/tmp")
  })
})
