import { describe, expect, it, vi } from "vitest"

const { apiBaseUrlMock, proxyJsonMock } = vi.hoisted(() => ({
  apiBaseUrlMock: vi.fn(() => "http://backend.example.com"),
  proxyJsonMock: vi.fn(),
}))

vi.mock("@/app/api/_utils/backend-proxy", () => ({
  apiBaseUrl: apiBaseUrlMock,
  proxyJson: proxyJsonMock,
}))

import { GET } from "./route"

describe("studio status route", () => {
  it("proxies the backend request with a fallback response", async () => {
    const req = new Request("http://localhost/api/studio/status")
    proxyJsonMock.mockResolvedValueOnce(Response.json({ claude_cli: true }))

    await GET(req)

    expect(proxyJsonMock).toHaveBeenCalledWith(
      req,
      "http://backend.example.com/api/studio/status",
      "GET",
      expect.objectContaining({
        onError: expect.any(Function),
      }),
    )

    const options = vi.mocked(proxyJsonMock).mock.calls[0][3]
    expect(options).toBeDefined()

    const fallback = options?.onError?.({
      error: new Error("offline"),
      isTimeout: false,
      upstreamUrl: "http://backend.example.com/api/studio/status",
    })

    expect(fallback).toBeInstanceOf(Response)
    expect(fallback?.status).toBe(500)
    await expect(fallback?.json()).resolves.toEqual({
      claude_cli: false,
      claude_agent_sdk: false,
      chat_surface: "managed_session",
      chat_transport: "anthropic_api",
      preferred_chat_transport: "claude_agent_sdk",
      slash_commands: ["help", "status", "new", "clear", "plan", "model", "agents", "mcp", "auth", "doctor"],
      permission_profiles: ["default", "full_access"],
      runtime_commands: ["agents", "auth", "doctor", "mcp"],
      error: "Failed to check Claude CLI status",
      fallback: "anthropic_api",
    })
  })
})
