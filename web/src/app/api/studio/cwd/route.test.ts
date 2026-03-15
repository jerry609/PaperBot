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

describe("studio cwd route", () => {
  it("proxies the backend request with a fallback response", async () => {
    const req = new Request("http://localhost/api/studio/cwd")
    proxyJsonMock.mockResolvedValueOnce(Response.json({ cwd: "/workspace" }))

    await GET(req)

    expect(proxyJsonMock).toHaveBeenCalledWith(
      req,
      "http://backend.example.com/api/studio/cwd",
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
      upstreamUrl: "http://backend.example.com/api/studio/cwd",
    })

    expect(fallback).toBeInstanceOf(Response)
    expect(fallback?.status).toBe(200)
    await expect(fallback?.json()).resolves.toEqual({
      cwd: process.env.HOME || "/tmp",
      source: "fallback",
      error: "Failed to get working directory from backend",
    })
  })
})
