import { describe, expect, it, vi } from "vitest"

const { apiBaseUrlMock, proxyStreamMock } = vi.hoisted(() => ({
  apiBaseUrlMock: vi.fn(() => "http://backend.example.com"),
  proxyStreamMock: vi.fn(),
}))

vi.mock("@/app/api/_utils/backend-proxy", () => ({
  apiBaseUrl: apiBaseUrlMock,
  proxyStream: proxyStreamMock,
}))

import { POST } from "./route"

describe("paperscool daily route", () => {
  it("proxies daily generation with stream/json passthrough options", async () => {
    const req = new Request("http://localhost/api/research/paperscool/daily", {
      method: "POST",
    })
    proxyStreamMock.mockResolvedValueOnce(new Response("data: ping\n\n"))

    await POST(req)

    expect(proxyStreamMock).toHaveBeenCalledWith(
      req,
      "http://backend.example.com/api/research/paperscool/daily",
      "POST",
      expect.objectContaining({
        accept: "text/event-stream, application/json",
        auth: true,
        passthroughNonStreamResponse: true,
        responseContentType: "text/event-stream",
        onError: expect.any(Function),
      }),
    )

    const options = vi.mocked(proxyStreamMock).mock.calls[0][3]
    const fallback = options?.onError?.({
      error: new Error("offline"),
      isTimeout: false,
      upstreamUrl: "http://backend.example.com/api/research/paperscool/daily",
    })

    expect(fallback).toBeInstanceOf(Response)
    expect(fallback?.status).toBe(502)
    await expect(fallback?.json()).resolves.toEqual({
      detail: "Upstream API unreachable",
      error: "offline",
    })
  })
})
