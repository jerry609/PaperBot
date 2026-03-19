import { beforeEach, describe, expect, it, vi } from "vitest"

const { apiBaseUrlMock, proxyStreamMock } = vi.hoisted(() => ({
  apiBaseUrlMock: vi.fn(() => "http://backend.example.com"),
  proxyStreamMock: vi.fn(),
}))

vi.mock("@/app/api/_utils/backend-proxy", () => ({
  apiBaseUrl: apiBaseUrlMock,
  proxyStream: proxyStreamMock,
}))

import { POST as genCodePost } from "@/app/api/gen-code/route"
import { POST as paperscoolAnalyzePost } from "@/app/api/research/paperscool/analyze/route"
import { POST as paperscoolDailyPost } from "@/app/api/research/paperscool/daily/route"
import { POST as studioChatPost } from "@/app/api/studio/chat/route"

type StreamRouteCase = {
  expectedOptions: Record<string, unknown>
  handler: (req: Request) => Promise<Response>
  name: string
  path: string
}

const streamRouteCases: StreamRouteCase[] = [
  {
    name: "studio chat",
    handler: studioChatPost,
    path: "/api/studio/chat",
    expectedOptions: {
      auth: true,
      onError: expect.any(Function),
      responseContentType: "text/event-stream",
    },
  },
  {
    name: "gen code",
    handler: genCodePost,
    path: "/api/gen-code",
    expectedOptions: {
      auth: true,
      responseContentType: "text/event-stream",
    },
  },
  {
    name: "paperscool analyze",
    handler: paperscoolAnalyzePost,
    path: "/api/research/paperscool/analyze",
    expectedOptions: {
      accept: "text/event-stream",
      responseContentType: "text/event-stream",
      onError: expect.any(Function),
    },
  },
]

describe("stream route contracts", () => {
  beforeEach(() => {
    vi.resetAllMocks()
    apiBaseUrlMock.mockReturnValue("http://backend.example.com")
  })

  it.each(streamRouteCases)("proxies $name through the shared stream helper", async ({
    expectedOptions,
    handler,
    path,
  }) => {
    const req = new Request(`http://localhost${path}`, { method: "POST" })
    proxyStreamMock.mockResolvedValueOnce(new Response("data: ping\n\n"))

    await handler(req)

    expect(proxyStreamMock).toHaveBeenCalledWith(
      req,
      `http://backend.example.com${path}`,
      "POST",
      expectedOptions,
    )
  })

  it("preserves the daily stream/json passthrough fallback contract", async () => {
    const req = new Request("http://localhost/api/research/paperscool/daily", {
      method: "POST",
    })
    proxyStreamMock.mockResolvedValueOnce(new Response("data: ping\n\n"))

    await paperscoolDailyPost(req)

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

    const calls = vi.mocked(proxyStreamMock).mock.calls as unknown[][]
    const options = calls[0]?.[3] as
      | { onError?: (context: { error: unknown; isTimeout: boolean; upstreamUrl: string }) => Response }
      | undefined
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

  it("returns a Studio-specific chat fallback when the backend is offline", async () => {
    const req = new Request("http://localhost/api/studio/chat", {
      method: "POST",
    })
    proxyStreamMock.mockResolvedValueOnce(new Response("data: ping\n\n"))

    await studioChatPost(req)

    const calls = vi.mocked(proxyStreamMock).mock.calls as unknown[][]
    const options = calls[0]?.[3] as
      | { onError?: (context: { error: unknown; isTimeout: boolean; upstreamUrl: string }) => Response }
      | undefined
    const fallback = options?.onError?.({
      error: new Error("offline"),
      isTimeout: false,
      upstreamUrl: "http://backend.example.com/api/studio/chat",
    })

    expect(fallback).toBeInstanceOf(Response)
    expect(fallback?.status).toBe(502)
    await expect(fallback?.json()).resolves.toEqual({
      detail: "Studio backend is unreachable (http://backend.example.com/api/studio/chat): offline",
    })
  })
})
