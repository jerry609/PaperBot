import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import {
  apiBaseUrl,
  proxyBinary,
  proxyJson,
  proxyStream,
  proxyText,
} from "./backend-proxy"
import { withBackendAuth } from "./auth-headers"

vi.mock("./auth-headers", () => ({
  backendBaseUrl: () => "http://backend.test",
  withBackendAuth: vi.fn(async (_req: Request, base: HeadersInit = {}) => {
    const headers = new Headers(base)
    headers.set("authorization", "Bearer test-token")
    return headers
  }),
}))

describe("backend-proxy", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn())
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.clearAllMocks()
  })

  it("uses the shared backend base URL", () => {
    expect(apiBaseUrl()).toBe("http://backend.test")
  })

  it("proxies JSON with backend auth when requested", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      new Response(JSON.stringify({ ok: true }), {
        status: 201,
        headers: { "content-type": "application/json" },
      }),
    )

    const req = new Request("http://app.test/api/auth/me", {
      body: JSON.stringify({ display_name: "PaperBot" }),
      headers: { "content-type": "application/json" },
      method: "PATCH",
    })

    const res = await proxyJson(req, "http://backend.test/api/auth/me", "PATCH", {
      auth: true,
    })

    expect(withBackendAuth).toHaveBeenCalledTimes(1)
    expect(fetch).toHaveBeenCalledTimes(1)
    const [, init] = vi.mocked(fetch).mock.calls[0]
    const headers = new Headers((init as RequestInit).headers)
    expect(headers.get("authorization")).toBe("Bearer test-token")
    expect(await res.json()).toEqual({ ok: true })
  })

  it("supports custom error handlers for text responses", async () => {
    vi.mocked(fetch).mockRejectedValueOnce(new Error("boom"))

    const res = await proxyText(
      new Request("http://app.test/api/newsletter/unsubscribe/token"),
      "http://backend.test/api/newsletter/unsubscribe/token",
      "GET",
      {
        onError: () => new Response("<html>fallback</html>", {
          headers: { "Content-Type": "text/html" },
          status: 502,
        }),
        responseContentType: "text/html",
      },
    )

    expect(res.status).toBe(502)
    expect(res.headers.get("content-type")).toContain("text/html")
    expect(await res.text()).toContain("fallback")
  })

  it("preserves binary download headers", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      new Response(new Uint8Array([1, 2, 3]), {
        headers: {
          "content-disposition": 'attachment; filename="papers.csv"',
          "content-type": "text/csv",
        },
        status: 200,
      }),
    )

    const res = await proxyBinary(
      new Request("http://app.test/api/papers/export"),
      "http://backend.test/api/research/papers/export",
      "GET",
      { accept: "*/*" },
    )

    expect(res.headers.get("content-disposition")).toContain("papers.csv")
    expect(res.headers.get("content-type")).toContain("text/csv")
  })

  it("passes through SSE streams and JSON fallbacks", async () => {
    vi.mocked(fetch)
      .mockResolvedValueOnce(
        new Response("data: ping\n\n", {
          headers: { "content-type": "text/event-stream" },
          status: 200,
        }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ done: true }), {
          headers: { "content-type": "application/json" },
          status: 200,
        }),
      )

    const streamReq = new Request("http://app.test/api/research/paperscool/daily", {
      body: JSON.stringify({}),
      headers: { "content-type": "application/json" },
      method: "POST",
    })
    const streamRes = await proxyStream(
      streamReq,
      "http://backend.test/api/research/paperscool/daily",
      "POST",
      {
        auth: true,
        passthroughNonStreamResponse: true,
        responseContentType: "text/event-stream",
      },
    )
    expect(streamRes.headers.get("content-type")).toContain("text/event-stream")
    expect(await streamRes.text()).toContain("data: ping")

    const jsonReq = new Request("http://app.test/api/research/paperscool/daily", {
      body: JSON.stringify({}),
      headers: { "content-type": "application/json" },
      method: "POST",
    })
    const jsonRes = await proxyStream(
      jsonReq,
      "http://backend.test/api/research/paperscool/daily",
      "POST",
      {
        auth: true,
        passthroughNonStreamResponse: true,
        responseContentType: "text/event-stream",
      },
    )
    expect(jsonRes.headers.get("content-type")).toContain("application/json")
    expect(await jsonRes.json()).toEqual({ done: true })
  })
})
