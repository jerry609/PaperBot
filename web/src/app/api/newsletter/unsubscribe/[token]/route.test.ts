import { describe, expect, it, vi } from "vitest"

const { apiBaseUrlMock, proxyTextMock } = vi.hoisted(() => ({
  apiBaseUrlMock: vi.fn(() => "http://backend.example.com"),
  proxyTextMock: vi.fn(),
}))

vi.mock("@/app/api/_utils/backend-proxy", () => ({
  apiBaseUrl: apiBaseUrlMock,
  proxyText: proxyTextMock,
}))

import { GET } from "./route"

describe("newsletter unsubscribe route", () => {
  it("proxies the unsubscribe page through the shared text helper", async () => {
    const req = new Request("http://localhost/api/newsletter/unsubscribe/token")
    proxyTextMock.mockResolvedValueOnce(
      new Response("<html><body>ok</body></html>", {
        headers: { "Content-Type": "text/html" },
      }),
    )

    const res = await GET(req, { params: Promise.resolve({ token: "token\"&<>" }) })

    expect(proxyTextMock).toHaveBeenCalledWith(
      req,
      "http://backend.example.com/api/newsletter/unsubscribe/token%22%26%3C%3E",
      "GET",
      expect.objectContaining({
        responseContentType: "text/html",
        onError: expect.any(Function),
      }),
    )
    expect(res).toBeInstanceOf(Response)
    expect(await res.text()).toContain("ok")

    const calls = vi.mocked(proxyTextMock).mock.calls as unknown[][]
    const options = calls[0]?.[3] as
      | { onError?: (context: { error: unknown; isTimeout: boolean; upstreamUrl: string }) => Response }
      | undefined
    const fallback = options?.onError?.({
      error: new Error("bad \"html\" <tag> & stuff"),
      isTimeout: false,
      upstreamUrl: "http://backend.example.com/api/newsletter/unsubscribe/token",
    })

    expect(fallback).toBeInstanceOf(Response)
    expect(fallback?.status).toBe(502)
    expect(fallback?.headers.get("content-type")).toContain("text/html")
    await expect(fallback?.text()).resolves.toContain(
      "bad &quot;html&quot; &lt;tag&gt; &amp; stuff",
    )
  })
})
