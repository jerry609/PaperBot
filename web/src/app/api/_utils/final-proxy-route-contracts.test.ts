import { describe, expect, it, vi } from "vitest"

const { apiBaseUrlMock, proxyBinaryMock, proxyTextMock } = vi.hoisted(() => ({
  apiBaseUrlMock: vi.fn(() => "http://backend.example.com"),
  proxyBinaryMock: vi.fn(),
  proxyTextMock: vi.fn(),
}))

vi.mock("@/app/api/_utils/backend-proxy", () => ({
  apiBaseUrl: apiBaseUrlMock,
  proxyBinary: proxyBinaryMock,
  proxyText: proxyTextMock,
}))

import { GET as newsletterUnsubscribeGet } from "@/app/api/newsletter/unsubscribe/[token]/route"
import { GET as papersExportGet } from "@/app/api/papers/export/route"

describe("final proxy route contracts", () => {
  it("proxies protected exports through the shared binary helper", async () => {
    proxyBinaryMock.mockResolvedValueOnce(new Response("bibtex-body"))

    const res = await papersExportGet(
      new Request("http://localhost/api/papers/export?format=bibtex", {
        method: "GET",
      }),
    )

    expect(proxyBinaryMock).toHaveBeenCalledWith(
      expect.any(Request),
      "http://backend.example.com/api/research/papers/export?format=bibtex",
      "GET",
      {
        accept: "*/*",
        auth: true,
        onError: expect.any(Function),
      },
    )
    expect(res).toBeInstanceOf(Response)
    expect(await res.text()).toBe("bibtex-body")

    const binaryCalls = vi.mocked(proxyBinaryMock).mock.calls as unknown[][]
    const binaryOptions = binaryCalls[0]?.[3] as
      | { onError?: (context: { error: unknown; isTimeout: boolean; upstreamUrl: string }) => Response }
      | undefined
    const fallback = binaryOptions?.onError?.({
      error: new Error("offline"),
      isTimeout: false,
      upstreamUrl: "http://backend.example.com/api/research/papers/export?format=bibtex",
    })

    expect(fallback).toBeInstanceOf(Response)
    expect(fallback?.status).toBe(502)
    await expect(fallback?.json()).resolves.toEqual({
      detail: "Upstream API unreachable",
      error: "offline",
    })
  })

  it("proxies newsletter unsubscribe pages through the shared text helper", async () => {
    const req = new Request("http://localhost/api/newsletter/unsubscribe/token")
    proxyTextMock.mockResolvedValueOnce(
      new Response("<html><body>ok</body></html>", {
        headers: { "Content-Type": "text/html" },
      }),
    )

    const res = await newsletterUnsubscribeGet(req, {
      params: Promise.resolve({ token: "token\"&<>'" }),
    })

    expect(proxyTextMock).toHaveBeenCalledWith(
      req,
      "http://backend.example.com/api/newsletter/unsubscribe/token%22%26%3C%3E'",
      "GET",
      expect.objectContaining({
        accept: "text/html",
        responseContentType: "text/html",
        onError: expect.any(Function),
      }),
    )
    expect(res).toBeInstanceOf(Response)
    expect(await res.text()).toContain("ok")

    const textCalls = vi.mocked(proxyTextMock).mock.calls as unknown[][]
    const textOptions = textCalls[0]?.[3] as
      | { onError?: (context: { error: unknown; isTimeout: boolean; upstreamUrl: string }) => Response }
      | undefined
    const fallback = textOptions?.onError?.({
      error: new Error("bad 'html' \"tag\" <tag> & stuff"),
      isTimeout: false,
      upstreamUrl: "http://backend.example.com/api/newsletter/unsubscribe/token",
    })

    expect(fallback).toBeInstanceOf(Response)
    expect(fallback?.status).toBe(502)
    expect(fallback?.headers.get("content-type")).toContain("text/html")
    await expect(fallback?.text()).resolves.toContain(
      "bad &#39;html&#39; &quot;tag&quot; &lt;tag&gt; &amp; stuff",
    )
  })
})
