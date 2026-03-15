import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const { apiBaseUrlMock, proxyBinaryMock } = vi.hoisted(() => ({
  apiBaseUrlMock: vi.fn(() => "http://backend.example.com"),
  proxyBinaryMock: vi.fn(),
}))

vi.mock("@/app/api/_utils/backend-proxy", () => ({
  apiBaseUrl: apiBaseUrlMock,
  proxyBinary: proxyBinaryMock,
}))

import { GET } from "./route"

describe("papers export proxy", () => {
  beforeEach(() => {
    vi.resetAllMocks()
    apiBaseUrlMock.mockReturnValue("http://backend.example.com")
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it("proxies protected exports through the shared binary helper", async () => {
    proxyBinaryMock.mockResolvedValueOnce(new Response("bibtex-body"))

    const res = await GET(
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

    const calls = vi.mocked(proxyBinaryMock).mock.calls as unknown[][]
    const options = calls[0]?.[3] as
      | { onError?: (context: { error: unknown; isTimeout: boolean; upstreamUrl: string }) => Response }
      | undefined
    const fallback = options?.onError?.({
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
})
