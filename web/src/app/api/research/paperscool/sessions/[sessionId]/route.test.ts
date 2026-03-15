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

describe("paperscool session route", () => {
  it("proxies the backend request without caching", async () => {
    const req = new Request("http://localhost/api/research/paperscool/sessions/some/session")
    proxyJsonMock.mockResolvedValueOnce(Response.json({ ok: true }))

    await GET(req, { params: Promise.resolve({ sessionId: "session/42" }) })

    expect(proxyJsonMock).toHaveBeenCalledWith(
      req,
      "http://backend.example.com/api/research/paperscool/sessions/session%2F42",
      "GET",
      { cache: "no-store" },
    )
  })
})
