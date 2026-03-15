import { describe, expect, it, vi } from "vitest"

const { researchApiBaseUrlMock, proxyJsonMock } = vi.hoisted(() => ({
  researchApiBaseUrlMock: vi.fn(() => "http://backend.example.com"),
  proxyJsonMock: vi.fn(),
}))

const { proxyStreamMock } = vi.hoisted(() => ({
  proxyStreamMock: vi.fn(),
}))

vi.mock("../../_base", () => ({
  apiBaseUrl: researchApiBaseUrlMock,
  proxyJson: proxyJsonMock,
}))

vi.mock("@/app/api/_utils/backend-proxy", () => ({
  proxyStream: proxyStreamMock,
}))

import { GET, POST } from "./route"

describe("research repro context route", () => {
  it("keeps the GET proxy on the research base helper", async () => {
    const req = new Request("http://localhost/api/research/repro/context?paper_id=123")
    proxyJsonMock.mockResolvedValueOnce(Response.json({ ok: true }))

    await GET(req)

    expect(proxyJsonMock).toHaveBeenCalledWith(
      req,
      "http://backend.example.com/api/research/repro/context?paper_id=123",
      "GET",
    )
  })

  it("proxies context generation through the shared stream helper", async () => {
    const req = new Request("http://localhost/api/research/repro/context", {
      method: "POST",
    })
    proxyStreamMock.mockResolvedValueOnce(new Response("data: ping\n\n"))

    await POST(req)

    expect(proxyStreamMock).toHaveBeenCalledWith(
      req,
      "http://backend.example.com/api/research/repro/context/generate",
      "POST",
      {
        accept: "text/event-stream",
        auth: true,
        responseContentType: "text/event-stream",
      },
    )
  })
})
