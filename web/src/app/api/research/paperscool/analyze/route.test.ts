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

describe("paperscool analyze route", () => {
  it("proxies analyze requests through the shared stream helper", async () => {
    const req = new Request("http://localhost/api/research/paperscool/analyze", {
      method: "POST",
    })
    proxyStreamMock.mockResolvedValueOnce(new Response("data: ping\n\n"))

    await POST(req)

    expect(proxyStreamMock).toHaveBeenCalledWith(
      req,
      "http://backend.example.com/api/research/paperscool/analyze",
      "POST",
      expect.objectContaining({
        accept: "text/event-stream",
        responseContentType: "text/event-stream",
        onError: expect.any(Function),
      }),
    )
  })
})
