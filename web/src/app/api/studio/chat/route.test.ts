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

describe("studio chat route", () => {
  it("proxies studio chat through the shared stream helper", async () => {
    const req = new Request("http://localhost/api/studio/chat", { method: "POST" })
    proxyStreamMock.mockResolvedValueOnce(new Response("data: ping\n\n"))

    await POST(req)

    expect(proxyStreamMock).toHaveBeenCalledWith(
      req,
      "http://backend.example.com/api/studio/chat",
      "POST",
      {
        auth: true,
        responseContentType: "text/event-stream",
      },
    )
  })
})
