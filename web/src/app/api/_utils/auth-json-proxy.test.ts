import { describe, expect, it, vi } from "vitest"

const { sharedApiBaseUrlMock, sharedProxyJsonMock } = vi.hoisted(() => ({
  sharedApiBaseUrlMock: vi.fn(() => "https://backend.test"),
  sharedProxyJsonMock: vi.fn(),
}))

vi.mock("./backend-proxy", () => ({
  apiBaseUrl: sharedApiBaseUrlMock,
  proxyJson: sharedProxyJsonMock,
}))

import { apiBaseUrl, proxyJson } from "./auth-json-proxy"

describe("auth json proxy", () => {
  it("delegates backend base URL resolution to the shared helper", () => {
    expect(apiBaseUrl()).toBe("https://backend.test")
    expect(sharedApiBaseUrlMock).toHaveBeenCalledTimes(1)
  })

  it("forces backend auth when delegating JSON proxies", async () => {
    const req = new Request("https://localhost/api/research/tracks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: "demo" }),
    })
    const response = Response.json({ ok: true }, { status: 201 })
    sharedProxyJsonMock.mockResolvedValueOnce(response)

    const res = await proxyJson(req, "https://backend/api/research/tracks", "POST")

    expect(sharedProxyJsonMock).toHaveBeenCalledWith(
      req,
      "https://backend/api/research/tracks",
      "POST",
      { auth: true },
    )
    expect(res).toBe(response)
  })
})
