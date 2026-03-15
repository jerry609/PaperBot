import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const { backendBaseUrlMock, withBackendAuthMock } = vi.hoisted(() => ({
  backendBaseUrlMock: vi.fn(() => "https://backend.test"),
  withBackendAuthMock: vi.fn(),
}))

vi.mock("./auth-headers", () => ({
  backendBaseUrl: backendBaseUrlMock,
  withBackendAuth: withBackendAuthMock,
}))

import { apiBaseUrl, proxyJson, proxyText } from "./backend-proxy"

describe("apiBaseUrl", () => {
  it("uses the shared backend base URL helper", () => {
    expect(apiBaseUrl()).toBe("https://backend.test")
    expect(backendBaseUrlMock).toHaveBeenCalledTimes(1)
  })
})

describe("backend proxy helpers", () => {
  const originalFetch = global.fetch

  beforeEach(() => {
    vi.resetAllMocks()
    backendBaseUrlMock.mockReturnValue("https://backend.test")
  })

  afterEach(() => {
    global.fetch = originalFetch
  })

  it("proxies GET requests without reading a request body", async () => {
    const fetchMock = vi.fn(async () =>
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    )
    global.fetch = fetchMock as typeof fetch

    const req = new Request("https://localhost/api/runbook/files?path=demo", { method: "GET" })
    const res = await proxyText(req, "https://backend/api/runbook/files?path=demo", "GET")

    expect(fetchMock).toHaveBeenCalledWith("https://backend/api/runbook/files?path=demo", {
      method: "GET",
      headers: { Accept: "application/json" },
      body: undefined,
      signal: expect.any(AbortSignal),
    })
    expect(await res.text()).toBe(JSON.stringify({ ok: true }))
    expect(res.status).toBe(200)
  })

  it("proxies JSON writes with backend auth when requested", async () => {
    withBackendAuthMock.mockResolvedValue({
      Accept: "application/json",
      "Content-Type": "application/json",
      authorization: "Bearer test-token",
    })

    const fetchMock = vi.fn(async () =>
      new Response(JSON.stringify({ ok: true }), {
        status: 202,
        headers: { "Content-Type": "application/json" },
      }),
    )
    global.fetch = fetchMock as typeof fetch

    const req = new Request("https://localhost/api/runbook/delete", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ display_name: "PaperBot" }),
    })
    const res = await proxyJson(req, "https://backend/api/auth/me", "PATCH", {
      auth: true,
    })

    expect(withBackendAuthMock).toHaveBeenCalledTimes(1)
    expect(fetchMock).toHaveBeenCalledWith("https://backend/api/auth/me", {
      method: "PATCH",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
        authorization: "Bearer test-token",
      },
      body: JSON.stringify({ display_name: "PaperBot" }),
      signal: expect.any(AbortSignal),
    })
    expect(res.status).toBe(202)
  })

  it("returns an empty response when the upstream returns 204", async () => {
    withBackendAuthMock.mockResolvedValue({
      Accept: "application/json",
      authorization: "Bearer test-token",
    })

    const fetchMock = vi.fn(async () => new Response(null, { status: 204 }))
    global.fetch = fetchMock as typeof fetch

    const req = new Request("https://localhost/api/auth/me", { method: "DELETE" })
    const res = await proxyJson(req, "https://backend/api/auth/me", "DELETE", {
      auth: true,
    })

    expect(fetchMock).toHaveBeenCalledWith("https://backend/api/auth/me", {
      method: "DELETE",
      headers: {
        Accept: "application/json",
        authorization: "Bearer test-token",
      },
      body: undefined,
      signal: expect.any(AbortSignal),
    })
    expect(res.status).toBe(204)
    expect(await res.text()).toBe("")
  })

  it("supports custom error handlers for upstream failures", async () => {
    const fetchMock = vi.fn(async () => {
      throw new Error("boom")
    })
    global.fetch = fetchMock as typeof fetch

    const res = await proxyJson(
      new Request("https://localhost/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: "paperbot@example.com" }),
      }),
      "https://backend/api/auth/login",
      "POST",
      {
        onError: () => Response.json({ detail: "Service unavailable" }, { status: 502 }),
      },
    )

    expect(res.status).toBe(502)
    await expect(res.json()).resolves.toEqual({ detail: "Service unavailable" })
  })
})
