import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const { withBackendAuthMock } = vi.hoisted(() => ({
  withBackendAuthMock: vi.fn(),
}))

vi.mock("./auth-headers", () => ({
  withBackendAuth: withBackendAuthMock,
}))

import { apiBaseUrl, proxyText } from "./backend-proxy"

describe("apiBaseUrl", () => {
  const originalPaperbotApiBaseUrl = process.env.PAPERBOT_API_BASE_URL

  afterEach(() => {
    if (originalPaperbotApiBaseUrl === undefined) {
      delete process.env.PAPERBOT_API_BASE_URL
    } else {
      process.env.PAPERBOT_API_BASE_URL = originalPaperbotApiBaseUrl
    }
  })

  it("uses PAPERBOT_API_BASE_URL when present", () => {
    process.env.PAPERBOT_API_BASE_URL = "http://paperbot-api"
    expect(apiBaseUrl()).toBe("http://paperbot-api")
  })

  it("falls back to localhost", () => {
    delete process.env.PAPERBOT_API_BASE_URL
    expect(apiBaseUrl()).toBe("http://127.0.0.1:8000")
  })
})

describe("proxyText", () => {
  const originalFetch = global.fetch

  beforeEach(() => {
    vi.resetAllMocks()
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

    const req = new Request("http://localhost/api/runbook/files?path=demo", { method: "GET" })
    const res = await proxyText(req, "http://backend/api/runbook/files?path=demo", "GET")

    expect(fetchMock).toHaveBeenCalledWith("http://backend/api/runbook/files?path=demo", {
      method: "GET",
      headers: { Accept: "application/json" },
      body: undefined,
    })
    expect(await res.text()).toBe(JSON.stringify({ ok: true }))
    expect(res.status).toBe(200)
  })

  it("adds backend auth headers for protected writes", async () => {
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

    const req = new Request("http://localhost/api/runbook/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: "/tmp/demo" }),
    })
    const res = await proxyText(req, "http://backend/api/runbook/delete", "POST", { auth: true })

    expect(withBackendAuthMock).toHaveBeenCalledTimes(1)
    expect(fetchMock).toHaveBeenCalledWith("http://backend/api/runbook/delete", {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
        authorization: "Bearer test-token",
      },
      body: JSON.stringify({ path: "/tmp/demo" }),
    })
    expect(res.status).toBe(202)
  })
})
