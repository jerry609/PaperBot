import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const { authMock } = vi.hoisted(() => ({
  authMock: vi.fn(),
}))

vi.mock("@/auth", () => ({
  auth: authMock,
}))

import { backendBaseUrl, withBackendAuth } from "./auth-headers"

describe("backendBaseUrl", () => {
  const originalBackendBaseUrl = process.env.BACKEND_BASE_URL
  const originalPaperbotApiBaseUrl = process.env.PAPERBOT_API_BASE_URL

  afterEach(() => {
    if (originalBackendBaseUrl === undefined) {
      delete process.env.BACKEND_BASE_URL
    } else {
      process.env.BACKEND_BASE_URL = originalBackendBaseUrl
    }

    if (originalPaperbotApiBaseUrl === undefined) {
      delete process.env.PAPERBOT_API_BASE_URL
    } else {
      process.env.PAPERBOT_API_BASE_URL = originalPaperbotApiBaseUrl
    }
  })

  it("falls back to PAPERBOT_API_BASE_URL when BACKEND_BASE_URL is unset", () => {
    delete process.env.BACKEND_BASE_URL
    process.env.PAPERBOT_API_BASE_URL = "https://paperbot-api.example.com"

    expect(backendBaseUrl()).toBe("https://paperbot-api.example.com")
  })
})

describe("withBackendAuth", () => {
  beforeEach(() => {
    vi.resetAllMocks()
  })

  it("prefers the incoming authorization header", async () => {
    const headers = await withBackendAuth(
      new Request("http://localhost/api/demo", {
        headers: { authorization: "Bearer incoming-token" },
      }),
      { Accept: "application/json" },
    )

    expect(authMock).not.toHaveBeenCalled()
    expect(new Headers(headers).get("authorization")).toBe("Bearer incoming-token")
  })

  it("uses the server session access token when present", async () => {
    authMock.mockResolvedValue({ accessToken: "session-token" })

    const headers = await withBackendAuth(new Request("http://localhost/api/demo"), {
      Accept: "application/json",
    })

    expect(new Headers(headers).get("authorization")).toBe("Bearer session-token")
  })
})
