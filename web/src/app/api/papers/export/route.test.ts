import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const { withBackendAuthMock } = vi.hoisted(() => ({
  withBackendAuthMock: vi.fn(),
}))

vi.mock("@/app/api/_utils/auth-headers", () => ({
  withBackendAuth: withBackendAuthMock,
}))

vi.mock("@/app/api/research/_base", () => ({
  apiBaseUrl: () => "http://backend.example.com",
}))

import { GET } from "./route"

describe("papers export proxy", () => {
  const originalFetch = global.fetch

  beforeEach(() => {
    vi.resetAllMocks()
  })

  afterEach(() => {
    global.fetch = originalFetch
  })

  it("forwards backend auth headers to the protected export endpoint", async () => {
    withBackendAuthMock.mockResolvedValue({
      Accept: "*/*",
      authorization: "Bearer export-token",
    })

    const fetchMock = vi.fn(async () =>
      new Response("bibtex-body", {
        status: 200,
        headers: {
          "content-type": "application/x-bibtex",
          "content-disposition": "attachment; filename=papers.bib",
        },
      }),
    )
    global.fetch = fetchMock as typeof fetch

    const res = await GET(
      new Request("http://localhost/api/papers/export?format=bibtex", {
        method: "GET",
      }),
    )

    expect(withBackendAuthMock).toHaveBeenCalledTimes(1)
    expect(fetchMock).toHaveBeenCalledWith(
      "http://backend.example.com/api/research/papers/export?format=bibtex",
      {
        method: "GET",
        headers: {
          Accept: "*/*",
          authorization: "Bearer export-token",
        },
      },
    )
    expect(res.status).toBe(200)
    expect(await res.text()).toBe("bibtex-body")
    expect(res.headers.get("Content-Type")).toBe("application/x-bibtex")
    expect(res.headers.get("Content-Disposition")).toBe("attachment; filename=papers.bib")
  })
})
