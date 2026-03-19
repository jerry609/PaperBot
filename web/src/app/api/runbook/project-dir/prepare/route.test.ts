import fs from "node:fs/promises"
import os from "node:os"
import path from "node:path"

import { afterEach, describe, expect, it, vi } from "vitest"

const { apiBaseUrlMock, withBackendAuthMock } = vi.hoisted(() => ({
  apiBaseUrlMock: vi.fn(() => "http://backend.example.com"),
  withBackendAuthMock: vi.fn(async (_req: Request, headers: HeadersInit = {}) => headers),
}))

vi.mock("@/app/api/_utils/backend-proxy", () => ({
  apiBaseUrl: apiBaseUrlMock,
}))

vi.mock("@/app/api/_utils/auth-headers", () => ({
  withBackendAuth: withBackendAuthMock,
}))

import { POST } from "./route"

describe("runbook project-dir prepare route", () => {
  afterEach(async () => {
    vi.restoreAllMocks()
  })

  it("falls back to local preparation when backend rejects an allowed directory", async () => {
    const originalHome = process.env.HOME
    const homeDir = path.join(os.tmpdir(), `paperbot-home-${Date.now()}`)
    const targetDir = path.join(homeDir, "Documents", "paperbot-route-test")

    process.env.HOME = homeDir
    vi.spyOn(global, "fetch").mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "project_dir is not allowed" }), {
        status: 403,
        headers: { "Content-Type": "application/json" },
      }),
    )

    try {
      const req = new Request("http://localhost/api/runbook/project-dir/prepare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project_dir: targetDir,
          create_if_missing: true,
        }),
      })

      const res = await POST(req)
      const payload = await res.json()

      expect(res.status).toBe(200)
      expect(payload.project_dir).toBe(targetDir)
      expect(payload.created).toBe(true)
      await expect(fs.stat(targetDir)).resolves.toBeDefined()
    } finally {
      process.env.HOME = originalHome
      await fs.rm(homeDir, { recursive: true, force: true })
    }
  })
})
