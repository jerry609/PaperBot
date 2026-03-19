import { describe, expect, it } from "vitest"

import {
  normalizeStudioTransportError,
  presentStudioError,
  readStudioErrorDetail,
} from "./studio-errors"

describe("studio-errors", () => {
  it("extracts detail from JSON error payloads", async () => {
    const res = new Response(JSON.stringify({ detail: "Studio backend is unreachable" }), {
      status: 502,
      headers: { "Content-Type": "application/json" },
    })

    await expect(readStudioErrorDetail(res, "fallback")).resolves.toBe(
      "Studio backend is unreachable",
    )
  })

  it("normalizes transport disconnect errors", () => {
    expect(normalizeStudioTransportError("Failed to fetch")).toBe(
      "Studio backend is unreachable. Check that the backend is running and retry.",
    )
    expect(normalizeStudioTransportError("Upstream API unreachable: http://backend")).toBe(
      "Studio backend is unreachable. Check that the backend is running and retry.",
    )
  })

  it("normalizes timeout errors", () => {
    expect(normalizeStudioTransportError("Studio backend timed out (http://backend)")).toBe(
      "Studio backend timed out. Check that the backend is healthy and retry.",
    )
  })

  it("preserves non-transport errors", () => {
    expect(normalizeStudioTransportError("project_dir is not allowed")).toBeNull()
    expect(presentStudioError("project_dir is not allowed", "fallback")).toBe(
      "project_dir is not allowed",
    )
  })
})
