import { describe, expect, it } from "vitest"

import { safeHref, safeInternalHref } from "./utils"

describe("safeHref", () => {
  it("accepts http, https, and app-relative links", () => {
    expect(safeHref("https://example.com/paper")).toBe("https://example.com/paper")
    expect(safeHref("http://example.com/paper")).toBe("http://example.com/paper")
    expect(safeHref("/papers/123")).toBe("/papers/123")
  })

  it("rejects unsafe schemes and protocol-relative links", () => {
    expect(safeHref("javascript:alert(1)")).toBeNull()
    expect(safeHref("data:text/html;base64,SGk=")).toBeNull()
    expect(safeHref("//example.com/paper")).toBeNull()
  })
})

describe("safeInternalHref", () => {
  it("only allows root-relative app links", () => {
    expect(safeInternalHref("/research?track_id=7")).toBe("/research?track_id=7")
    expect(safeInternalHref("#queue")).toBe("#queue")
    expect(safeInternalHref("https://example.com/paper")).toBeNull()
    expect(safeInternalHref("papers/123")).toBeNull()
  })
})
