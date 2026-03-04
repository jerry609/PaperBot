import { describe, expect, it } from "vitest"
import { renderToStaticMarkup } from "react-dom/server"
import { PaperIcon, getAbbreviation } from "./PaperIcon"

describe("PaperIcon", () => {
  it("handles abbreviation edge cases", () => {
    expect(getAbbreviation("")).toBe("??")
    expect(getAbbreviation("the and of")).toBe("??")
    expect(getAbbreviation("深度 学习")).toBe("深学")
  })

  it("renders deterministic SVG for the same input", () => {
    const first = renderToStaticMarkup(
      <PaperIcon paperId="paper-fixed-id" title="Graph Attention Networks" size={48} />
    )
    const second = renderToStaticMarkup(
      <PaperIcon paperId="paper-fixed-id" title="Graph Attention Networks" size={48} />
    )

    expect(first).toBe(second)
    expect(first).toContain('aria-label="GA"')
  })

  it("falls back to safe icon size for invalid input", () => {
    const svg = renderToStaticMarkup(
      <PaperIcon paperId="paper-invalid-size" title="Test" size={Number.NaN} />
    )
    expect(svg).toContain('width="48"')
    expect(svg).toContain('height="48"')
    expect(svg).toContain('viewBox="0 0 48 48"')
  })
})
