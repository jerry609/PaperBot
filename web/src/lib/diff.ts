export type DiffLine = {
  type: "add" | "remove" | "unchanged"
  content: string
  lineNumber: { old?: number; new?: number }
}

function computeLCS(a: string[], b: string[]): string[] {
  const m = a.length
  const n = b.length
  const dp: number[][] = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0))

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (a[i - 1] === b[j - 1]) dp[i][j] = dp[i - 1][j - 1] + 1
      else dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1])
    }
  }

  const lcs: string[] = []
  let i = m
  let j = n
  while (i > 0 && j > 0) {
    if (a[i - 1] === b[j - 1]) {
      lcs.unshift(a[i - 1])
      i--
      j--
    } else if (dp[i - 1][j] > dp[i][j - 1]) i--
    else j--
  }
  return lcs
}

export function computeDiff(oldLines: string[], newLines: string[]): DiffLine[] {
  const result: DiffLine[] = []
  const lcs = computeLCS(oldLines, newLines)

  let oldIdx = 0
  let newIdx = 0
  let lcsIdx = 0

  while (oldIdx < oldLines.length || newIdx < newLines.length) {
    if (lcsIdx < lcs.length && oldIdx < oldLines.length && oldLines[oldIdx] === lcs[lcsIdx]) {
      if (newIdx < newLines.length && newLines[newIdx] === lcs[lcsIdx]) {
        result.push({
          type: "unchanged",
          content: oldLines[oldIdx],
          lineNumber: { old: oldIdx + 1, new: newIdx + 1 },
        })
        oldIdx++
        newIdx++
        lcsIdx++
      } else {
        result.push({
          type: "add",
          content: newLines[newIdx],
          lineNumber: { new: newIdx + 1 },
        })
        newIdx++
      }
    } else if (oldIdx < oldLines.length) {
      result.push({
        type: "remove",
        content: oldLines[oldIdx],
        lineNumber: { old: oldIdx + 1 },
      })
      oldIdx++
    } else if (newIdx < newLines.length) {
      result.push({
        type: "add",
        content: newLines[newIdx],
        lineNumber: { new: newIdx + 1 },
      })
      newIdx++
    }
  }

  return result
}

export type DiffHunk = {
  id: string
  before: string
  after: string
  old: string
  new: string
  stats: { added: number; removed: number }
}

export function computeHunks(oldValue: string, newValue: string, contextLines = 2): DiffHunk[] {
  const oldLines = oldValue.split("\n")
  const newLines = newValue.split("\n")
  const diff = computeDiff(oldLines, newLines)

  const changeIndexes = diff
    .map((d, i) => (d.type === "unchanged" ? -1 : i))
    .filter((i) => i !== -1)

  if (changeIndexes.length === 0) return []

  // Group into hunks based on gaps of unchanged lines.
  const hunks: Array<{ start: number; end: number }> = []
  let start = changeIndexes[0]
  let end = changeIndexes[0]

  for (let k = 1; k < changeIndexes.length; k++) {
    const idx = changeIndexes[k]
    if (idx - end <= contextLines * 2 + 1) {
      end = idx
    } else {
      hunks.push({ start, end })
      start = idx
      end = idx
    }
  }
  hunks.push({ start, end })

  return hunks.map((h, hunkIndex) => {
    const beforeStart = Math.max(0, h.start - contextLines)
    const afterEnd = Math.min(diff.length - 1, h.end + contextLines)

    const beforeLines = diff
      .slice(beforeStart, h.start)
      .filter((l) => l.type === "unchanged")
      .map((l) => l.content)

    const afterLines = diff
      .slice(h.end + 1, afterEnd + 1)
      .filter((l) => l.type === "unchanged")
      .map((l) => l.content)

    const core = diff.slice(h.start, h.end + 1)
    const oldCore = core.filter((l) => l.type !== "add").map((l) => l.content)
    const newCore = core.filter((l) => l.type !== "remove").map((l) => l.content)

    const stats = core.reduce(
      (acc, l) => {
        if (l.type === "add") acc.added += 1
        if (l.type === "remove") acc.removed += 1
        return acc
      },
      { added: 0, removed: 0 }
    )

    const id = `hunk-${hunkIndex + 1}`
    return {
      id,
      before: beforeLines.join("\n"),
      after: afterLines.join("\n"),
      old: oldCore.join("\n"),
      new: newCore.join("\n"),
      stats,
    }
  })
}

