import { promises as fs } from "node:fs"
import path from "node:path"

type DailyReportJudge = {
  overall?: number | null
  recommendation?: string | null
}

type DailyDigestCard = {
  highlight?: string
  method?: string
  finding?: string
  tags?: string[]
}

type DailyReportItem = {
  paper_id?: string
  title?: string
  url?: string
  external_url?: string
  pdf_url?: string
  score?: number
  snippet?: string
  subject_or_venue?: string
  venue?: string
  authors?: string[]
  matched_queries?: string[]
  matched_keywords?: string[]
  branches?: string[]
  sources?: string[]
  judge?: DailyReportJudge
  digest_card?: DailyDigestCard
}

type DailyReportQuery = {
  normalized_query?: string
  raw_query?: string
  top_items?: DailyReportItem[]
}

type DailyReportTrend = {
  query?: string
  analysis?: string
}

type DailyReport = {
  title?: string
  date?: string
  generated_at?: string
  source?: string
  sources?: string[]
  stats?: {
    unique_items?: number
    total_query_hits?: number
    query_count?: number
  }
  queries?: DailyReportQuery[]
  global_top?: DailyReportItem[]
  llm_analysis?: {
    query_trends?: DailyReportTrend[]
  }
}

export interface DashboardDailyBriefHighlight {
  id: string
  title: string
  href: string
  queryLabel: string
  venueLabel: string
  metricLabel: string
  summary: string
  sourceBadges: string[]
  tags: string[]
  recommendation?: string | null
}

export interface DashboardDailyBrief {
  title: string
  date: string
  generatedAt?: string | null
  sourceLabel: string
  sourceBadges: string[]
  stats: {
    uniqueItems: number
    totalQueryHits: number
    queryCount: number
  }
  highlights: DashboardDailyBriefHighlight[]
  queryPulse: Array<{
    query: string
    hits: number
  }>
  trendRows: Array<{
    query: string
    analysis: string
  }>
}

const SOURCE_LABELS: Record<string, string> = {
  papers_cool: "papers.cool",
  arxiv_api: "arXiv API",
  hf_daily: "HF Daily",
  arxiv: "arXiv",
  venue: "Venue",
}

function toSourceLabel(value: string): string {
  const normalized = value.trim()
  if (!normalized) return "DailyPaper"
  return SOURCE_LABELS[normalized] || normalized.replace(/_/g, " ")
}

function toRecommendationLabel(value?: string | null): string | null {
  const normalized = String(value || "").trim()
  if (!normalized) return null

  switch (normalized) {
    case "must_read":
      return "Must Read"
    case "worth_reading":
      return "Worth Reading"
    case "skim":
      return "Skim"
    case "skip":
      return "Skip"
    default:
      return normalized
  }
}

function pickHighlightHref(item: DailyReportItem): string {
  const candidates = [item.external_url, item.url, item.pdf_url]
  const firstUrl = candidates.find((value) => typeof value === "string" && value.trim())
  return firstUrl?.trim() || "/workflows"
}

function buildHighlightKey(item: DailyReportItem, queryLabel: string): string {
  return String(item.paper_id || item.url || item.title || queryLabel).trim()
}

function dedupeHighlightItems(
  items: Array<{ item: DailyReportItem; queryLabel: string }>,
): Array<{ item: DailyReportItem; queryLabel: string }> {
  const seen = new Set<string>()
  const deduped: Array<{ item: DailyReportItem; queryLabel: string }> = []

  for (const candidate of items) {
    const key = buildHighlightKey(candidate.item, candidate.queryLabel)
    if (!key || seen.has(key)) continue
    seen.add(key)
    deduped.push(candidate)
  }

  return deduped
}

function buildHighlightSummary(item: DailyReportItem): string {
  const digest = item.digest_card || {}
  const summary = String(digest.highlight || item.snippet || "").trim()
  if (summary) return summary

  const authors = (item.authors || []).filter(Boolean).slice(0, 3).join(", ")
  if (authors) return authors

  return "已进入今日简报候选池，建议优先判断是否值得继续深入。"
}

function buildHighlightMetric(item: DailyReportItem): string {
  if (item.judge?.overall != null) {
    return `Judge ${Number(item.judge.overall).toFixed(1)}`
  }

  if (item.score != null) {
    return `Score ${Number(item.score).toFixed(1)}`
  }

  return "Daily pick"
}

function collectHighlightBadges(item: DailyReportItem): string[] {
  const badges = [...(item.sources || []), ...(item.branches || [])]
    .map((value) => toSourceLabel(String(value || "")))
    .filter(Boolean)

  return Array.from(new Set(badges)).slice(0, 3)
}

function collectHighlightTags(item: DailyReportItem): string[] {
  const digestTags = (item.digest_card?.tags || []).map((value) => String(value || "").trim())
  const keywordTags = (item.matched_keywords || []).map((value) => String(value || "").trim())
  return Array.from(new Set([...digestTags, ...keywordTags].filter(Boolean))).slice(0, 3)
}

export function buildDashboardDailyBrief(report: DailyReport): DashboardDailyBrief {
  const queryRows = (report.queries || []).map((query) => ({
    queryLabel: String(query.normalized_query || query.raw_query || "").trim(),
    items: query.top_items || [],
  }))

  const queryItems = queryRows.flatMap((row) =>
    row.items.map((item) => ({
      item,
      queryLabel: row.queryLabel || "Daily digest",
    })),
  )
  const globalItems = (report.global_top || []).map((item) => ({
    item,
    queryLabel: (item.matched_queries || []).map((value) => String(value || "").trim()).find(Boolean) || "Daily digest",
  }))

  const highlightSeed = globalItems.length > 0 ? globalItems : queryItems
  const highlights = dedupeHighlightItems(highlightSeed).slice(0, 4).map(({ item, queryLabel }, index) => ({
    id: buildHighlightKey(item, `${queryLabel}-${index}`),
    title: String(item.title || "Untitled paper"),
    href: pickHighlightHref(item),
    queryLabel,
    venueLabel: String(item.subject_or_venue || item.venue || "DailyPaper"),
    metricLabel: buildHighlightMetric(item),
    summary: buildHighlightSummary(item),
    sourceBadges: collectHighlightBadges(item),
    tags: collectHighlightTags(item),
    recommendation: toRecommendationLabel(item.judge?.recommendation),
  }))

  const queryPulse = queryRows
    .filter((row) => row.queryLabel)
    .map((row) => ({
      query: row.queryLabel,
      hits: row.items.length,
    }))
    .sort((left, right) => right.hits - left.hits)
    .slice(0, 6)

  const trendRows = (report.llm_analysis?.query_trends || [])
    .map((row) => ({
      query: String(row.query || "").trim(),
      analysis: String(row.analysis || "").trim(),
    }))
    .filter((row) => row.query && row.analysis)
    .slice(0, 3)

  const reportSourceBadges = Array.from(
    new Set(
      [...(report.sources || []), report.source || ""]
        .map((value) => toSourceLabel(String(value || "")))
        .filter(Boolean),
    ),
  )

  return {
    title: String(report.title || "DailyPaper Digest"),
    date: String(report.date || ""),
    generatedAt: report.generated_at || null,
    sourceLabel: reportSourceBadges[0] || "DailyPaper",
    sourceBadges: reportSourceBadges,
    stats: {
      uniqueItems: Number(report.stats?.unique_items || 0),
      totalQueryHits: Number(report.stats?.total_query_hits || 0),
      queryCount: Number(report.stats?.query_count || queryRows.length),
    },
    highlights,
    queryPulse,
    trendRows,
  }
}

async function resolveReportDirectory(): Promise<string | null> {
  const candidates = [
    path.join(process.cwd(), "reports", "dailypaper"),
    path.join(process.cwd(), "..", "reports", "dailypaper"),
  ]

  for (const candidate of candidates) {
    try {
      const stats = await fs.stat(candidate)
      if (stats.isDirectory()) return candidate
    } catch {
      continue
    }
  }

  return null
}

async function readLatestDailyReport(): Promise<DailyReport | null> {
  const reportDir = await resolveReportDirectory()
  if (!reportDir) return null

  const entries = await fs.readdir(reportDir, { withFileTypes: true })
  const jsonFiles = entries
    .filter((entry) => entry.isFile() && entry.name.endsWith(".json"))
    .map((entry) => entry.name)

  if (jsonFiles.length === 0) return null

  const filesWithMtime = await Promise.all(
    jsonFiles.map(async (name) => {
      const fullPath = path.join(reportDir, name)
      const stats = await fs.stat(fullPath)
      return {
        fullPath,
        mtimeMs: stats.mtimeMs,
      }
    }),
  )

  filesWithMtime.sort((left, right) => right.mtimeMs - left.mtimeMs)

  for (const entry of filesWithMtime) {
    try {
      const raw = await fs.readFile(entry.fullPath, "utf-8")
      return JSON.parse(raw) as DailyReport
    } catch {
      continue
    }
  }

  return null
}

export async function fetchLatestDashboardDailyBrief(): Promise<DashboardDailyBrief | null> {
  const report = await readLatestDailyReport()
  if (!report) return null
  return buildDashboardDailyBrief(report)
}
