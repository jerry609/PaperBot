import { promises as fs } from "node:fs"
import path from "node:path"

type DailyReportJudge = {
  overall?: number | null
  recommendation?: string | null
  one_line_summary?: string | null
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
  sources?: string[]
  judge?: DailyReportJudge
  digest_card?: {
    highlight?: string
    tags?: string[]
  }
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
  global_top?: DailyReportItem[]
  queries?: DailyReportQuery[]
  llm_analysis?: {
    query_trends?: DailyReportTrend[]
  }
}

export type DashboardBriefRecommendation = {
  id: string
  title: string
  href: string
  meta: string
  summary: string
  tags: string[]
  metric: string
  recommendation?: string | null
}

export type DashboardBriefTrend = {
  query: string
  analysis: string
}

export type DashboardBriefSnapshot = {
  title: string
  date?: string | null
  generatedAt?: string | null
  sourceLabel: string
  recommendations: DashboardBriefRecommendation[]
  trendRows: DashboardBriefTrend[]
}

const SOURCE_LABELS: Record<string, string> = {
  papers_cool: "papers.cool",
  arxiv_api: "arXiv API",
  hf_daily: "HF Daily",
  arxiv: "arXiv",
  venue: "Venue",
}

function toSourceLabel(value?: string | null): string {
  const normalized = String(value || "").trim()
  if (!normalized) return "Daily Brief"
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

function buildMetric(item: DailyReportItem): string {
  if (item.judge?.overall != null) {
    return `Judge ${Number(item.judge.overall).toFixed(1)}`
  }

  if (item.score != null) {
    return `Score ${Number(item.score).toFixed(1)}`
  }

  return "Brief pick"
}

function buildHref(item: DailyReportItem): string {
  const candidates = [item.external_url, item.url, item.pdf_url]
  const href = candidates.find((value) => typeof value === "string" && value.trim())
  return href?.trim() || "/workflows"
}

function buildSummary(item: DailyReportItem, recommendation: string | null): string {
  const highlight = String(item.digest_card?.highlight || "").trim()
  if (highlight) return highlight

  const oneLine = String(item.judge?.one_line_summary || "").trim()
  if (oneLine) return oneLine

  const snippet = String(item.snippet || "").trim()
  if (snippet) return snippet

  if (recommendation) return `${recommendation} from the latest DailyPaper brief.`
  return "Picked from the latest DailyPaper brief."
}

function buildKey(item: DailyReportItem, index: number): string {
  return String(item.paper_id || item.url || item.title || `brief-${index}`).trim()
}

function dedupeItems(items: DailyReportItem[]): DailyReportItem[] {
  const seen = new Set<string>()
  const deduped: DailyReportItem[] = []

  for (const item of items) {
    const key = buildKey(item, deduped.length)
    if (!key || seen.has(key)) continue
    seen.add(key)
    deduped.push(item)
  }

  return deduped
}

export function buildDashboardBrief(report: DailyReport): DashboardBriefSnapshot {
  const queryItems = (report.queries || []).flatMap((query) => query.top_items || [])
  const primaryItems = dedupeItems((report.global_top || []).length ? report.global_top || [] : queryItems)

  return {
    title: String(report.title || "Daily Brief"),
    date: report.date || null,
    generatedAt: report.generated_at || null,
    sourceLabel: toSourceLabel((report.sources || [report.source || ""]).find(Boolean)),
    recommendations: primaryItems.slice(0, 4).map((item, index) => {
      const recommendation = toRecommendationLabel(item.judge?.recommendation)

      return {
        id: buildKey(item, index),
        title: String(item.title || "Untitled paper"),
        href: buildHref(item),
        meta: String(item.subject_or_venue || item.venue || "Daily Brief"),
        summary: buildSummary(item, recommendation),
        tags: Array.from(
          new Set(
            [...(item.digest_card?.tags || []), ...(item.matched_keywords || [])]
              .map((value) => String(value || "").trim())
              .filter(Boolean),
          ),
        ).slice(0, 3),
        metric: buildMetric(item),
        recommendation,
      }
    }),
    trendRows: (report.llm_analysis?.query_trends || [])
      .map((row) => ({
        query: String(row.query || "").trim(),
        analysis: String(row.analysis || "").trim(),
      }))
      .filter((row) => row.query && row.analysis)
      .slice(0, 3),
  }
}

async function resolveReportDirectory(): Promise<string | null> {
  const cwd = process.cwd()
  const candidates = [
    process.env.PAPERBOT_DAILYPAPER_OUTPUT_DIR,
    path.join(cwd, "reports", "dailypaper"),
    path.join(cwd, "..", "reports", "dailypaper"),
    path.join(cwd, "..", "output", "reports"),
    path.join(cwd, "output", "reports"),
  ].filter(Boolean) as string[]

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

async function readLatestReport(): Promise<DailyReport | null> {
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
      return { fullPath, mtimeMs: stats.mtimeMs }
    }),
  )

  filesWithMtime.sort((left, right) => right.mtimeMs - left.mtimeMs)

  for (const file of filesWithMtime) {
    try {
      const raw = await fs.readFile(file.fullPath, "utf-8")
      return JSON.parse(raw) as DailyReport
    } catch {
      continue
    }
  }

  return null
}

export async function fetchLatestDashboardBrief(): Promise<DashboardBriefSnapshot | null> {
  const report = await readLatestReport()
  if (!report) return null
  return buildDashboardBrief(report)
}
