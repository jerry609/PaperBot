import type { IntelligenceFeedItem } from "./types"

export interface DashboardIntelligenceCard {
  id: string
  source: string
  sourceLabel: string
  title: string
  summary: string
  href: string
  isExternal: boolean
  researchHref: string
  researchLabel: string
  metricLabel: string
  reasonChips: string[]
  matchedTrackNames: string[]
  timestamp?: string | null
}

function buildMetricLabel(item: IntelligenceFeedItem): string {
  const metricName = String(item.metric?.name || "").trim()
  const metricValue = Number(item.metric?.value || 0)
  const metricDelta = Number(item.metric?.delta || 0)

  if (!metricName) {
    return `score ${Math.round(Number(item.score || 0))}`
  }

  const deltaSuffix = metricDelta > 0 ? ` (+${metricDelta})` : metricDelta < 0 ? ` (${metricDelta})` : ""
  return `${metricName} ${metricValue}${deltaSuffix}`
}

function buildReasonChips(item: IntelligenceFeedItem): string[] {
  const directReasons = (item.match_reasons || []).map((reason) => String(reason).trim()).filter(Boolean)
  if (directReasons.length > 0) {
    return directReasons.slice(0, 4)
  }

  const fallbackReasons = [
    ...(item.keyword_hits || []).map((keyword) => `keyword: ${keyword}`),
    ...(Number(item.metric?.delta || 0) > 0 ? [`delta: +${Number(item.metric?.delta || 0)}`] : []),
    ...(item.author_matches || []).map((author) => `author: ${author}`),
    ...(item.repo_matches || []).map((repo) => `repo: ${repo}`),
  ]

  return fallbackReasons.slice(0, 4)
}

function buildSignalHref(item: IntelligenceFeedItem): string {
  const url = String(item.url || "").trim()
  if (url) {
    return url
  }

  const repo = String(item.repo_full_name || "").trim()
  if (repo) {
    return `https://github.com/${repo}`
  }

  return "/research"
}

function buildResearchHref(item: IntelligenceFeedItem): string {
  const params = new URLSearchParams()
  const firstTrack = (item.matched_tracks || [])[0]
  if (firstTrack?.track_id) {
    params.set("track_id", String(firstTrack.track_id))
  }

  const query = String(item.research_query || "").trim()
  if (query) {
    params.set("query", query)
  }

  const search = params.toString()
  return search ? `/research?${search}` : "/research"
}

export function buildDashboardIntelligenceCards(
  items: IntelligenceFeedItem[],
): DashboardIntelligenceCard[] {
  if (items.length === 0) {
    return [
      {
        id: "empty-intelligence",
        source: "signal",
        sourceLabel: "Community Radar",
        title: "No urgent community signals",
        summary:
          "Keyword hits, trend deltas, and author or repository linkages are calm across the watched third-party sources.",
        href: "/research",
        isExternal: false,
        researchHref: "/research",
        researchLabel: "进入 Research",
        metricLabel: "stable",
        reasonChips: [],
        matchedTrackNames: [],
        timestamp: null,
      },
    ]
  }

  return items.slice(0, 3).map((item) => {
    const href = buildSignalHref(item)
    const firstTrack = (item.matched_tracks || [])[0]
    return {
      id: item.id,
      source: String(item.source || "signal"),
      sourceLabel: String(item.source_label || "Community Radar"),
      title: String(item.title || item.source_label || "Community signal"),
      summary: String(item.summary || "Third-party source reported a new community signal."),
      href,
      isExternal: /^https?:\/\//.test(href),
      researchHref: buildResearchHref(item),
      researchLabel: firstTrack?.track_name ? `转到 ${firstTrack.track_name}` : "进入 Research",
      metricLabel: buildMetricLabel(item),
      reasonChips: buildReasonChips(item),
      matchedTrackNames: (item.matched_tracks || []).map((track) => String(track.track_name || "")).filter(Boolean),
      timestamp: item.detected_at || item.published_at || null,
    }
  })
}