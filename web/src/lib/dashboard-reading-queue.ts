export type DashboardReadingQueuePriority = "high" | "medium" | "low"

export type DashboardReadingQueueItem = {
  id: string
  paperRef: string | null
  internalPaperId: number | null
  title: string
  venue: string
  summary: string
  tags: string[]
  sourceLabel: string
  priority: DashboardReadingQueuePriority
  timeLabel: string
  href: string
  researchHref: string
  isExternal?: boolean
  metric?: string
  recommendation?: string | null
  authors?: string[]
  year?: number | null
  paperSource?: "arxiv" | "semantic_scholar" | "openalex" | null
  isSaved?: boolean
  canSave?: boolean
}
