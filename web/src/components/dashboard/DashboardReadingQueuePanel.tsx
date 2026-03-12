"use client"

import Link from "next/link"
import { useState } from "react"
import {
  AlertCircle,
  Archive,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  ExternalLink,
  FileText,
  Loader2,
  MessageSquare,
  Sparkles,
} from "lucide-react"

import { Button } from "@/components/ui/button"
import { cn, safeHref, safeInternalHref } from "@/lib/utils"
import type { DashboardReadingQueueItem, DashboardReadingQueuePriority } from "@/lib/dashboard-reading-queue"

type QueueDetail = {
  abstract?: string | null
  judgeSummary?: string | null
  method?: string | null
  dataset?: string | null
  conclusion?: string | null
  limitations?: string | null
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" ? value as Record<string, unknown> : null
}

function getNestedRecord(source: Record<string, unknown> | null, key: string): Record<string, unknown> | null {
  return asRecord(source?.[key])
}

function getOptionalString(source: Record<string, unknown> | null, key: string): string | null {
  const value = source?.[key]
  return typeof value === "string" ? value : null
}

function PriorityBadge({ level }: { level: DashboardReadingQueuePriority }) {
  const styles = {
    high: "border-rose-100 bg-rose-50 text-rose-700",
    medium: "border-amber-100 bg-amber-50 text-amber-700",
    low: "border-slate-100 bg-slate-50 text-slate-600",
  } satisfies Record<DashboardReadingQueuePriority, string>

  const labels = {
    high: "High",
    medium: "Medium",
    low: "Low",
  } satisfies Record<DashboardReadingQueuePriority, string>

  return (
    <span className={`inline-flex items-center rounded-full border px-2.5 py-1 text-[11px] font-semibold ${styles[level]}`}>
      {level === "high" ? <AlertCircle className="mr-1 size-3" /> : null}
      {labels[level]}
    </span>
  )
}

function extractQueueDetail(detailPayload: unknown, cardPayload: unknown): QueueDetail {
  const detail = getNestedRecord(asRecord(detailPayload), "detail")
  const paper = getNestedRecord(detail, "paper")
  const latestJudge = getNestedRecord(detail, "latest_judge")
  const structuredCard = getNestedRecord(asRecord(cardPayload), "structured_card")

  return {
    abstract: getOptionalString(paper, "abstract"),
    judgeSummary: getOptionalString(latestJudge, "one_line_summary"),
    method: getOptionalString(structuredCard, "method"),
    dataset: getOptionalString(structuredCard, "dataset"),
    conclusion: getOptionalString(structuredCard, "conclusion"),
    limitations: getOptionalString(structuredCard, "limitations"),
  }
}

function QueueCard({
  item,
  activeTrackId,
  onArchived,
}: {
  item: DashboardReadingQueueItem
  activeTrackId: number | null
  onArchived: (id: string) => void
}) {
  const [expanded, setExpanded] = useState(false)
  const [detail, setDetail] = useState<QueueDetail | null>(null)
  const [loadingDetail, setLoadingDetail] = useState(false)
  const [loadingSave, setLoadingSave] = useState(false)
  const [loadingArchive, setLoadingArchive] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isSaved, setIsSaved] = useState(Boolean(item.isSaved))

  const detailRef = item.internalPaperId !== null ? String(item.internalPaperId) : item.paperRef
  const canLoadDetail = Boolean(detailRef)
  const canSave = Boolean(item.canSave) && (item.internalPaperId !== null || Boolean(activeTrackId))
  const canArchive = Boolean(isSaved && detailRef)
  const safeResearchHref = safeInternalHref(item.researchHref) || "/research"
  const safePaperHref = item.isExternal ? safeHref(item.href) : safeInternalHref(item.href)

  async function loadDetail() {
    if (!detailRef || detail || loadingDetail) return
    setLoadingDetail(true)
    setError(null)
    try {
      const [detailRes, cardRes] = await Promise.all([
        fetch(`/api/research/papers/${encodeURIComponent(detailRef)}?user_id=default`, { cache: "no-store" }),
        fetch(`/api/research/papers/${encodeURIComponent(detailRef)}/card?user_id=default`, { cache: "no-store" }),
      ])

      const detailPayload = detailRes.ok ? await detailRes.json() : null
      const cardPayload = cardRes.ok ? await cardRes.json() : null
      setDetail(extractQueueDetail(detailPayload, cardPayload))
    } catch (fetchError) {
      const message = fetchError instanceof Error ? fetchError.message : String(fetchError)
      setError(message)
    } finally {
      setLoadingDetail(false)
    }
  }

  async function handleToggleSummary() {
    const nextExpanded = !expanded
    setExpanded(nextExpanded)
    if (nextExpanded) {
      await loadDetail()
    }
  }

  async function handleSave() {
    if (!canSave || loadingSave) return
    setLoadingSave(true)
    setError(null)
    try {
      if (item.internalPaperId !== null) {
        const res = await fetch(`/api/papers/${item.internalPaperId}/save`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_id: "default",
            track_id: activeTrackId,
          }),
        })
        if (!res.ok) {
          const detailText = await res.text()
          throw new Error(detailText || "Failed to save paper")
        }
      } else if (item.paperRef) {
        const res = await fetch("/api/research/papers/feedback", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_id: "default",
            track_id: activeTrackId,
            paper_id: item.paperRef,
            action: "save",
            paper_title: item.title,
            paper_authors: item.authors || [],
            paper_year: item.year,
            paper_venue: item.venue,
            paper_url: item.isExternal ? safeHref(item.href) || undefined : undefined,
            paper_source: item.paperSource || "semantic_scholar",
          }),
        })
        if (!res.ok) {
          const detailText = await res.text()
          throw new Error(detailText || "Failed to save paper")
        }
      }
      setIsSaved(true)
    } catch (saveError) {
      const message = saveError instanceof Error ? saveError.message : String(saveError)
      setError(message)
    } finally {
      setLoadingSave(false)
    }
  }

  async function handleArchive() {
    if (!canArchive || loadingArchive || !detailRef) return
    setLoadingArchive(true)
    setError(null)
    try {
      const res = await fetch(`/api/research/papers/${encodeURIComponent(detailRef)}/status`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: "default",
          status: "archived",
          mark_saved: true,
        }),
      })
      if (!res.ok) {
        const detailText = await res.text()
        throw new Error(detailText || "Failed to archive paper")
      }
      onArchived(item.id)
    } catch (archiveError) {
      const message = archiveError instanceof Error ? archiveError.message : String(archiveError)
      setError(message)
    } finally {
      setLoadingArchive(false)
    }
  }

  return (
    <article className="group rounded-2xl border border-slate-100 bg-white p-5 shadow-sm transition-all hover:border-indigo-200 hover:shadow-md">
      <div className="flex items-start justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <PriorityBadge level={item.priority} />
          <span className="text-xs font-medium text-slate-400">{item.sourceLabel}</span>
          {item.metric ? (
            <span className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] font-medium text-slate-600">
              {item.metric}
            </span>
          ) : null}
        </div>
        <span className="text-xs text-slate-400">{item.timeLabel}</span>
      </div>

      <h3 className="mt-3 text-lg font-bold leading-snug text-slate-800 transition-colors group-hover:text-indigo-600">
        {item.title}
      </h3>
      <p className="mt-1 flex items-center text-sm text-slate-500">
        <FileText className="mr-1.5 size-3.5" />
        {item.venue}
      </p>

      <div className="mt-4 rounded-xl border border-indigo-50 bg-indigo-50/70 p-3">
        <div className="flex items-start gap-2">
          <Sparkles className="mt-0.5 size-4 shrink-0 text-indigo-500" />
          <p className={cn("text-sm leading-relaxed text-slate-700", !expanded && "line-clamp-2")}>
            {item.summary}
          </p>
        </div>
      </div>

      {expanded ? (
        <div className="mt-4 rounded-xl border border-slate-100 bg-slate-50/70 p-4">
          {loadingDetail ? (
            <div className="flex items-center gap-2 text-sm text-slate-500">
              <Loader2 className="size-4 animate-spin" />
              Loading summary…
            </div>
          ) : (
            <div className="space-y-3">
              {detail?.judgeSummary ? (
                <p className="text-sm font-medium text-slate-800">{detail.judgeSummary}</p>
              ) : null}
              {detail?.abstract ? (
                <p className="text-sm leading-6 text-slate-600">{detail.abstract}</p>
              ) : (
                <p className="text-sm leading-6 text-slate-600">{item.summary}</p>
              )}

              <div className="grid gap-2 sm:grid-cols-2">
                {detail?.method ? (
                  <div className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600">
                    <span className="font-semibold text-slate-900">Method:</span> {detail.method}
                  </div>
                ) : null}
                {detail?.dataset ? (
                  <div className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600">
                    <span className="font-semibold text-slate-900">Dataset:</span> {detail.dataset}
                  </div>
                ) : null}
                {detail?.conclusion ? (
                  <div className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600 sm:col-span-2">
                    <span className="font-semibold text-slate-900">Conclusion:</span> {detail.conclusion}
                  </div>
                ) : null}
                {detail?.limitations ? (
                  <div className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600 sm:col-span-2">
                    <span className="font-semibold text-slate-900">Limitations:</span> {detail.limitations}
                  </div>
                ) : null}
              </div>
            </div>
          )}
        </div>
      ) : null}

      {item.tags.length > 0 ? (
        <div className="mt-4 flex flex-wrap gap-2">
          {item.tags.map((tag) => (
            <span
              key={`${item.id}-${tag}`}
              className="rounded-md bg-slate-50 px-2.5 py-1 text-xs font-medium text-slate-600"
            >
              {tag}
            </span>
          ))}
        </div>
      ) : null}

      {error ? (
        <div className="mt-4 rounded-lg border border-rose-100 bg-rose-50 px-3 py-2 text-sm text-rose-700">
          {error}
        </div>
      ) : null}

      <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-slate-50 pt-4">
        <div className="flex flex-wrap gap-2">
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="rounded-lg"
            onClick={() => {
              handleToggleSummary().catch(() => {})
            }}
            disabled={loadingDetail || !canLoadDetail}
          >
            {expanded ? <ChevronDown className="mr-1.5 size-4" /> : <ChevronRight className="mr-1.5 size-4" />}
            Summary
          </Button>
          <Button
            type="button"
            variant={isSaved ? "secondary" : "outline"}
            size="sm"
            className="rounded-lg"
            onClick={() => {
              handleSave().catch(() => {})
            }}
            disabled={loadingSave || !canSave || isSaved}
          >
            {loadingSave ? <Loader2 className="mr-1.5 size-4 animate-spin" /> : isSaved ? <CheckCircle2 className="mr-1.5 size-4" /> : null}
            {isSaved ? "Saved" : "Save to library"}
          </Button>
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="rounded-lg"
            onClick={() => {
              handleArchive().catch(() => {})
            }}
            disabled={loadingArchive || !canArchive}
          >
            {loadingArchive ? <Loader2 className="mr-1.5 size-4 animate-spin" /> : <Archive className="mr-1.5 size-4" />}
            Archive
          </Button>
        </div>

        <div className="flex items-center gap-2">
          <Button asChild variant="ghost" size="sm" className="rounded-lg">
            <Link href={safeResearchHref}>
              <MessageSquare className="mr-1.5 size-4" />
              Analyze
            </Link>
          </Button>
          {item.isExternal && safePaperHref ? (
            <Button asChild size="sm" className="rounded-lg bg-indigo-600 hover:bg-indigo-700">
              <a href={safePaperHref} target="_blank" rel="noreferrer">
                <ExternalLink className="mr-1.5 size-4" />
                Open paper
              </a>
            </Button>
          ) : safePaperHref ? (
            <Button asChild size="sm" className="rounded-lg bg-indigo-600 hover:bg-indigo-700">
              <Link href={safePaperHref}>Open paper</Link>
            </Button>
          ) : (
            <Button size="sm" className="rounded-lg bg-indigo-600 hover:bg-indigo-700" disabled>
              Open paper
            </Button>
          )}
        </div>
      </div>
    </article>
  )
}

export default function DashboardReadingQueuePanel({
  initialItems,
  activeTrackId,
}: {
  initialItems: DashboardReadingQueueItem[]
  activeTrackId: number | null
}) {
  const [items, setItems] = useState(initialItems)

  if (items.length === 0) {
    return (
      <div className="rounded-2xl border border-dashed border-slate-300 bg-white p-6 text-sm text-slate-600 shadow-sm">
        No queued papers yet.
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {items.map((item) => (
        <QueueCard
          key={item.id}
          item={item}
          activeTrackId={activeTrackId}
          onArchived={(id) => setItems((prev) => prev.filter((entry) => entry.id !== id))}
        />
      ))}
    </div>
  )
}
