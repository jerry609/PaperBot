"use client"

import { useEffect, useMemo, useState } from "react"
import { Loader2, RefreshCw } from "lucide-react"

import { Button } from "@/components/ui/button"
import { getErrorMessage } from "@/lib/fetch"
import {
  normalizePaperPreferenceAction,
  type PaperFeedbackAction,
  type PaperFeedbackRequestAction,
} from "@/lib/paper-feedback"
import { Card, CardContent } from "@/components/ui/card"

import { PaperCard, type Paper } from "./PaperCard"

type FeedItem = {
  paper: {
    id?: number
    title?: string
    abstract?: string
    authors?: string[]
    year?: number
    venue?: string
    citation_count?: number
    url?: string
  }
  latest_judge?: Paper["latest_judge"]
  latest_feedback_action?: string | null
  is_saved?: boolean
  is_liked?: boolean
  is_disliked?: boolean
}

type FeedResponse = {
  items: FeedItem[]
  total: number
  limit: number
  offset: number
}

interface FeedTabProps {
  userId: string
  trackId: number | null
  onFeedbackAction?: (
    paperId: string,
    action: PaperFeedbackRequestAction,
    rank: number,
    paper: Paper
  ) => Promise<PaperFeedbackAction | null | undefined> | PaperFeedbackAction | null | undefined
}

function toPaper(item: FeedItem): Paper {
  const id = String(item.paper.id || "")
  const preferenceAction =
    normalizePaperPreferenceAction(item.latest_feedback_action) ||
    (item.is_liked ? "like" : item.is_disliked ? "dislike" : null)
  return {
    paper_id: id,
    title: item.paper.title || "Untitled",
    abstract: item.paper.abstract || "",
    authors: item.paper.authors || [],
    year: item.paper.year,
    venue: item.paper.venue,
    citation_count: item.paper.citation_count || 0,
    url: item.paper.url,
    latest_judge: item.latest_judge,
    feedback_action: preferenceAction,
    is_saved: Boolean(item.is_saved),
    is_liked: preferenceAction === "like",
    is_disliked: preferenceAction === "dislike",
  }
}

export function FeedTab({ userId, trackId, onFeedbackAction }: FeedTabProps) {
  const [items, setItems] = useState<FeedItem[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const papers = useMemo(() => items.map(toPaper), [items])

  const load = async () => {
    if (!trackId) {
      setItems([])
      return
    }
    setLoading(true)
    setError(null)
    try {
      const qs = new URLSearchParams({
        user_id: userId,
        limit: "20",
        offset: "0",
      })
      const res = await fetch(`/api/research/tracks/${trackId}/feed?${qs.toString()}`)
      if (!res.ok) {
        throw new Error(`${res.status} ${res.statusText}`)
      }
      const payload = (await res.json()) as FeedResponse
      setItems(payload.items || [])
    } catch (e) {
      setError(getErrorMessage(e))
      setItems([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load().catch(() => {})
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userId, trackId])

  if (!trackId) {
    return <div className="py-8 text-sm text-muted-foreground">Select a track to view feed.</div>
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-sm text-muted-foreground">Track feed from DailyPaper + personalized ranking.</p>
        <Button variant="outline" size="sm" onClick={() => load().catch(() => {})} disabled={loading}>
          {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
          Refresh
        </Button>
      </div>

      {error && (
        <Card className="border-red-200 bg-red-50">
          <CardContent className="py-3 text-sm text-red-700">{error}</CardContent>
        </Card>
      )}

      {loading && !papers.length ? (
        <div className="py-8 text-sm text-muted-foreground">Loading feed...</div>
      ) : !papers.length ? (
        <div className="py-8 text-sm text-muted-foreground">No feed items yet for this track.</div>
      ) : (
        <div className="space-y-3">
          {papers.map((paper, idx) => (
            <PaperCard
              key={`${paper.paper_id}-${idx}`}
              paper={paper}
              rank={idx}
              onFeedbackAction={
                onFeedbackAction
                  ? (action) => onFeedbackAction(paper.paper_id, action, idx, paper)
                  : undefined
              }
            />
          ))}
        </div>
      )}
    </div>
  )
}
