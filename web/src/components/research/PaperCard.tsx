"use client"

import { useEffect, useState } from "react"
import { Check, ChevronDown, ChevronRight, ExternalLink, FlaskConical, Database, CheckCircle, AlertTriangle, Heart, Loader2, Save, ThumbsDown } from "lucide-react"

import {
  normalizePaperPreferenceAction,
  togglePaperPreferenceAction,
  toggleSaveFeedbackAction,
  type PaperFeedbackAction,
  type PaperPreferenceAction,
  type PaperFeedbackRequestAction,
} from "@/lib/paper-feedback"
import { cn, safeHref } from "@/lib/utils"
import { ReasoningBlock, ToolActionsGroup } from "@/components/ai-elements"
import { Badge } from "@/components/ui/badge"

export type Paper = {
  paper_id: string
  title: string
  abstract?: string
  year?: number
  venue?: string
  citation_count?: number
  authors?: string[]
  url?: string
  latest_judge?: {
    overall?: number
    recommendation?: string
    one_line_summary?: string
    judge_model?: string
    evidence_quotes?: Array<{ text: string; source_url?: string; page_hint?: string }>
  }
  is_saved?: boolean
  is_liked?: boolean
  is_disliked?: boolean
  feedback_action?: PaperFeedbackAction | null
  retrieval_sources?: string[]
  retrieval_score?: number
  source?: string
  structured_card?: {
    method?: string
    dataset?: string
    conclusion?: string
    limitations?: string
  }
}

interface PaperCardProps {
  paper: Paper
  rank?: number
  reasons?: string[]
  onFeedbackAction?: (
    action: PaperFeedbackRequestAction
  ) => Promise<PaperFeedbackAction | null | undefined> | PaperFeedbackAction | null | undefined
  isLoading?: boolean
  className?: string
}

function derivePreferenceAction(
  feedbackAction: Paper["feedback_action"],
  isLiked: Paper["is_liked"],
  isDisliked: Paper["is_disliked"]
): PaperPreferenceAction | null {
  const explicitAction = normalizePaperPreferenceAction(feedbackAction)
  if (explicitAction) {
    return explicitAction
  }
  if (isLiked) {
    return "like"
  }
  if (isDisliked) {
    return "dislike"
  }
  return null
}

export function PaperCard({
  paper,
  rank,
  reasons,
  onFeedbackAction,
  isLoading = false,
  className,
}: PaperCardProps) {
  const [isSaved, setIsSaved] = useState(Boolean(paper.is_saved))
  const derivedPreferenceAction = derivePreferenceAction(
    paper.feedback_action,
    paper.is_liked,
    paper.is_disliked
  )
  const [preferenceAction, setPreferenceAction] = useState<PaperPreferenceAction | null>(
    derivedPreferenceAction
  )

  useEffect(() => {
    setIsSaved(Boolean(paper.is_saved))
    setPreferenceAction(derivedPreferenceAction)
  }, [derivedPreferenceAction, paper.is_saved])

  const isLiked = preferenceAction === "like"
  const isDisliked = preferenceAction === "dislike"
  const [actionLoading, setActionLoading] = useState<string | null>(null)
  const [evidenceOpen, setEvidenceOpen] = useState(false)
  const [cardOpen, setCardOpen] = useState(false)
  const [cardLoading, setCardLoading] = useState(false)
  const [structuredCard, setStructuredCard] = useState(paper.structured_card || null)

  const authorText = paper.authors?.slice(0, 3).join(", ") || "Unknown authors"
  const hasMoreAuthors = (paper.authors?.length || 0) > 3
  const safeUrl = safeHref(paper.url)
  const judge = paper.latest_judge
  const judgeOverall = Number(judge?.overall || 0)
  const judgeRec = String(judge?.recommendation || "").replace(/_/g, " ")
  const evidenceQuotes = judge?.evidence_quotes || []

  const runSaveAction = async () => {
    if (!onFeedbackAction) return
    const requestAction = toggleSaveFeedbackAction(isSaved)
    setActionLoading("save")
    try {
      await onFeedbackAction(requestAction)
      setIsSaved((prev) => !prev)
    } finally {
      setActionLoading(null)
    }
  }

  const runPreferenceAction = async (targetAction: PaperPreferenceAction) => {
    if (!onFeedbackAction) return
    const requestAction = togglePaperPreferenceAction(preferenceAction, targetAction)
    setActionLoading(targetAction)
    try {
      await onFeedbackAction(requestAction)
      setPreferenceAction(requestAction === targetAction ? targetAction : null)
    } finally {
      setActionLoading(null)
    }
  }

  const handleSave = async () => {
    await runSaveAction()
  }

  const handleLike = async () => {
    await runPreferenceAction("like")
  }

  const handleDislike = async () => {
    await runPreferenceAction("dislike")
  }

  const handleToggleCard = async () => {
    if (cardOpen) {
      setCardOpen(false)
      return
    }
    setCardOpen(true)
    if (structuredCard) return
    setCardLoading(true)
    try {
      const res = await fetch(`/api/research/papers/${encodeURIComponent(paper.paper_id)}/card`)
      if (res.ok) {
        const data = await res.json()
        setStructuredCard(data.structured_card || null)
      }
    } catch {
      // silently fail
    } finally {
      setCardLoading(false)
    }
  }

  return (
    <div
      className={cn(
        "rounded-lg border bg-card p-4 space-y-3 transition-all duration-200",
        "hover:bg-accent/50 hover:shadow-sm",
        isDisliked && "opacity-60",
        className
      )}
    >
      {/* Header with title and link */}
      <div className="space-y-1">
        <div className="flex items-start justify-between gap-2">
          <h3 className="font-medium leading-snug text-base">
            {rank !== undefined && (
              <span className="text-muted-foreground mr-2">#{rank + 1}</span>
            )}
            {paper.title}
          </h3>
          {safeUrl && (
            <a
              href={safeUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="shrink-0 text-muted-foreground hover:text-foreground transition-colors"
              title="Open paper"
            >
              <ExternalLink className="h-4 w-4" />
            </a>
          )}
        </div>

        {/* Meta info */}
        <div className="text-sm text-muted-foreground flex flex-wrap items-center gap-x-1.5">
          <span className="truncate max-w-[200px] sm:max-w-none">{authorText}</span>
          {hasMoreAuthors && <span>et al.</span>}
          {paper.venue && (
            <>
              <span>·</span>
              <span className="truncate max-w-[150px] sm:max-w-none">{paper.venue}</span>
            </>
          )}
          {paper.year && (
            <>
              <span>·</span>
              <span>{paper.year}</span>
            </>
          )}
          {paper.citation_count !== undefined && paper.citation_count > 0 && (
            <>
              <span>·</span>
              <span>{paper.citation_count} citations</span>
            </>
          )}
        </div>
      </div>

      {/* Abstract preview */}
      {paper.abstract && (
        <p className="text-sm text-muted-foreground line-clamp-2 sm:line-clamp-3">
          {paper.abstract}
        </p>
      )}

      {/* Structured Card */}
      <div>
        <button
          type="button"
          onClick={handleToggleCard}
          className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          {cardOpen ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
          Structured Card
        </button>
        {cardOpen && (
          <div className="mt-1.5 space-y-1.5">
            {cardLoading ? (
              <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <Loader2 className="h-3 w-3 animate-spin" /> Extracting...
              </div>
            ) : structuredCard ? (
              <div className="grid gap-1.5 text-xs">
                {structuredCard.method && (
                  <div className="flex items-start gap-1.5">
                    <FlaskConical className="h-3.5 w-3.5 mt-0.5 shrink-0 text-blue-500" />
                    <div><span className="font-medium">Method:</span> {structuredCard.method}</div>
                  </div>
                )}
                {structuredCard.dataset && (
                  <div className="flex items-start gap-1.5">
                    <Database className="h-3.5 w-3.5 mt-0.5 shrink-0 text-green-500" />
                    <div><span className="font-medium">Dataset:</span> {structuredCard.dataset}</div>
                  </div>
                )}
                {structuredCard.conclusion && (
                  <div className="flex items-start gap-1.5">
                    <CheckCircle className="h-3.5 w-3.5 mt-0.5 shrink-0 text-emerald-500" />
                    <div><span className="font-medium">Conclusion:</span> {structuredCard.conclusion}</div>
                  </div>
                )}
                {structuredCard.limitations && (
                  <div className="flex items-start gap-1.5">
                    <AlertTriangle className="h-3.5 w-3.5 mt-0.5 shrink-0 text-amber-500" />
                    <div><span className="font-medium">Limitations:</span> {structuredCard.limitations}</div>
                  </div>
                )}
              </div>
            ) : (
              <p className="text-xs text-muted-foreground">No structured card available.</p>
            )}
          </div>
        )}
      </div>

      {/* Recommendation reasons */}
      {reasons && reasons.length > 0 && (
        <ReasoningBlock reasons={reasons} compact title="Why this paper" />
      )}

      {judge && judgeOverall > 0 && (
        <div className="flex flex-wrap gap-1.5">
          <Badge variant="secondary" className="text-xs">
            Judge {judgeOverall.toFixed(1)}
          </Badge>
          {judgeRec && (
            <Badge variant="outline" className="text-xs capitalize">
              {judgeRec}
            </Badge>
          )}
        </div>
      )}

      {/* Evidence quotes */}
      {judge && judgeOverall > 0 && (
        <div>
          <button
            type="button"
            onClick={() => setEvidenceOpen(!evidenceOpen)}
            className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            {evidenceOpen ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
            Evidence {evidenceQuotes.length > 0 ? `(${evidenceQuotes.length})` : ""}
          </button>
          {evidenceOpen && (
            <div className="mt-1.5 space-y-2">
              {evidenceQuotes.length > 0 ? (
                evidenceQuotes.map((eq, i) => (
                  <div key={i} className="border-l-2 border-muted-foreground/30 pl-3 text-xs text-muted-foreground">
                    <p className="italic">{eq.text}</p>
                    <div className="mt-0.5 flex items-center gap-2">
                      {eq.source_url && (
                        <a
                          href={safeHref(eq.source_url) || "#"}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-primary hover:underline"
                        >
                          source
                        </a>
                      )}
                      {eq.page_hint && <span>p. {eq.page_hint}</span>}
                    </div>
                  </div>
                ))
              ) : (
                <Badge variant="outline" className="text-[10px] text-muted-foreground">No evidence</Badge>
              )}
            </div>
          )}
        </div>
      )}

      {/* Action buttons */}
      <ToolActionsGroup
        className="pt-1"
        ariaLabel="Paper actions"
        actions={[
          ...(onFeedbackAction
            ? [
                {
                  id: "save",
                  label: isSaved ? "Saved" : "Save",
                  variant: (isSaved ? "default" : "outline") as
                    | "default"
                    | "outline"
                    | "ghost"
                    | "destructive"
                    | "secondary",
                  className: cn(
                    "transition-all",
                    isSaved && "bg-green-600 hover:bg-green-700 text-white"
                  ),
                  onClick: handleSave,
                  disabled: isLoading || actionLoading !== null,
                  icon:
                    actionLoading === "save" ? (
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    ) : isSaved ? (
                      <Check className="h-3.5 w-3.5" />
                    ) : (
                      <Save className="h-3.5 w-3.5" />
                    ),
                },
              ]
            : []),
          ...(onFeedbackAction
            ? [
                {
                  id: "like",
                  label: isLiked ? "Liked" : "Like",
                  variant: "ghost" as const,
                  className: cn("transition-all", isLiked && "text-red-500 hover:text-red-600"),
                  onClick: handleLike,
                  disabled: isLoading || actionLoading !== null,
                  icon:
                    actionLoading === "like" ? (
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    ) : (
                      <Heart className={cn("h-3.5 w-3.5", isLiked && "fill-current")} />
                    ),
                },
              ]
            : []),
          ...(onFeedbackAction
            ? [
                {
                  id: "dislike",
                  label: isDisliked ? "Hidden" : "Not relevant",
                  variant: "ghost" as const,
                  className: cn(
                    "transition-all",
                    isDisliked
                      ? "text-orange-500 hover:text-orange-600"
                      : "text-muted-foreground hover:text-destructive"
                  ),
                  onClick: handleDislike,
                  disabled: isLoading || actionLoading !== null,
                  icon:
                    actionLoading === "dislike" ? (
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    ) : (
                      <ThumbsDown className={cn("h-3.5 w-3.5", isDisliked && "fill-current")} />
                    ),
                },
              ]
            : []),
        ]}
      />
    </div>
  )
}
