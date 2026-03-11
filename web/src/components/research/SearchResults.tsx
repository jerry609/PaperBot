"use client"

import { Loader2 } from "lucide-react"

import {
  type PaperFeedbackAction,
  type PaperFeedbackRequestAction,
} from "@/lib/paper-feedback"
import { cn } from "@/lib/utils"
import { Card, CardContent } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"

import { PaperCard, type Paper } from "./PaperCard"

interface SearchResultsProps {
  papers: Paper[]
  reasons?: Record<string, string[]>
  isSearching?: boolean
  hasSearched: boolean
  className?: string
  selectedSources?: string[]
  onToggleSource?: (source: string) => void
  onFeedbackAction?: (
    paperId: string,
    action: PaperFeedbackRequestAction,
    rank: number,
    paper: Paper
  ) => Promise<PaperFeedbackAction | null | undefined> | PaperFeedbackAction | null | undefined
}

const SOURCE_OPTIONS: Array<{ value: string; label: string }> = [
  { value: "semantic_scholar", label: "S2" },
  { value: "arxiv", label: "arXiv" },
  { value: "openalex", label: "OpenAlex" },
  { value: "papers_cool", label: "papers.cool" },
  { value: "hf_daily", label: "HF Daily" },
]

function PaperCardSkeleton() {
  return (
    <div className="rounded-lg border bg-card p-4 space-y-3">
      {/* Title skeleton */}
      <div className="space-y-2">
        <Skeleton className="h-5 w-3/4" />
        <Skeleton className="h-4 w-1/2" />
      </div>
      {/* Abstract skeleton */}
      <div className="space-y-1.5">
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-5/6" />
      </div>
      {/* Actions skeleton */}
      <div className="flex gap-2 pt-1">
        <Skeleton className="h-8 w-20" />
        <Skeleton className="h-8 w-16" />
        <Skeleton className="h-8 w-24" />
      </div>
    </div>
  )
}

export function SearchResults({
  papers,
  reasons,
  isSearching = false,
  hasSearched,
  className,
  selectedSources = ["semantic_scholar"],
  onToggleSource,
  onFeedbackAction,
}: SearchResultsProps) {
  // Not searched yet - show nothing
  if (!hasSearched) {
    return null
  }

  // Searching - show loading skeletons
  if (isSearching) {
    return (
      <div className={cn("w-full", className)}>
        <div className="flex items-center gap-3 mb-4">
          <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
          <p className="text-sm text-muted-foreground">Searching for papers...</p>
        </div>
        <div className="space-y-3">
          {[...Array(5)].map((_, i) => (
            <PaperCardSkeleton key={i} />
          ))}
        </div>
      </div>
    )
  }

  // No results
  if (papers.length === 0) {
    return (
      <div className={cn("w-full py-12 animate-in fade-in duration-300", className)}>
        <div className="text-center text-muted-foreground">
          <p className="text-xl font-medium">No papers found</p>
          <p className="text-base mt-2">
            Try adjusting your search query or selecting a different track
          </p>
        </div>
      </div>
    )
  }

  // Show results with staggered animation
  return (
    <div className={cn("w-full animate-in fade-in duration-300", className)}>
      <Card>
        <CardContent className="p-4 sm:p-5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="space-y-0.5">
              <p className="text-sm text-muted-foreground">Recommendation stream</p>
              <p className="text-sm text-muted-foreground">
                Found <span className="font-medium text-foreground">{papers.length}</span> papers
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-1.5">
              {SOURCE_OPTIONS.map((source) => {
                const active = selectedSources.includes(source.value)
                return (
                  <button
                    key={source.value}
                    type="button"
                    onClick={() => onToggleSource?.(source.value)}
                    className={cn(
                      "rounded-md border px-2 py-1 text-xs transition-colors",
                      active
                        ? "border-primary bg-primary/10 text-primary"
                        : "border-border text-muted-foreground hover:text-foreground"
                    )}
                  >
                    {source.label}
                  </button>
                )
              })}
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="mt-4 space-y-3">
        {papers.map((paper, idx) => (
          <PaperCard
            key={paper.paper_id}
            paper={paper}
            rank={idx}
            reasons={reasons?.[paper.paper_id]}
            onFeedbackAction={
              onFeedbackAction
                ? (action) => onFeedbackAction(paper.paper_id, action, idx, paper)
                : undefined
            }
            className={cn(
              "animate-in fade-in slide-in-from-bottom-2",
              // Staggered animation delay
              idx === 0 && "duration-300",
              idx === 1 && "duration-300 delay-[50ms]",
              idx === 2 && "duration-300 delay-[100ms]",
              idx === 3 && "duration-300 delay-[150ms]",
              idx === 4 && "duration-300 delay-[200ms]",
              idx >= 5 && "duration-300 delay-[250ms]"
            )}
          />
        ))}
      </div>
    </div>
  )
}
