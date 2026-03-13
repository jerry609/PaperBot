"use client"

import { BookMarked, BrainCircuit, CheckSquare2, Clock3, Database, Tags } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import type {
  ResearchTrackContextResponse,
  ResearchTrackSummary,
} from "@/lib/types"

type StatItem = {
  label: string
  value: string
  icon: typeof Database
}

export function buildStatItems(context: ResearchTrackContextResponse): StatItem[] {
  return [
    {
      label: "Pending Memory",
      value: String(context.memory.pending_items),
      icon: Clock3,
    },
    {
      label: "Saved Papers",
      value: String(context.saved_papers.total_items),
      icon: BookMarked,
    },
    {
      label: "Feedback",
      value: String(context.feedback.total_items),
      icon: BrainCircuit,
    },
  ]
}

function getTrackDescription(track: ResearchTrackSummary): string {
  const description = String(track.description || "").trim()
  if (description) {
    return description
  }
  const keywords = track.keywords || []
  if (keywords.length > 0) {
    return `Focused on ${keywords.slice(0, 4).join(", ")}.`
  }
  return "Use the track snapshot to keep context, memory, and saved papers aligned."
}

interface ResearchTrackContextPanelProps {
  context: ResearchTrackContextResponse
  onOpenMemory: () => void
}

export function ResearchTrackContextPanel({
  context,
  onOpenMemory,
}: ResearchTrackContextPanelProps) {
  const statItems = buildStatItems(context)
  const track = context.track
  const keywords = track.keywords || []
  const memoryTags = context.memory.top_tags || []

  return (
    <Card className="border-border/70 bg-card/85 shadow-sm backdrop-blur-sm">
      <CardHeader className="pb-3">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Badge variant="secondary">Track Snapshot</Badge>
              {track.is_active ? <Badge variant="outline">Active Track</Badge> : null}
            </div>
            <div className="space-y-1">
              <CardTitle className="text-xl">{track.name}</CardTitle>
              <p className="text-sm text-muted-foreground">{getTrackDescription(track)}</p>
            </div>
          </div>
          <Button variant="outline" size="sm" onClick={onOpenMemory} className="gap-1.5">
            <Database className="h-4 w-4" />
            Open Memory Drawer
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-3 md:grid-cols-3">
          {statItems.map((item) => {
            const Icon = item.icon
            return (
              <div
                key={item.label}
                className="rounded-xl border border-border/70 bg-muted/30 px-3 py-3"
              >
                <div className="flex items-center gap-2 text-xs uppercase tracking-[0.14em] text-muted-foreground">
                  <Icon className="h-3.5 w-3.5" />
                  {item.label}
                </div>
                <div className="mt-2 text-2xl font-semibold tracking-tight">{item.value}</div>
              </div>
            )
          })}
        </div>

        {keywords.length > 0 ? (
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="outline" className="gap-1.5">
              <Tags className="h-3.5 w-3.5" />
              Track Keywords
            </Badge>
            {keywords.slice(0, 6).map((keyword) => (
              <Badge key={keyword} variant="secondary">
                {keyword}
              </Badge>
            ))}
          </div>
        ) : null}

        <Separator />

        <div className="grid gap-4 lg:grid-cols-[1.4fr,1fr]">
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <CheckSquare2 className="h-4 w-4 text-muted-foreground" />
              <h3 className="text-sm font-medium">Recent Track Work</h3>
            </div>
            <div className="grid gap-3 md:grid-cols-2">
              <div className="rounded-xl border border-border/70 bg-background/80 p-3">
                <p className="text-xs uppercase tracking-[0.14em] text-muted-foreground">Tasks</p>
                <div className="mt-2 space-y-2">
                  {context.tasks.length > 0 ? (
                    context.tasks.slice(0, 3).map((task) => (
                      <div key={task.id} className="rounded-lg bg-muted/40 px-2.5 py-2">
                        <div className="text-sm font-medium">{task.title}</div>
                        <div className="mt-1 text-xs text-muted-foreground">
                          Status {task.status || "todo"}
                        </div>
                      </div>
                    ))
                  ) : (
                    <p className="text-sm text-muted-foreground">No tasks yet for this track.</p>
                  )}
                </div>
              </div>

              <div className="rounded-xl border border-border/70 bg-background/80 p-3">
                <p className="text-xs uppercase tracking-[0.14em] text-muted-foreground">Milestones</p>
                <div className="mt-2 space-y-2">
                  {context.milestones.length > 0 ? (
                    context.milestones.slice(0, 3).map((milestone) => (
                      <div key={milestone.id} className="rounded-lg bg-muted/40 px-2.5 py-2">
                        <div className="text-sm font-medium">{milestone.name}</div>
                        <div className="mt-1 text-xs text-muted-foreground">
                          Status {milestone.status || "todo"}
                        </div>
                      </div>
                    ))
                  ) : (
                    <p className="text-sm text-muted-foreground">No milestones yet for this track.</p>
                  )}
                </div>
              </div>
            </div>
          </div>

          <div className="space-y-3 rounded-xl border border-border/70 bg-background/80 p-3">
            <p className="text-xs uppercase tracking-[0.14em] text-muted-foreground">Memory & Feedback Shape</p>
            <div className="grid gap-2 sm:grid-cols-2">
              <div className="rounded-lg bg-muted/40 px-2.5 py-2">
                <div className="text-xs text-muted-foreground">Approved Memory</div>
                <div className="mt-1 text-lg font-semibold">{context.memory.approved_items}</div>
              </div>
              <div className="rounded-lg bg-muted/40 px-2.5 py-2">
                <div className="text-xs text-muted-foreground">Feedback Coverage</div>
                <div className="mt-1 text-lg font-semibold">
                  {Math.round(Number(context.eval_summary.feedback_coverage || 0) * 100)}%
                </div>
              </div>
            </div>
            <div className="space-y-2">
              <div className="text-xs text-muted-foreground">Top Memory Tags</div>
              <div className="flex flex-wrap gap-2">
                {memoryTags.length > 0 ? (
                  memoryTags.map((tag) => (
                    <Badge key={tag} variant="outline">
                      {tag}
                    </Badge>
                  ))
                ) : (
                  <span className="text-sm text-muted-foreground">No track-scoped memory tags yet.</span>
                )}
              </div>
            </div>
            <div className="space-y-2">
              <div className="text-xs text-muted-foreground">Effective Feedback Mix</div>
              <div className="flex flex-wrap gap-2">
                {Object.entries(context.feedback.actions).length > 0 ? (
                  Object.entries(context.feedback.actions).map(([action, count]) => (
                    <Badge key={action} variant="secondary">
                      {action}: {count}
                    </Badge>
                  ))
                ) : (
                  <span className="text-sm text-muted-foreground">No effective feedback recorded yet.</span>
                )}
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
