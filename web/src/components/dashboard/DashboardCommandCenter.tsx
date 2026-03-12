import Link from "next/link"
import {
  ArrowRight,
  BookOpen,
  CalendarClock,
  FlaskConical,
  Settings2,
  Workflow,
  type LucideIcon,
} from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

interface DashboardCommandCenterProps {
  trackCount: number
  activeTrackName?: string
  savedCount: number
  runningPipelines: number
  deadlineCount: number
  tokenCalls: number
  activeTrackId?: number | null
}

interface ModuleTile {
  title: string
  description: string
  href: string
  icon: LucideIcon
  metric: string
  tone: string
}

export function DashboardCommandCenter({
  trackCount,
  activeTrackName,
  savedCount,
  runningPipelines,
  deadlineCount,
  tokenCalls,
  activeTrackId,
}: DashboardCommandCenterProps) {
  const moduleTiles: ModuleTile[] = [
    {
      title: "Research",
      description: activeTrackName
        ? `Active track: ${activeTrackName}`
        : "Create your first track and start search",
      href: activeTrackId ? `/research?track_id=${activeTrackId}` : "/research",
      icon: FlaskConical,
      metric: `${trackCount} tracks`,
      tone: "text-blue-600 bg-blue-500/10",
    },
    {
      title: "Papers Library",
      description: "Saved papers and export-ready references",
      href: "/papers",
      icon: BookOpen,
      metric: `${savedCount} saved`,
      tone: "text-emerald-600 bg-emerald-500/10",
    },
    {
      title: "Workflows",
      description: "Open the full Search, DailyPaper and analysis workbench",
      href: "/workflows",
      icon: Workflow,
      metric: `${runningPipelines} running`,
      tone: "text-violet-600 bg-violet-500/10",
    },
    {
      title: "Settings",
      description: "Providers, model routing and delivery channels",
      href: "/settings",
      icon: Settings2,
      metric: `${tokenCalls} LLM calls`,
      tone: "text-amber-600 bg-amber-500/10",
    },
  ]

  return (
    <Card className="border-border/60">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between gap-3">
          <CardTitle className="text-base">Command Center</CardTitle>
          <Badge variant="secondary" className="gap-1">
            <CalendarClock className="h-3.5 w-3.5" />
            {deadlineCount} upcoming deadlines
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="grid gap-3 md:grid-cols-2">
        {moduleTiles.map((tile) => (
          <Link
            key={tile.title}
            href={tile.href}
            className="group rounded-xl border bg-card p-4 transition hover:border-primary/40 hover:bg-muted/40"
          >
            <div className="flex items-start justify-between gap-3">
              <div className="space-y-1">
                <p className="text-sm font-semibold">{tile.title}</p>
                <p className="text-xs text-muted-foreground">{tile.description}</p>
              </div>
              <div className={`rounded-lg p-2 ${tile.tone}`}>
                <tile.icon className="h-4 w-4" />
              </div>
            </div>
            <div className="mt-3 flex items-center justify-between text-xs">
              <span className="text-muted-foreground">{tile.metric}</span>
              <span className="inline-flex items-center gap-1 font-medium text-primary">
                Open
                <ArrowRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5" />
              </span>
            </div>
          </Link>
        ))}
      </CardContent>
    </Card>
  )
}
