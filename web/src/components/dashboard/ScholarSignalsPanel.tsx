import Link from "next/link"
import { ArrowRight, Users, UserRoundCheck } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { Scholar } from "@/lib/types"

type ScholarSignalsPanelProps = {
  scholars: Scholar[]
}

export function ScholarSignalsPanel({ scholars }: ScholarSignalsPanelProps) {
  const activeCount = scholars.filter((item) => item.status === "active").length
  const topItems = [...scholars]
    .sort((a, b) => {
      if (a.status !== b.status) {
        return a.status === "active" ? -1 : 1
      }
      return (b.cached_papers || 0) - (a.cached_papers || 0)
    })
    .slice(0, 5)

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between gap-2">
          <div>
            <CardTitle className="text-base">Scholar Signals</CardTitle>
            <p className="mt-1 text-xs text-muted-foreground">
              Watchlist health and latest scholar activity
            </p>
          </div>
          <Badge variant="outline" className="gap-1">
            <Users className="h-3 w-3" />
            {scholars.length}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-3">
        <div className="grid grid-cols-2 gap-2">
          <div className="rounded-lg border bg-muted/25 p-2.5">
            <p className="text-[11px] text-muted-foreground">Active</p>
            <p className="text-lg font-semibold">{activeCount}</p>
          </div>
          <div className="rounded-lg border bg-muted/25 p-2.5">
            <p className="text-[11px] text-muted-foreground">Idle</p>
            <p className="text-lg font-semibold">{Math.max(0, scholars.length - activeCount)}</p>
          </div>
        </div>

        {!topItems.length ? (
          <div className="rounded-lg border border-dashed p-3 text-xs text-muted-foreground">
            No scholars tracked yet. Add scholar subscriptions to power trend and network signals.
          </div>
        ) : (
          <div className="space-y-2">
            {topItems.map((scholar) => (
              <div key={scholar.id} className="rounded-lg border p-2.5">
                <div className="flex items-center justify-between gap-2">
                  <p className="truncate text-sm font-medium">{scholar.name}</p>
                  <Badge variant={scholar.status === "active" ? "default" : "secondary"} className="capitalize">
                    {scholar.status}
                  </Badge>
                </div>
                <p className="mt-1 truncate text-xs text-muted-foreground">{scholar.recent_activity}</p>
                <div className="mt-1.5 flex flex-wrap gap-1">
                  {(scholar.keywords || []).slice(0, 2).map((keyword) => (
                    <Badge key={`${scholar.id}-${keyword}`} variant="outline" className="text-[10px]">
                      {keyword}
                    </Badge>
                  ))}
                  <span className="text-[10px] text-muted-foreground">
                    cached {scholar.cached_papers || 0}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}

        <div className="flex gap-2">
          <Button asChild size="sm" className="flex-1">
            <Link href="/scholars">
              Open Scholars
              <ArrowRight className="ml-1 h-3.5 w-3.5" />
            </Link>
          </Button>
          <Button asChild size="sm" variant="outline" className="flex-1">
            <Link href="/settings">
              <UserRoundCheck className="mr-1 h-3.5 w-3.5" />
              Manage
            </Link>
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
