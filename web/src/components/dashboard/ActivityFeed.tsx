import type { Activity } from "@/lib/types"
import { fetchActivities } from "@/lib/api"
import { auth } from "@/auth"

import { NewPaperCard } from "./feed/NewPaperCard"
import { MilestoneCard } from "./feed/MilestoneCard"
import { ConferenceCard } from "./feed/ConferenceCard"

interface ActivityFeedProps {
  activities?: Activity[]
  maxItems?: number
  showTitle?: boolean
}

export async function ActivityFeed({ activities, maxItems = 6, showTitle = true }: ActivityFeedProps) {
  const items = activities ?? (await (async () => {
    const session = await auth()
    const accessToken = (session as any)?.accessToken as string | undefined
    return fetchActivities(accessToken)
  })())
  const visible = items.slice(0, maxItems)

  return (
    <div className="space-y-3">
      {showTitle ? <h3 className="text-sm font-semibold tracking-tight">Recent Activity</h3> : null}
      {visible.length === 0 ? (
        <p className="py-5 text-center text-xs text-muted-foreground">
          No recent activity. Start by searching papers or running a harvest.
        </p>
      ) : (
        <div className="space-y-2">
          {visible.map((activity) => {
            switch (activity.type) {
              case "published":
                return <NewPaperCard key={activity.id} activity={activity} />
              case "milestone":
                return <MilestoneCard key={activity.id} activity={activity} />
              case "conference":
                return <ConferenceCard key={activity.id} activity={activity} />
              default:
                return null
            }
          })}
        </div>
      )}
    </div>
  )
}
