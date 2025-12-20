import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { fetchActivities } from "@/lib/api"
import { NewPaperCard } from "./feed/NewPaperCard"
import { MilestoneCard } from "./feed/MilestoneCard"
import { ConferenceCard } from "./feed/ConferenceCard"

export async function ActivityFeed() {
    const activities = await fetchActivities()

    return (
        <div className="col-span-4 lg:col-span-3 lg:row-span-2 space-y-6">
            <div className="flex items-center justify-between">
                <h3 className="text-xl font-semibold tracking-tight">Your Feed</h3>
                {/* Potentially add feed filters here */}
            </div>

            <div className="space-y-4">
                {activities.map((activity) => {
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
        </div>
    )
}
