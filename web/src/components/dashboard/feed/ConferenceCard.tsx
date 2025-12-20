"use client"

import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { CalendarClock, MapPin } from "lucide-react"
import { Activity } from "@/lib/types"
import { Progress } from "@/components/ui/progress"

interface ConferenceCardProps {
    activity: Activity
}

export function ConferenceCard({ activity }: ConferenceCardProps) {
    if (!activity.conference) return null

    return (
        <Card className="border-orange-100 dark:border-orange-900/50">
            <CardContent className="pt-6">
                <div className="flex gap-4">
                    <div className="flex h-12 w-12 flex-col items-center justify-center rounded-lg bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400 shrink-0 border border-orange-200 dark:border-orange-800">
                        <CalendarClock className="h-6 w-6" />
                    </div>
                    <div className="flex-1 space-y-1">
                        <h4 className="font-bold text-base">
                            Conference Deadline Reminder: {activity.conference.name}
                        </h4>
                        <div className="text-sm text-muted-foreground flex items-center gap-2">
                            <MapPin className="h-3 w-3" /> {activity.conference.location}
                            <span>•</span>
                            <span>{activity.conference.date}</span>
                        </div>
                        <p className="text-sm font-medium text-orange-600 dark:text-orange-400 mt-2">
                            Submission deadline in {activity.conference.deadline_countdown}.
                        </p>
                        <Progress value={75} className="h-2 mt-2 bg-orange-100 dark:bg-orange-950 [&>div]:bg-orange-500" />

                        <div className="flex gap-2 mt-4">
                            <Button size="sm" className="bg-orange-500 hover:bg-orange-600 text-white">
                                View conference
                            </Button>
                            <Button size="sm" variant="ghost">
                                Dismiss
                            </Button>
                        </div>
                    </div>
                </div>
            </CardContent>
        </Card>
    )
}
