"use client"

import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Trophy, ArrowUpRight } from "lucide-react"
import { Activity } from "@/lib/types"

interface MilestoneCardProps {
    activity: Activity
}

export function MilestoneCard({ activity }: MilestoneCardProps) {
    if (!activity.milestone) return null

    return (
        <Card className="bg-gradient-to-r from-emerald-50 to-teal-50 dark:from-emerald-950/20 dark:to-teal-950/20 border-emerald-100 dark:border-emerald-900">
            <CardContent className="pt-6 relative overflow-hidden">
                {/* Decoration */}
                <Trophy className="absolute -right-4 -top-4 h-24 w-24 text-emerald-100 dark:text-emerald-900/50 rotate-12" />

                <div className="flex gap-4 relative z-10">
                    <div className="flex h-10 w-10 items-center justify-center rounded-full bg-emerald-100 text-emerald-600 dark:bg-emerald-900 dark:text-emerald-400 shrink-0">
                        <ArrowUpRight className="h-6 w-6" />
                    </div>
                    <div className="space-y-2">
                        <h4 className="font-bold text-base text-foreground">
                            {activity.milestone.title}
                        </h4>
                        <p className="text-sm text-muted-foreground leading-relaxed">
                            {activity.milestone.description}
                        </p>
                        <Button size="sm" className="bg-emerald-600 hover:bg-emerald-700 text-white border-none mt-2">
                            See details
                        </Button>
                    </div>
                </div>
            </CardContent>
        </Card>
    )
}
